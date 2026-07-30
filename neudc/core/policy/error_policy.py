"""What a node does when its ``process()`` raises.

Before Stage 9 the answer was always the same: log the exception and move on, so a
transient failure (a busy GPU, a half-written file) silently cost a frame and a
permanent one cost every frame, indistinguishably. The policy makes that choice
explicit and configurable per node:

* ``skip``  — log, dead-letter, drop the message, keep going. The historical
  behaviour and the default, so an untouched config behaves exactly as before.
* ``retry`` — re-run ``process()`` up to ``max_retries`` times with exponential
  backoff; if the last attempt still fails, dead-letter and drop.
* ``fail``  — dead-letter and stop the node, which brings the pipeline down.

Cancellation (#41)
------------------
A generator ``process()`` can also be interrupted mid-stream by an out-of-band signal
(turn cancellation / barge-in), independent of retry/backoff. ``execute``/
``_execute_generator`` accept an optional ``is_cancelled`` callable, checked before
each ``next(gen)``; when it returns True the generator is closed (``gen.close()``,
which raises ``GeneratorExit`` inside it at its current ``yield`` so a well-behaved
generator can clean up) and draining stops. This is **not** an error: it is expected,
benign termination of an interrupted turn, so it is never dead-lettered, never counted
as a failure, and never raises :class:`NodeFailure` — unlike a genuine exception, which
still goes through :meth:`ErrorPolicy._give_up`.

Two properties worth knowing before configuring ``retry``:

* **Only ``process()`` is retried**, never the send that follows it. Re-sending is
  not idempotent — a downstream node would see the message twice.
* **``process()`` must tolerate being called twice on the same message.** Nodes that
  mutate the message in place (``DrawNode`` draws onto ``frame.image``) will retry
  on top of a half-modified message. Use ``skip`` for those unless the work is
  genuinely idempotent.

``process()`` as a generator (#37)
-----------------------------------
A streaming node's ``process()`` may be a generator: each ``yield`` is delivered to
the caller-supplied ``on_item`` callback immediately, instead of accumulating a full
result first. Retry interacts with this differently from the atomic case: **a failure
is only retried if it happens before the first yield.** Once anything has been
delivered for an attempt, re-running ``process()`` from scratch would duplicate that
output downstream, so a mid-stream failure goes straight to the same dead-letter/fail/
drop handling a retry-exhausted message gets, without spending any remaining attempts.
"""

from __future__ import annotations

import inspect
import random
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neudc.core.observability.metrics import (
    DEAD_LETTERED,
    DROPPED,
    ERRORS,
    FAILURES,
    MESSAGES_PROCESSED,
    PROCESS_LATENCY,
    RETRIES,
)
from neudc.core.policy.dead_letter import DeadLetterSink
from neudc.utils import LOGGER

if TYPE_CHECKING:
    from collections.abc import Callable


class ErrorAction(StrEnum):
    """What to do with a message whose processing failed."""

    SKIP = "skip"
    RETRY = "retry"
    FAIL = "fail"


# Public API name (neudc.core.policy.NodeFailure); renaming to *Error would be a breaking change.
class NodeFailure(RuntimeError):  # noqa: N818
    """Raised by the policy when the node must stop rather than continue.

    Only ``ErrorAction.FAIL`` produces it; the node's run loop turns it into a
    stop signal that the pipeline observes.
    """


@dataclass
class ErrorStats:
    """Per-node error counters, ready to be exported as metrics (#14)."""

    processed: int = 0
    errors: int = 0
    retries: int = 0
    dropped: int = 0
    dead_lettered: int = 0
    failures: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return the counters as a plain dict."""
        return asdict(self)


class ErrorPolicyConfig(BaseModel):
    """Per-node ``error_policy`` block from the pipeline YAML."""

    model_config = ConfigDict(extra="forbid")

    on_error: ErrorAction = ErrorAction.SKIP
    max_retries: int = Field(default=0, ge=0)
    initial_backoff: float = Field(default=0.1, gt=0)
    max_backoff: float = Field(default=5.0, gt=0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    jitter: bool = True
    dead_letter_dir: str | None = None
    store_payload: bool = False

    @model_validator(mode="after")
    def _check_consistency(self) -> ErrorPolicyConfig:
        if self.on_error is ErrorAction.RETRY and self.max_retries < 1:
            msg = "on_error: retry requires max_retries >= 1 (otherwise it is just 'skip')"
            raise ValueError(msg)
        if self.max_backoff < self.initial_backoff:
            msg = f"max_backoff ({self.max_backoff}) must be >= initial_backoff ({self.initial_backoff})"
            raise ValueError(msg)
        if self.store_payload and self.dead_letter_dir is None:
            msg = "store_payload requires dead_letter_dir"
            raise ValueError(msg)
        return self


@dataclass
class ErrorPolicy:
    """Applies an :class:`ErrorPolicyConfig` around a node's ``process()``.

    Constructed by the node factory from the node's config and attached to the node;
    the node's run loop calls :meth:`execute` instead of calling ``process()`` directly.
    """

    config: ErrorPolicyConfig = field(default_factory=ErrorPolicyConfig)
    node_id: str = "node"
    stats: ErrorStats = field(default_factory=ErrorStats)
    _sink: DeadLetterSink | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the dead-letter sink if `config.dead_letter_dir` was configured."""
        if self.config.dead_letter_dir is not None:
            self._sink = DeadLetterSink(self.config.dead_letter_dir, store_payload=self.config.store_payload)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None, node_id: str = "node") -> ErrorPolicy:
        """Build a policy from a raw ``error_policy`` mapping (or the defaults if absent)."""
        parsed = ErrorPolicyConfig(**config) if config else ErrorPolicyConfig()
        return cls(config=parsed, node_id=node_id)

    def execute(
        self,
        func: Callable[[Any], Any],
        item: Any,
        wait: Callable[[float], bool] | None = None,
        on_item: Callable[[Any], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Any:
        """Run ``func(item)`` under the policy.

        Args:
        ----
            func: The node's ``process`` (or anything with the same shape). May be a
                generator function (#37): see the module docstring for how retry
                differs in that case.
            item: The message to process.
            wait: Optional interruptible sleep, normally a stop event's ``wait``:
                called with the backoff delay, returning True when the node is
                stopping. Without it the policy sleeps outright, which would make a
                long backoff delay shutdown by that much.
            on_item: Only meaningful when ``func`` is a generator function. Called
                once per yielded value, immediately, so the caller can stream each
                one out (e.g. to the mailbox) without waiting for the generator to
                finish. If omitted, yielded values are collected into a list and
                returned once the generator is exhausted, same as any other value.
            is_cancelled: Only meaningful when ``func`` is a generator function (#41).
                Checked before each ``next(gen)``; when it returns True the generator
                is closed and draining stops without emitting the rest of the turn.
                Defaults to None, which preserves all existing (non-cancellable)
                behaviour unchanged.

        Returns:
        -------
            Whatever ``func`` returned, or None if the message was dropped. For a
            generator ``func`` with ``on_item`` given, always None (everything was
            already handed to ``on_item``); without ``on_item``, the list of yielded
            values.

        Raises:
        ------
            NodeFailure: When the policy is ``fail`` and processing did not succeed.

        """
        if inspect.isgeneratorfunction(func):
            return self._execute_generator(func, item, wait, on_item, is_cancelled)

        attempts = 1 + (self.config.max_retries if self.config.on_error is ErrorAction.RETRY else 0)
        last_error: BaseException | None = None

        for attempt in range(attempts):
            start = time.monotonic()
            try:
                result = func(item)
            except Exception as error:  # noqa: BLE001 - the policy is the handler of last resort
                PROCESS_LATENCY.labels(node=self.node_id).observe(time.monotonic() - start)
                last_error = error
                self.stats.errors += 1
                ERRORS.labels(node=self.node_id).inc()
                remaining = attempts - attempt - 1
                if remaining == 0:
                    break
                delay = self._backoff_delay(attempt)
                self.stats.retries += 1
                RETRIES.labels(node=self.node_id).inc()
                LOGGER.warning(
                    f"[{self.node_id}] {type(error).__name__} in process(), "
                    f"retrying in {delay:.3f}s ({remaining} attempt(s) left): {error}"
                )
                if self._sleep(delay, wait):
                    # Stopping mid-backoff: abandon the message rather than block shutdown.
                    LOGGER.info(f"[{self.node_id}] stop requested during backoff, dropping message")
                    self.stats.dropped += 1
                    DROPPED.labels(node=self.node_id).inc()
                    return None
            else:
                PROCESS_LATENCY.labels(node=self.node_id).observe(time.monotonic() - start)
                self.stats.processed += 1
                MESSAGES_PROCESSED.labels(node=self.node_id).inc()
                return result

        return self._give_up(item, last_error)

    def _execute_generator(
        self,
        func: Callable[[Any], Any],
        item: Any,
        wait: Callable[[float], bool] | None,
        on_item: Callable[[Any], None] | None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Any:
        """Drain a generator-valued ``process()``, streaming each yield out immediately.

        Mirrors :meth:`execute`'s attempt/backoff loop, with one deliberate
        difference: retry is only attempted for a failure *before* the first yield of
        that attempt (see module docstring). A failure after any yield is terminal —
        this attempt already delivered output, so re-running would duplicate it.

        ``is_cancelled`` (#41), when given, is checked before each ``next(gen)``; a
        True result closes the generator and returns immediately, without emitting the
        rest of the turn and without treating this as an error (see module docstring).
        """
        attempts = 1 + (self.config.max_retries if self.config.on_error is ErrorAction.RETRY else 0)
        last_error: BaseException | None = None

        for attempt in range(attempts):
            start = time.monotonic()
            yielded_count = 0
            collected: list[Any] = []
            try:
                gen = func(item)
                while True:
                    if is_cancelled is not None and is_cancelled():
                        LOGGER.debug(f"[{self.node_id}] turn cancelled, closing generator mid-stream")
                        gen.close()
                        PROCESS_LATENCY.labels(node=self.node_id).observe(time.monotonic() - start)
                        self.stats.processed += 1
                        MESSAGES_PROCESSED.labels(node=self.node_id).inc()
                        return None if on_item is not None else collected
                    value = next(gen)
                    yielded_count += 1
                    if on_item is not None:
                        on_item(value)
                    else:
                        collected.append(value)
            except StopIteration:
                PROCESS_LATENCY.labels(node=self.node_id).observe(time.monotonic() - start)
                self.stats.processed += 1
                MESSAGES_PROCESSED.labels(node=self.node_id).inc()
                return None if on_item is not None else collected
            except Exception as error:  # noqa: BLE001 - the policy is the handler of last resort
                PROCESS_LATENCY.labels(node=self.node_id).observe(time.monotonic() - start)
                last_error = error
                self.stats.errors += 1
                ERRORS.labels(node=self.node_id).inc()

                if yielded_count > 0:
                    LOGGER.warning(
                        f"[{self.node_id}] {type(error).__name__} in process() after "
                        f"{yielded_count} item(s) already delivered; not retrying "
                        f"(would duplicate downstream output): {error}"
                    )
                    return self._give_up(item, error)

                remaining = attempts - attempt - 1
                if remaining == 0:
                    break
                delay = self._backoff_delay(attempt)
                self.stats.retries += 1
                RETRIES.labels(node=self.node_id).inc()
                LOGGER.warning(
                    f"[{self.node_id}] {type(error).__name__} in process() before any yield, "
                    f"retrying in {delay:.3f}s ({remaining} attempt(s) left): {error}"
                )
                if self._sleep(delay, wait):
                    LOGGER.info(f"[{self.node_id}] stop requested during backoff, dropping message")
                    self.stats.dropped += 1
                    DROPPED.labels(node=self.node_id).inc()
                    return None

        return self._give_up(item, last_error)

    def _give_up(self, item: Any, error: BaseException | None) -> Any:
        """Record the message as dead-lettered and apply the terminal action."""
        if self._sink is not None:
            self._sink.record(self.node_id, item, error)
            self.stats.dead_lettered += 1
            DEAD_LETTERED.labels(node=self.node_id).inc()

        if self.config.on_error is ErrorAction.FAIL:
            self.stats.failures += 1
            FAILURES.labels(node=self.node_id).inc()
            msg = f"[{self.node_id}] processing failed and error policy is 'fail': {error!r}"
            LOGGER.critical(msg)
            raise NodeFailure(msg) from error

        self.stats.dropped += 1
        DROPPED.labels(node=self.node_id).inc()
        LOGGER.error(f"[{self.node_id}] dropping message after failed processing", exc_info=error)
        return None

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff for a zero-based attempt index, capped at ``max_backoff``.

        With ``jitter`` the delay is spread over the upper half of the interval
        (equal jitter): enough spread to keep several nodes from retrying in
        lockstep, while never collapsing the delay to nearly zero.
        """
        delay = min(self.config.max_backoff, self.config.initial_backoff * self.config.backoff_multiplier**attempt)
        if self.config.jitter:
            delay = delay / 2 + random.uniform(0, delay / 2)  # noqa: S311 - backoff spread, not cryptography
        return delay

    @staticmethod
    def _sleep(delay: float, wait: Callable[[float], bool] | None) -> bool:
        """Wait out the backoff. Returns True if the node was asked to stop meanwhile."""
        if wait is None:
            time.sleep(delay)
            return False
        return bool(wait(delay))
