"""Timer / reminder tool: schedules a delayed announcement back into the pipeline."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from neudc.core.communication.messaging.types import TextChunk
from neudc.core.node.processors.mixins.tools.base import ToolError
from neudc.utils import LOGGER

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ("TimerTool",)

#: Upper bound on `seconds` — a sane guard against a runaway or typo'd timer (e.g. the
#: LLM confusing minutes/seconds), not a real product limit.
MAX_TIMER_SECONDS = 24 * 60 * 60


class TimerTool:
    """`set_timer`: schedules a reminder that announces itself when it fires.

    Returns an immediate confirmation string — the call itself must not block the
    node's `process()` for the timer's duration. The actual reminder is delivered
    later, out-of-band, as a new `TextChunk` sent directly through `mailbox`: same
    `session_id` as the turn that scheduled it, `source="timer"` so it's easy to
    recognize downstream, `is_final=True` so `TtsMixin` speaks it as soon as it
    arrives.
    """

    name = "set_timer"
    description = (
        'set_timer({"seconds": <number>, "label": <string, optional>}) - schedules a reminder '
        "that will be spoken after the given number of seconds."
    )

    def __init__(
        self,
        mailbox: Any,
        *,
        session_id: str = "default",
        scheduler: Callable[[float, Callable[[], None]], Any] | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
        ----
            mailbox: Mailbox the owning node sends through — used to deliver the
                reminder asynchronously once the timer fires.
            session_id: Stamped on the reminder `TextChunk` this tool sends later.
            scheduler: `(seconds, callback) -> object with a .start()/.daemon`,
                defaults to `threading.Timer`. Overridable so tests don't need to
                sleep for real.

        """
        self._mailbox = mailbox
        self._session_id = session_id
        self._scheduler = scheduler or threading.Timer

    def __call__(self, seconds: float, label: str = "") -> str:
        """Schedule a reminder, returning an immediate confirmation.

        Args:
        ----
            seconds: Delay before the reminder fires. Must be in `(0, MAX_TIMER_SECONDS]`.
            label: Optional text describing what the reminder is for.

        Returns:
        -------
            A confirmation string, meant to be spliced into the LLM's reply.

        Raises:
        ------
            ToolError: If `seconds` is out of range.

        """
        if not (0 < seconds <= MAX_TIMER_SECONDS):
            msg = f"seconds must be in (0, {MAX_TIMER_SECONDS}], got {seconds}"
            raise ToolError(msg)

        timer = self._scheduler(seconds, lambda: self._fire(label))
        timer.daemon = True
        timer.start()

        suffix = f" ({label})" if label else ""
        return f"Timer set for {seconds:g} seconds{suffix}."

    def _fire(self, label: str) -> None:
        text = f"Reminder: {label}" if label else "Timer finished."
        try:
            self._mailbox.send(
                TextChunk(timestamp=time.time(), source="timer", text=text, session_id=self._session_id),
            )
        except Exception as e:  # noqa: BLE001 - a background timer callback must never crash the process
            LOGGER.exception("Error delivering timer reminder", exc_info=e)
