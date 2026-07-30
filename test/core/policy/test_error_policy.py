"""Tests for the per-node error policy (Stage 9, #11).

Before this, an exception in ``process()`` was logged and the message vanished. The
policy makes that a choice: skip (the old behaviour), retry with backoff, or fail the
pipeline — and records what was lost either way.
"""

from __future__ import annotations

import itertools
import time

import pytest
from pydantic import ValidationError

from neudc.core.policy import ErrorAction, ErrorPolicy, ErrorPolicyConfig, NodeFailure


class _BoomError(RuntimeError):
    """Distinct exception type so tests assert on the real failure, not any failure."""


def _failing(times: int):
    """Return a callable that raises _BoomError for the first ``times`` calls, then echoes."""
    state = {"calls": 0}

    def func(item):
        state["calls"] += 1
        if state["calls"] <= times:
            msg = f"attempt {state['calls']}"
            raise _BoomError(msg)
        return item

    func.state = state  # type: ignore[attr-defined]
    return func


def _no_wait(_delay: float) -> bool:
    """Stand-in for the node's stop event: never sleeps, never asks to stop."""
    return False


def _recording_wait(delays: list[float]):
    """Interruptible-wait stub that records each backoff delay instead of sleeping."""

    def wait(delay: float) -> bool:
        delays.append(delay)
        return False

    return wait


# === Defaults: the historical behaviour ===


def test_default_policy_skips_and_does_not_retry() -> None:
    policy = ErrorPolicy(node_id="n")
    func = _failing(times=99)

    assert policy.execute(func, "msg", wait=_no_wait) is None

    assert func.state["calls"] == 1  # skip means one attempt, no retry
    assert policy.stats.errors == 1
    assert policy.stats.dropped == 1
    assert policy.stats.retries == 0


def test_successful_call_returns_result_and_counts_it() -> None:
    policy = ErrorPolicy(node_id="n")

    assert policy.execute(lambda item: item.upper(), "msg", wait=_no_wait) == "MSG"
    assert policy.stats.processed == 1
    assert policy.stats.errors == 0


# === Retry ===


def test_retry_recovers_from_a_transient_failure() -> None:
    policy = ErrorPolicy.from_config({"on_error": "retry", "max_retries": 3}, node_id="n")
    func = _failing(times=2)

    assert policy.execute(func, "msg", wait=_no_wait) == "msg"

    assert func.state["calls"] == 3  # two failures, then success
    assert policy.stats.retries == 2
    assert policy.stats.processed == 1
    assert policy.stats.dropped == 0


def test_retry_gives_up_after_max_retries() -> None:
    policy = ErrorPolicy.from_config({"on_error": "retry", "max_retries": 2}, node_id="n")
    func = _failing(times=99)

    assert policy.execute(func, "msg", wait=_no_wait) is None

    assert func.state["calls"] == 3  # the initial attempt plus two retries
    assert policy.stats.errors == 3
    assert policy.stats.dropped == 1


def test_backoff_grows_exponentially_and_is_capped() -> None:
    delays: list[float] = []
    policy = ErrorPolicy.from_config(
        {
            "on_error": "retry",
            "max_retries": 5,
            "initial_backoff": 0.1,
            "backoff_multiplier": 3.0,
            "max_backoff": 1.0,
            "jitter": False,
        },
        node_id="n",
    )

    policy.execute(_failing(times=99), "msg", wait=_recording_wait(delays))

    assert delays == pytest.approx([0.1, 0.3, 0.9, 1.0, 1.0])  # 3x each step, then clamped


def test_jitter_keeps_delays_in_the_upper_half_of_the_interval() -> None:
    delays: list[float] = []
    policy = ErrorPolicy.from_config(
        {"on_error": "retry", "max_retries": 8, "initial_backoff": 1.0, "backoff_multiplier": 1.0, "jitter": True},
        node_id="n",
    )

    policy.execute(_failing(times=99), "msg", wait=_recording_wait(delays))

    # Equal jitter: spread enough to break lockstep, never collapsing towards zero.
    assert all(0.5 <= delay <= 1.0 for delay in delays), delays
    assert len(set(delays)) > 1, "jitter should vary the delay"


def test_stop_during_backoff_abandons_the_message_immediately() -> None:
    # A node shutting down must not be held hostage by a long retry backoff.
    policy = ErrorPolicy.from_config({"on_error": "retry", "max_retries": 5, "max_backoff": 30.0}, node_id="n")
    func = _failing(times=99)

    assert policy.execute(func, "msg", wait=lambda _delay: True) is None

    assert func.state["calls"] == 1  # stopped during the first backoff, never retried
    assert policy.stats.dropped == 1


# === Fail ===


def test_fail_policy_raises_node_failure() -> None:
    policy = ErrorPolicy.from_config({"on_error": "fail"}, node_id="detector")

    with pytest.raises(NodeFailure) as excinfo:
        policy.execute(_failing(times=99), "msg", wait=_no_wait)

    assert "detector" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, _BoomError)
    assert policy.stats.failures == 1


def test_fail_policy_lets_successful_messages_through() -> None:
    policy = ErrorPolicy.from_config({"on_error": "fail"}, node_id="n")

    assert policy.execute(lambda item: item, "msg", wait=_no_wait) == "msg"
    assert policy.stats.failures == 0


# === Dead-letter wiring ===


def test_dead_letter_records_the_dropped_message(tmp_path) -> None:
    policy = ErrorPolicy.from_config(
        {"on_error": "skip", "dead_letter_dir": str(tmp_path)},
        node_id="detector",
    )

    policy.execute(_failing(times=99), "msg", wait=_no_wait)

    assert (tmp_path / "detector.jsonl").exists()
    assert policy.stats.dead_lettered == 1


def test_no_dead_letter_dir_means_no_files(tmp_path) -> None:
    policy = ErrorPolicy(node_id="n")

    policy.execute(_failing(times=99), "msg", wait=_no_wait)

    assert list(tmp_path.iterdir()) == []
    assert policy.stats.dead_lettered == 0


# === Config validation (fails fast, before anything starts) ===


def test_retry_without_retries_is_rejected() -> None:
    with pytest.raises(ValidationError, match="max_retries"):
        ErrorPolicyConfig(on_error=ErrorAction.RETRY, max_retries=0)


def test_max_backoff_below_initial_is_rejected() -> None:
    with pytest.raises(ValidationError, match="max_backoff"):
        ErrorPolicyConfig(initial_backoff=5.0, max_backoff=1.0)


def test_store_payload_without_a_directory_is_rejected() -> None:
    with pytest.raises(ValidationError, match="dead_letter_dir"):
        ErrorPolicyConfig(store_payload=True)


def test_unknown_key_is_rejected() -> None:
    # A typo in the YAML must fail loudly rather than be silently ignored.
    with pytest.raises(ValidationError):
        ErrorPolicyConfig(on_erro="retry")


def test_negative_retries_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ErrorPolicyConfig(max_retries=-1)


# === Streaming process() as a generator (#37) ===


def test_generator_delivers_items_incrementally_not_accumulated() -> None:
    # The whole point of #37: each yield must reach on_item as it is produced, not
    # all at once after the generator finishes. A sleep between yields makes the two
    # cases distinguishable by wall-clock timing.
    delay = 0.05

    def stream(item):
        for i in range(3):
            time.sleep(delay)
            yield f"{item}-{i}"

    policy = ErrorPolicy(node_id="n")
    timestamps: list[float] = []
    result = policy.execute(stream, "msg", wait=_no_wait, on_item=lambda _v: timestamps.append(time.monotonic()))

    assert result is None  # everything already went to on_item
    assert len(timestamps) == 3
    # Each item arrives roughly `delay` after the previous one, not all bunched at the end.
    gaps = [b - a for a, b in itertools.pairwise(timestamps)]
    assert all(gap >= delay * 0.5 for gap in gaps), gaps
    assert policy.stats.processed == 1  # one input handled, not one per yielded item


def test_generator_without_on_item_collects_into_a_list() -> None:
    def stream(item):
        yield f"{item}-a"
        yield f"{item}-b"

    policy = ErrorPolicy(node_id="n")

    assert policy.execute(stream, "msg", wait=_no_wait) == ["msg-a", "msg-b"]


def test_generator_failure_before_first_yield_is_retried() -> None:
    calls = {"n": 0}

    def flaky(item):
        calls["n"] += 1
        if calls["n"] == 1:
            msg = "boom before any yield"
            raise _BoomError(msg)
        yield item

    policy = ErrorPolicy.from_config({"on_error": "retry", "max_retries": 2}, node_id="n")
    collected: list = []

    result = policy.execute(flaky, "msg", wait=_no_wait, on_item=collected.append)

    assert result is None
    assert collected == ["msg"]
    assert calls["n"] == 2  # one failed attempt, one successful retry
    assert policy.stats.retries == 1
    assert policy.stats.dropped == 0


def test_generator_failure_after_a_yield_is_never_retried() -> None:
    calls = {"n": 0}

    def partial_then_boom(item):
        calls["n"] += 1
        yield f"{item}-partial"
        msg = "boom mid-stream"
        raise _BoomError(msg)

    policy = ErrorPolicy.from_config({"on_error": "retry", "max_retries": 5}, node_id="n")
    collected: list = []

    result = policy.execute(partial_then_boom, "msg", wait=_no_wait, on_item=collected.append)

    assert result is None
    assert collected == ["msg-partial"]  # already-yielded item was still delivered
    assert calls["n"] == 1  # never retried, despite max_retries=5
    assert policy.stats.retries == 0
    assert policy.stats.dropped == 1


def test_generator_failure_after_a_yield_still_dead_letters(tmp_path) -> None:
    def partial_then_boom(item):
        yield f"{item}-partial"
        msg = "boom mid-stream"
        raise _BoomError(msg)

    policy = ErrorPolicy.from_config(
        {"on_error": "retry", "max_retries": 3, "dead_letter_dir": str(tmp_path)},
        node_id="detector",
    )

    policy.execute(partial_then_boom, "msg", wait=_no_wait, on_item=lambda _v: None)

    assert (tmp_path / "detector.jsonl").exists()
    assert policy.stats.dead_lettered == 1


# === Cancellation mid-stream (#41) ===


def test_is_cancelled_closes_the_generator_before_the_next_yield() -> None:
    calls = {"n": 0}

    def stream(item):
        for i in range(5):
            calls["n"] += 1
            yield i

    policy = ErrorPolicy(node_id="n")
    collected: list = []

    def is_cancelled() -> bool:
        return len(collected) >= 2

    result = policy.execute(stream, "msg", wait=_no_wait, on_item=collected.append, is_cancelled=is_cancelled)

    assert result is None
    assert collected == [0, 1]  # the tail (2, 3, 4) was never emitted
    assert calls["n"] == 2  # the generator was never resumed for a 3rd item
    # Benign, expected termination: not an error, not dead-lettered, not a failure.
    assert policy.stats.errors == 0
    assert policy.stats.dropped == 0
    assert policy.stats.failures == 0
    assert policy.stats.processed == 1


def test_cancellation_calls_generator_close_letting_it_clean_up() -> None:
    cleanup = {"ran": False}

    def stream(item):
        try:
            yield "a"
            yield "b"
            yield "c"
        finally:
            cleanup["ran"] = True

    policy = ErrorPolicy(node_id="n")
    collected: list = []

    result = policy.execute(
        stream,
        "msg",
        wait=_no_wait,
        on_item=collected.append,
        is_cancelled=lambda: len(collected) >= 1,
    )

    assert result is None
    assert collected == ["a"]
    assert cleanup["ran"] is True  # gen.close() raised GeneratorExit, the finally ran


def test_is_cancelled_true_before_the_first_yield_emits_nothing() -> None:
    def stream(item):
        yield "should never be reached"

    policy = ErrorPolicy(node_id="n")
    collected: list = []

    result = policy.execute(stream, "msg", wait=_no_wait, on_item=collected.append, is_cancelled=lambda: True)

    assert result is None
    assert collected == []
    assert policy.stats.errors == 0
    assert policy.stats.failures == 0


def test_is_cancelled_never_returning_true_behaves_like_no_cancellation() -> None:
    def stream(item):
        yield "a"
        yield "b"

    policy = ErrorPolicy(node_id="n")

    assert policy.execute(stream, "msg", wait=_no_wait, is_cancelled=lambda: False) == ["a", "b"]


def test_omitting_is_cancelled_preserves_existing_generator_behaviour() -> None:
    # Default None must not change anything for callers that predate #41.
    def stream(item):
        yield "a"
        yield "b"

    policy = ErrorPolicy(node_id="n")

    assert policy.execute(stream, "msg", wait=_no_wait) == ["a", "b"]


def test_generator_respects_fail_policy() -> None:
    def boom_immediately(item):
        msg = "boom"
        raise _BoomError(msg)
        yield item  # unreachable on purpose, keeps this a generator function

    policy = ErrorPolicy.from_config({"on_error": "fail"}, node_id="detector")

    with pytest.raises(NodeFailure):
        policy.execute(boom_immediately, "msg", wait=_no_wait, on_item=lambda _v: None)

    assert policy.stats.failures == 1
