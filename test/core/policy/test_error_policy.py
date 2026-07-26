"""Tests for the per-node error policy (Stage 9, #11).

Before this, an exception in ``process()`` was logged and the message vanished. The
policy makes that a choice: skip (the old behaviour), retry with backoff, or fail the
pipeline — and records what was lost either way.
"""

from __future__ import annotations

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
