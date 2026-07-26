"""Tests for the unified per-node metrics registry and its wiring (Stage 12, #14).

Each test uses a unique ``node`` label so Prometheus's process-global registry (shared
across the whole test run) can't leak values between tests.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import pytest
from prometheus_client import REGISTRY

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging.types import BaseMessage
from neudc.core.policy import ErrorAction, ErrorPolicy, ErrorPolicyConfig

_counter = itertools.count()


class _IdleThreadedNode(BaseThreadedNode):
    """Threaded node that never receives anything; used only for lifecycle assertions."""

    def _collect_data(self) -> Any:
        time.sleep(0.01)
        return None

    def process(self, *args: Any, **kwargs: Any) -> Any:
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _IdleThreadedNode:
        return cls(config["mailbox"])


def _node_id(prefix: str) -> str:
    """Build a fresh, collision-free node label for a test."""
    return f"{prefix}-{next(_counter)}"


def _sample(name: str, node: str) -> float | None:
    return REGISTRY.get_sample_value(name, {"node": node})


def test_error_policy_records_processed_and_latency() -> None:
    node_id = _node_id("processed")
    policy = ErrorPolicy(node_id=node_id)

    policy.execute(lambda item: item, "payload")

    assert _sample("neudc_node_messages_processed_total", node_id) == 1
    assert _sample("neudc_node_process_latency_seconds_count", node_id) == 1


def test_error_policy_records_errors_and_drop_on_skip() -> None:
    node_id = _node_id("skip")
    policy = ErrorPolicy(config=ErrorPolicyConfig(on_error=ErrorAction.SKIP), node_id=node_id)

    def boom(_item: object) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    policy.execute(boom, "payload")

    assert _sample("neudc_node_errors_total", node_id) == 1
    assert _sample("neudc_node_dropped_total", node_id) == 1


def test_error_policy_records_retries() -> None:
    node_id = _node_id("retry")
    policy = ErrorPolicy(
        config=ErrorPolicyConfig(on_error=ErrorAction.RETRY, max_retries=2, initial_backoff=0.001, jitter=False),
        node_id=node_id,
    )
    calls = {"n": 0}

    def flaky(_item: object) -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            msg = "transient"
            raise RuntimeError(msg)
        return "ok"

    result = policy.execute(flaky, "payload", wait=lambda _delay: False)

    assert result == "ok"
    assert _sample("neudc_node_retries_total", node_id) == 1
    assert _sample("neudc_node_messages_processed_total", node_id) == 1


def test_error_policy_records_failures_on_fail_policy() -> None:
    node_id = _node_id("fail")
    policy = ErrorPolicy(config=ErrorPolicyConfig(on_error=ErrorAction.FAIL), node_id=node_id)

    def boom(_item: object) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(Exception, match="boom"):
        policy.execute(boom, "payload")

    assert _sample("neudc_node_failures_total", node_id) == 1


def test_mailbox_reports_queue_hwm_on_construction() -> None:
    mailbox = ZMQMailbox(name=_node_id("hwm"))
    try:
        assert _sample("neudc_node_queue_hwm", mailbox.name) == pytest.approx(1000)
    finally:
        mailbox.stop()


def test_mailbox_reports_queue_depth_after_receive() -> None:
    node_id = _node_id("depth")
    receiver = ZMQMailbox(name=node_id)
    sender = ZMQMailbox(name=_node_id("depth-sender"))
    try:
        sender.add_publisher(receiver.consume_port)
        sender.send(BaseMessage(timestamp=0.0, source="test"))

        message = receiver.receive(timeout=2.0)
        assert message is not None
        # The gauge is updated on both enqueue (receiver thread) and dequeue (this call);
        # after draining the single message the depth must read back down to zero.
        assert _sample("neudc_node_queue_depth", node_id) == 0
    finally:
        sender.stop()
        receiver.stop()


def test_threaded_node_health_gauge_reflects_lifecycle() -> None:
    node_id = _node_id("health")
    mailbox = ZMQMailbox(name=node_id)
    node = _IdleThreadedNode(mailbox, id=node_id)
    try:
        node.start()
        deadline = time.time() + 2.0
        while time.time() < deadline and _sample("neudc_node_health", node_id) != 1:
            time.sleep(0.01)
        assert _sample("neudc_node_health", node_id) == 1
    finally:
        node.stop()

    assert _sample("neudc_node_health", node_id) == 0
