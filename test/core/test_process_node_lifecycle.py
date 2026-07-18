"""Parent-side lifecycle tests for BaseProcessNode (Stage 2).

These cover the construction hand-off and the readiness event without spawning a
child process (spawn + locally-defined classes is inherently flaky). The
cross-process readiness path is exercised indirectly by the multipipe manager test.
"""

from __future__ import annotations

from typing import Any

import pytest

from neudc.core.base.base_node import BaseNode
from neudc.core.base.base_process import BaseProcessNode
from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox


class _EchoProcessNode(BaseProcessNode):
    """Minimal concrete process node used only for construction-time assertions."""

    def process(self, item: Any) -> Any:
        return item


def test_construction_captures_config_and_releases_parent_mailbox() -> None:
    mailbox = ZMQMailbox()
    assert mailbox._running is True

    node = _EchoProcessNode(mailbox, id="echo")

    # A picklable snapshot is captured for the child...
    assert isinstance(node.mailbox_config, dict)
    assert node.mailbox_config["consume_port"] == mailbox.consume_port
    # ...and the parent mailbox is explicitly released so the child can rebind.
    assert mailbox._running is False


def test_ready_event_starts_unset() -> None:
    mailbox = ZMQMailbox()
    node = _EchoProcessNode(mailbox, id="echo")

    assert hasattr(node, "_ready_event")
    assert node._ready_event.is_set() is False
    # Not started yet, so waiting with a zero timeout reports "not ready".
    assert node.wait_ready(timeout=0) is False


def test_health_state_is_shared_not_reassigned() -> None:
    """The child must mutate the shared health Values so the parent's status() is truthful."""
    mailbox = ZMQMailbox()
    node = _EchoProcessNode(mailbox, id="echo")

    assert node.status() is False  # HEALTH_INITIAL before the child marks running
    healthy_id = id(node._healthy)
    time_id = id(node._last_success_time)

    node._mark_running()

    # Same shared objects (mutated, not replaced) — otherwise the parent never sees it.
    assert id(node._healthy) == healthy_id
    assert id(node._last_success_time) == time_id
    assert node.status() is True
    assert node._last_success_time.value > 0.0


def test_start_afterwords_does_not_start_a_second_consumer_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for #12: run()'s own loop is the sole mailbox consumer.

    `_start_afterwords()` used to also call `BaseNode.init_runtime()`, which spun up a
    background thread running `BaseNode._run()` — a second collect/process/send loop
    racing `run()`'s loop for messages on the same mailbox.
    """
    mailbox = ZMQMailbox()
    node = _EchoProcessNode(mailbox, id="echo")

    init_runtime_calls: list[BaseNode] = []
    monkeypatch.setattr(BaseNode, "init_runtime", lambda self: init_runtime_calls.append(self))

    try:
        node._start_afterwords()

        assert init_runtime_calls == []
        # BaseNode.__init__ defaults, untouched because init_runtime() never ran.
        assert node.thread is None
        assert node.is_running is False
        # The mailbox was still rebound and health monitoring still started.
        assert node.mailbox.consume_port == node.mailbox_config["consume_port"]
        assert node._health_thread.is_alive()
    finally:
        node.stop_event.set()
        node.mailbox.stop()
