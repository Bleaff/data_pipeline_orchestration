"""Parent-side lifecycle tests for BaseProcessNode (Stage 2).

These cover the construction hand-off and the readiness event without spawning a
child process (spawn + locally-defined classes is inherently flaky). The
cross-process readiness path is exercised indirectly by the multipipe manager test.

The health-monitor tests below (#39) run the child's own ``run()`` loop directly in a
plain ``threading.Thread`` within the test process -- never via ``node.start()``
(``mp.Process.start``), for the same reason: a locally-defined class cannot be
pickled for ``spawn``. ``run()`` -> ``_start_afterwords()`` rebinding the mailbox
in-process is exactly what ``test_start_afterwords_does_not_start_a_second_consumer_loop``
already proves works without a real OS process.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from neudc.core.base.base_node import BaseNode
from neudc.core.base.base_process import BaseProcessNode
from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox

if TYPE_CHECKING:
    import pytest


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
    # Not inlinable to `init_runtime_calls.append`: that's already a bound method, so setting
    # it directly as a class attribute would not rebind to `self` when called as node.init_runtime().
    monkeypatch.setattr(BaseNode, "init_runtime", lambda self: init_runtime_calls.append(self))  # noqa: PLW0108

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


# === Health monitor: idle vs hung (#39) ===


class _HangingProcessNode(BaseProcessNode):
    """Process node whose process() blocks until released, to simulate a genuine hang.

    ``_collect_data`` is overridden to hand back an item immediately on the loop's
    first pass (bypassing the real mailbox timeout, irrelevant to what's under test
    here) so ``process()`` is entered right away and then never returns until
    ``release`` is set -- freezing ``_last_iteration_time`` exactly like a real stuck
    ``process()`` would.
    """

    def __init__(self, mailbox: Any, id: str = "hang") -> None:
        super().__init__(mailbox, id=id)
        self.release = threading.Event()
        self._served = False

    def _collect_data(self) -> Any:
        if self._served:
            # Only one item is ever handed out; after that process() is permanently
            # blocked below and this is never called again anyway.
            return None
        self._served = True
        return "msg"

    def process(self, item: Any) -> Any:
        self.release.wait(timeout=30)  # safety bound: never hang a leaked thread forever
        return item

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _HangingProcessNode:
        return cls(config["mailbox"])


def test_idle_node_stays_healthy_past_health_timeout() -> None:
    """Done-when criterion 1 (#39): no incoming data for longer than health_timeout stays healthy.

    The node's own run() loop keeps cycling (nothing to collect, ~0.1s mailbox
    receive timeout) and refreshes `_last_iteration_time` every pass, so it never
    goes stale even though nothing was ever received or processed.
    """
    mailbox = ZMQMailbox()
    node = _EchoProcessNode(mailbox, id="echo-idle")
    node.health_timeout = 0.3
    node.health_check_interval = 0.05

    run_thread = threading.Thread(target=node.run, daemon=True)
    run_thread.start()
    try:
        assert node.wait_ready(timeout=2.0), "child never signalled readiness"
        # Comfortably longer than health_timeout, with zero messages ever sent.
        time.sleep(0.8)
        assert node.status() is True
    finally:
        node.stop_event.set()
        run_thread.join(timeout=2.0)
        node.mailbox.stop()


def test_hung_process_is_flagged_unhealthy_after_timeout() -> None:
    """Done-when criterion 2 (#39): a genuinely hung process() is flagged unhealthy.

    Unlike the idle case, the loop cannot return to refresh `_last_iteration_time`
    while stuck inside process(), so the health monitor correctly trips.
    """
    mailbox = ZMQMailbox()
    node = _HangingProcessNode(mailbox, id="echo-hang")
    node.health_timeout = 0.3
    node.health_check_interval = 0.05

    run_thread = threading.Thread(target=node.run, daemon=True)
    run_thread.start()
    try:
        assert node.wait_ready(timeout=2.0), "child never signalled readiness"
        time.sleep(0.8)
        assert node.status() is False
    finally:
        node.release.set()  # let the blocked process() call return so the loop/thread can exit
        node.stop_event.set()
        run_thread.join(timeout=2.0)
        node.mailbox.stop()
