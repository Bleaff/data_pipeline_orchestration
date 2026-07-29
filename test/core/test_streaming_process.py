"""Streaming process() (#37) as the node run loops actually use it.

Mirrors ``test_node_error_policy.py``'s pattern for #12: both loops go through the same
``_handle``, so these tests pin the shared path rather than one loop's own copy.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

from neudc.core.base.base_process import BaseProcessNode
from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging.types import BaseMessage, ControlAction, ControlMessage


class _StreamThreadedNode(BaseThreadedNode):
    """Threaded node whose process() yields N items, spaced out, for exactly one input."""

    def __init__(self, mailbox: Any, n: int = 3, delay: float = 0.05) -> None:
        super().__init__(mailbox, id="stream")
        self.n = n
        self.delay = delay
        self._sent = False

    def _collect_data(self) -> Any:
        if self._sent:
            time.sleep(0.05)
            return None
        self._sent = True
        return "msg"

    def process(self, item: Any) -> Any:
        for i in range(self.n):
            time.sleep(self.delay)
            yield f"{item}-{i}"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _StreamThreadedNode:
        return cls(config["mailbox"])


class _StreamProcessNode(BaseProcessNode):
    """Process node used for parent-side _handle assertions only (no spawn, see #12 test notes)."""

    def process(self, item: Any) -> Any:
        yield f"{item}-0"
        yield f"{item}-1"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _StreamProcessNode:
        return cls(config["mailbox"])


def _drain_with_timestamps(mailbox: ZMQMailbox, expected: int, timeout: float = 5.0) -> list[tuple[Any, float]]:
    received: list[tuple[Any, float]] = []
    deadline = time.time() + timeout
    while len(received) < expected and time.time() < deadline:
        msg = mailbox.receive(timeout=0.1)
        if msg is not None:
            received.append((msg, time.monotonic()))
    return received


def _drain_all(mailbox: ZMQMailbox, timeout: float) -> list[Any]:
    """Drain whatever arrives within the window, without a target count."""
    received: list[Any] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = mailbox.receive(timeout=0.1)
        if msg is not None:
            received.append(msg)
    return received


# === Threaded loop ===


def test_threaded_node_streams_generator_output_incrementally() -> None:
    receiver = ZMQMailbox()
    sender_mailbox = ZMQMailbox()
    sender_mailbox.add_publisher(receiver.consume_port)
    node = _StreamThreadedNode(sender_mailbox, n=3, delay=0.05)
    try:
        node.start()
        got = _drain_with_timestamps(receiver, 3)

        assert [msg for msg, _ in got] == ["msg-0", "msg-1", "msg-2"]
        gaps = [t2 - t1 for (_, t1), (_, t2) in itertools.pairwise(got)]
        assert all(gap >= 0.02 for gap in gaps), gaps  # arrived spread out, not bunched
    finally:
        node.stop()
        receiver.stop()


# === Process loop (parent side, no spawn — same convention as #12's tests) ===


def test_process_node_handle_streams_via_on_item() -> None:
    node = _StreamProcessNode(ZMQMailbox(), id="stream")
    collected: list[Any] = []

    result = node._handle("msg", on_item=collected.append)

    assert result is None  # already streamed, nothing left for the loop to send
    assert collected == ["msg-0", "msg-1"]


def test_process_node_handle_without_on_item_returns_collected_list() -> None:
    node = _StreamProcessNode(ZMQMailbox(), id="stream")

    assert node._handle("msg") == ["msg-0", "msg-1"]


# === Turn cancellation / barge-in (#41) ===


class _CancellableStreamNode(BaseThreadedNode):
    """Threaded node whose single input yields many spaced-out items for one turn.

    Each yielded output carries the same (session_id, turn_id) as the input, mirroring
    a real streaming node (e.g. token-by-token LLM output for one turn).
    """

    def __init__(self, mailbox: Any, session_id: str = "s-1", turn_id: int = 1, n: int = 20, delay: float = 0.05):
        super().__init__(mailbox, id="cancellable-stream")
        self.session_id = session_id
        self.turn_id = turn_id
        self.n = n
        self.delay = delay
        self._sent = False

    def _collect_data(self) -> Any:
        if self._sent:
            time.sleep(0.05)
            return None
        self._sent = True
        return BaseMessage(timestamp=0.0, session_id=self.session_id, turn_id=self.turn_id)

    def process(self, item: Any) -> Any:
        for i in range(self.n):
            time.sleep(self.delay)
            yield BaseMessage(timestamp=float(i), session_id=self.session_id, turn_id=self.turn_id)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _CancellableStreamNode:
        return cls(config["mailbox"])


def test_cancellation_interrupts_the_generator_mid_stream_and_drops_the_tail() -> None:
    """Issue #41, criterion 2.

    A cancel signal stops a running generator at its current yield, without killing
    the node's thread/loop and without emitting the rest of the (now-cancelled) turn.
    """
    receiver = ZMQMailbox()
    node_mailbox = ZMQMailbox()
    node_mailbox.add_publisher(receiver.consume_port)

    controller = ZMQMailbox()
    controller.add_control_publisher("node", node_mailbox.control_consume_port)

    node = _CancellableStreamNode(node_mailbox, session_id="s-1", turn_id=1, n=20, delay=0.05)
    try:
        node.start()
        # Let a couple of items through before cancelling.
        time.sleep(0.15)
        controller.send_control(
            ControlMessage(timestamp=999.0, action=ControlAction.CANCEL, session_id="s-1", turn_id=1)
        )

        got = _drain_all(receiver, timeout=1.5)

        assert 0 < len(got) < 20, "expected a partial, interrupted turn, not the full stream"
        # Not killed: the node's own loop/thread is still alive and considers itself running.
        assert node.is_running is True
        assert node.thread is not None
        assert node.thread.is_alive()

        # No further items arrive afterwards - the tail of the cancelled turn is gone.
        more = _drain_all(receiver, timeout=0.3)
        assert more == []
    finally:
        node.stop()
        receiver.stop()
        controller.stop()


def test_uncancelled_turn_streams_in_full_unaffected() -> None:
    # Regression: a node with a wired-but-silent control channel must behave exactly
    # like the plain streaming case in test_threaded_node_streams_generator_output_incrementally.
    receiver = ZMQMailbox()
    node_mailbox = ZMQMailbox()
    node_mailbox.add_publisher(receiver.consume_port)
    node = _CancellableStreamNode(node_mailbox, session_id="s-2", turn_id=2, n=4, delay=0.02)
    try:
        node.start()
        got = _drain_with_timestamps(receiver, 4, timeout=5.0)

        assert len(got) == 4
        assert [msg.timestamp for msg, _ in got] == [0.0, 1.0, 2.0, 3.0]
    finally:
        node.stop()
        receiver.stop()
