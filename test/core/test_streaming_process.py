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


# === Threaded loop ===


def test_threaded_node_streams_generator_output_incrementally() -> None:
    receiver = ZMQMailbox()
    sender_mailbox = ZMQMailbox()
    sender_mailbox.add_publisher("receiver", receiver.consume_port)
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
