"""Regression tests for #40.

Both run loops (`BaseNode._run` and `BaseProcessNode.run`) used to gate the send with
`if result:` instead of `if result is not None:`. That is harmless today only because
none of the current message types are falsy when non-empty-but-real (see below), but it
would silently swallow a legitimately empty-but-valid result the moment one exists — no
send, no log, no metric. The fix routes both loops through a shared `BaseNode._should_send`
helper (mirroring how `_handle` is already shared to avoid the #12 kind of drift), which
treats `None` as "drop" and an explicitly empty `Batch` as "drop", and forwards everything
else regardless of truthiness.

Mirrors the patterns already established in test_streaming_process.py: a real
`BaseThreadedNode` + `ZMQMailbox` pair for the threaded loop, and direct `_handle`/
`_should_send`/`_send_item` calls (no spawn) for the process loop, per
test_process_node_lifecycle.py's convention of avoiding a real child process in tests.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from neudc.core.base.base_process import BaseProcessNode
from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging.types import Batch, Frame


def _drain_all(mailbox: ZMQMailbox, timeout: float) -> list[Any]:
    """Drain whatever arrives within the window, without a target count."""
    received: list[Any] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = mailbox.receive(timeout=0.1)
        if msg is not None:
            received.append(msg)
    return received


def _make_frame(frame_id: int = 0) -> Frame:
    return Frame(
        timestamp=0.0,
        image=np.zeros((1, 1, 3), dtype=np.uint8),
        source_frame="test",
        frame_id=frame_id,
        boxes=[],
        frame_id_last=frame_id,
    )


class _EchoProcessNode(BaseProcessNode):
    """Minimal process node whose process() just returns whatever it was given."""

    def process(self, item: Any) -> Any:
        return item

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _EchoProcessNode:
        return cls(config["mailbox"])


# === `_should_send` unit coverage ===


def test_should_send_is_false_for_none() -> None:
    node = _EchoProcessNode(ZMQMailbox(), id="echo")
    assert node._should_send(None) is False


def test_should_send_is_true_for_falsy_but_real_results() -> None:
    node = _EchoProcessNode(ZMQMailbox(), id="echo")
    assert node._should_send("") is True
    assert node._should_send([]) is True
    assert node._should_send(0) is True


def test_should_send_withholds_empty_batch_explicitly() -> None:
    node = _EchoProcessNode(ZMQMailbox(), id="echo")
    assert node._should_send(Batch(frames=[])) is False


def test_should_send_allows_non_empty_batch() -> None:
    node = _EchoProcessNode(ZMQMailbox(), id="echo")
    assert node._should_send(Batch(frames=[_make_frame()])) is True


# === Threaded loop (`BaseNode._run`) end-to-end ===


class _FalsyResultThreadedNode(BaseThreadedNode):
    """Threaded node whose single process() call returns a falsy-but-real result."""

    def __init__(self, mailbox: Any) -> None:
        super().__init__(mailbox, id="falsy-result")
        self._sent = False

    def _collect_data(self) -> Any:
        if self._sent:
            time.sleep(0.05)
            return None
        self._sent = True
        return "msg"

    def process(self, item: Any) -> Any:
        # Falsy but not None: stands in for e.g. a zero-sample AudioChunk or an empty
        # TextChunk, should either type ever grow a __len__/__bool__ override (#40).
        return ""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _FalsyResultThreadedNode:
        return cls(config["mailbox"])


def test_threaded_loop_sends_falsy_but_not_none_result() -> None:
    """Fails against the old `if result:` check: "" is falsy, so it would never arrive."""
    receiver = ZMQMailbox()
    sender_mailbox = ZMQMailbox()
    sender_mailbox.add_publisher("receiver", receiver.consume_port)
    node = _FalsyResultThreadedNode(sender_mailbox)
    try:
        node.start()
        got = _drain_all(receiver, timeout=1.0)
        assert got == [""]
    finally:
        node.stop()
        receiver.stop()


class _EmptyBatchThreadedNode(BaseThreadedNode):
    """Threaded node whose single process() call returns an explicitly empty Batch."""

    def __init__(self, mailbox: Any) -> None:
        super().__init__(mailbox, id="empty-batch")
        self._sent = False

    def _collect_data(self) -> Any:
        if self._sent:
            time.sleep(0.05)
            return None
        self._sent = True
        return "msg"

    def process(self, item: Any) -> Any:
        return Batch(frames=[])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _EmptyBatchThreadedNode:
        return cls(config["mailbox"])


def test_threaded_loop_does_not_send_empty_batch() -> None:
    """Regression: an empty Batch must still be withheld, now checked by length."""
    receiver = ZMQMailbox()
    sender_mailbox = ZMQMailbox()
    sender_mailbox.add_publisher("receiver", receiver.consume_port)
    node = _EmptyBatchThreadedNode(sender_mailbox)
    try:
        node.start()
        got = _drain_all(receiver, timeout=0.5)
        assert got == []
    finally:
        node.stop()
        receiver.stop()


# === Process loop (`BaseProcessNode.run`), parent side, no spawn ===
#
# `run()` itself is never invoked directly in this test suite (spawn + locally-defined
# classes is inherently flaky, see test_process_node_lifecycle.py's module docstring).
# These tests instead exercise `run()`'s per-iteration body -- `_handle` -> `_should_send`
# -> `_send_item` -- directly against a real ZMQMailbox pair, the same convention used by
# test_streaming_process.py's process-loop tests.


def test_process_loop_body_sends_falsy_but_not_none_handle_result() -> None:
    """Fails against the old `if result:` check: "" is falsy, so it would never arrive."""
    receiver = ZMQMailbox()
    node_mailbox = ZMQMailbox()
    node_mailbox.add_publisher("receiver", receiver.consume_port)
    node = _EchoProcessNode(node_mailbox, id="echo")
    # Parent-side __init__ already stopped `node_mailbox` (mirrors real spawn hand-off,
    # see test_process_node_lifecycle.py); rebind it here, same as `run()` does via
    # `_start_afterwords()`, without spawning a child process.
    node._start_afterwords()
    try:
        result = node._handle("")
        assert node._should_send(result) is True
        node._send_item(result)

        got = _drain_all(receiver, timeout=1.0)
        assert got == [""]
    finally:
        node.stop_event.set()
        node.mailbox.stop()
        receiver.stop()


def test_process_loop_body_does_not_send_empty_batch() -> None:
    receiver = ZMQMailbox()
    node_mailbox = ZMQMailbox()
    node_mailbox.add_publisher("receiver", receiver.consume_port)
    node = _EchoProcessNode(node_mailbox, id="echo")
    node._start_afterwords()
    try:
        result = node._handle(Batch(frames=[]))
        assert node._should_send(result) is False

        got = _drain_all(receiver, timeout=0.3)
        assert got == []
    finally:
        node.stop_event.set()
        node.mailbox.stop()
        receiver.stop()
