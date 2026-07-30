"""Tests for the priority control channel (#41): turn cancellation / barge-in.

A `ControlMessage` (e.g. a cancel signal) must reach its target over a channel that
is entirely independent of the data FIFO, so it is not stuck behind a backlog it is
meant to interrupt. These tests mirror the existing mailbox/push-pull test style
(`test_mailbox.py`, `test_push_pull.py`, `test_base_message.py`).
"""

from __future__ import annotations

import time

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging import codec
from neudc.core.communication.messaging.types import BaseMessage, ControlAction, ControlMessage


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# === ControlMessage / ControlAction round-trip ===


def test_control_message_is_a_base_message_with_an_action() -> None:
    msg = ControlMessage(timestamp=1.0, action=ControlAction.CANCEL, session_id="s-1", turn_id=3)

    assert isinstance(msg, BaseMessage)
    assert msg.action is ControlAction.CANCEL
    assert msg.session_id == "s-1"
    assert msg.turn_id == 3


def test_control_message_roundtrips_through_the_codec() -> None:
    msg = ControlMessage(timestamp=2.0, source="controller", action=ControlAction.CANCEL, session_id="s-2", turn_id=7)

    out = codec.loads(codec.dumps(msg))

    assert type(out) is ControlMessage
    assert out.action is ControlAction.CANCEL
    assert out.session_id == "s-2"
    assert out.turn_id == 7


def test_control_message_requires_an_action() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ControlMessage(timestamp=1.0)


# === Mailbox control channel: independent transport ===


def test_control_channel_is_a_distinct_port_from_the_data_channel() -> None:
    mailbox = ZMQMailbox()
    try:
        assert mailbox.control_consume_port != mailbox.consume_port
    finally:
        mailbox.stop()


def test_add_control_publisher_and_send_control() -> None:
    controller = ZMQMailbox()
    target = ZMQMailbox()
    try:
        controller.add_control_publisher("target", target.control_consume_port)

        controller.send_control(ControlMessage(timestamp=1.0, action=ControlAction.CANCEL, session_id="s", turn_id=1))

        assert _wait_until(lambda: target.is_cancelled("s", 1))
    finally:
        controller.stop()
        target.stop()


def test_is_cancelled_is_false_for_an_unrelated_turn() -> None:
    controller = ZMQMailbox()
    target = ZMQMailbox()
    try:
        controller.add_control_publisher("target", target.control_consume_port)
        controller.send_control(ControlMessage(timestamp=1.0, action=ControlAction.CANCEL, session_id="s", turn_id=1))

        assert _wait_until(lambda: target.is_cancelled("s", 1))
        # A different turn on the same or another session was never cancelled.
        assert target.is_cancelled("s", 2) is False
        assert target.is_cancelled("other", 1) is False
    finally:
        controller.stop()
        target.stop()


def test_no_control_publisher_means_never_cancelled() -> None:
    # A mailbox with no control channel wired up (the common case) must behave exactly
    # as it did before this feature existed: is_cancelled is always False.
    mailbox = ZMQMailbox()
    try:
        assert mailbox.is_cancelled(None, None) is False
        assert mailbox.is_cancelled("any", 1) is False
    finally:
        mailbox.stop()


# === The core "done when" criterion: control beats a backlogged data queue ===


def test_control_signal_reaches_target_before_backlogged_data_queue_drains() -> None:
    """A cancel must not wait behind a backlog on the data path (issue #41, criterion 1).

    The data queue is filled well past its capacity and never drained (`receive()` is
    never called). The control channel is a completely separate socket/thread, so the
    cancellation must still be observed via `is_cancelled` without touching the data
    queue at all.
    """
    sender = ZMQMailbox()
    receiver = ZMQMailbox(message_queue_size=5)
    try:
        sender.add_publisher(receiver.consume_port)
        sender.add_control_publisher("receiver", receiver.control_consume_port)

        # Flood the data path; the receiver never calls receive(), so this backs up
        # both the bounded local queue and the socket behind it.
        for i in range(30):
            sender.send(BaseMessage(timestamp=float(i)))

        # Give the data receiver loop time to actually become backlogged.
        assert _wait_until(lambda: receiver._message_queue.qsize() >= 5, timeout=3.0)

        # The control channel is independent: the cancel signal must still arrive.
        sender.send_control(ControlMessage(timestamp=999.0, action=ControlAction.CANCEL, session_id="s", turn_id=1))
        assert _wait_until(lambda: receiver.is_cancelled("s", 1), timeout=2.0)

        # The data backlog was never drained in the meantime - proves the control
        # path did not have to wait for (or go through) the data queue.
        assert receiver._message_queue.qsize() >= 5
    finally:
        sender.stop()
        receiver.stop()


# === Pickling (BaseProcessNode fork boundary) ===


def test_getstate_includes_control_channel_state() -> None:
    mailbox = ZMQMailbox()
    try:
        state = mailbox.__getstate__()

        assert state["control_consume_port"] == mailbox.control_consume_port
        assert state["control_pub_sockets"] == {}
        assert state["control_sub_queue"] is None
        assert state["_control_thread"] is None
    finally:
        mailbox.stop()


def test_from_state_rebuilds_a_working_control_channel() -> None:
    receiver = ZMQMailbox()
    control_port = receiver.control_consume_port
    state = receiver.__getstate__()
    receiver.stop()  # explicit hand-off, same convention as the data channel

    rebuilt = ZMQMailbox.from_state(state)
    try:
        assert rebuilt.control_consume_port == control_port

        controller = ZMQMailbox()
        try:
            controller.add_control_publisher("rebuilt", rebuilt.control_consume_port)
            controller.send_control(
                ControlMessage(timestamp=1.0, action=ControlAction.CANCEL, session_id="s", turn_id=5)
            )

            assert _wait_until(lambda: rebuilt.is_cancelled("s", 5))
        finally:
            controller.stop()
    finally:
        rebuilt.stop()


def test_getstate_does_not_mutate_control_channel_either() -> None:
    mailbox = ZMQMailbox()
    try:
        assert mailbox._running is True
        mailbox.__getstate__()
        # Pure serializer: pickling must not stop the control thread/socket either.
        assert mailbox._running is True
        assert mailbox._control_thread is not None
        assert mailbox._control_thread.is_alive()
    finally:
        mailbox.stop()
