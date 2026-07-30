"""Transport tests for the PUSH/PULL ZeroQueue (Stage 4).

The defining property vs the old PUB/SUB transport: messages sent right after
connect (before the consumer is draining) are queued and delivered in order,
never silently dropped.
"""

from __future__ import annotations

import time

import zmq

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.zero_queue import ZeroQueuePub, ZeroQueueSub


def test_sockets_are_push_pull() -> None:
    sub = ZeroQueueSub()
    pub = ZeroQueuePub(port=sub.port)
    try:
        assert sub.socket_sub.type == zmq.PULL
        assert pub.socket_pub.type == zmq.PUSH
    finally:
        pub.stop()
        sub.stop()


def test_immediately_sent_messages_are_not_dropped() -> None:
    """PUB/SUB dropped early messages (slow joiner); PUSH/PULL must keep them all, in order."""
    sub = ZeroQueueSub()
    pub = ZeroQueuePub(port=sub.port)
    n = 50
    for i in range(n):
        pub.put({"i": i})

    received: list = []
    deadline = time.time() + 5
    while len(received) < n and time.time() < deadline:
        msg = sub.get(timeout=0.1)
        if msg is not None:
            received.append(msg)

    assert [m["i"] for m in received] == list(range(n))  # FIFO, no loss
    pub.stop()
    sub.stop()


def test_mailbox_delivers_in_order_without_loss() -> None:
    sender = ZMQMailbox()
    receiver = ZMQMailbox()
    sender.add_publisher("receiver", receiver.consume_port)

    n = 200
    for i in range(n):
        sender.send({"i": i})

    received: list = []
    deadline = time.time() + 8
    while len(received) < n and time.time() < deadline:
        msg = receiver.receive(timeout=0.1)
        if msg is not None:
            received.append(msg)

    assert [m["i"] for m in received] == list(range(n))
    sender.stop()
    receiver.stop()
