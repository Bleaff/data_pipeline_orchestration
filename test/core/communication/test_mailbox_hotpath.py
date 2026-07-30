"""Hot-path regression tests for ZMQMailbox (Stage 1).

Covers the fixes for:
- receive() crashing on non-Frame payloads (AttributeError on ``.frame_id``);
- the receiver loop dropping a message / injecting ``None`` when the internal
  queue is full;
- removal of the per-message ``time.sleep`` that capped send throughput.
"""

from __future__ import annotations

import time

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox


def _drain(mailbox: ZMQMailbox, expected: int, timeout: float = 5.0) -> list:
    received: list = []
    deadline = time.time() + timeout
    while len(received) < expected and time.time() < deadline:
        msg = mailbox.receive(timeout=0.1)
        if msg is not None:
            received.append(msg)
    return received


def test_receive_dict_payload_does_not_crash() -> None:
    """A plain dict must round-trip without touching Frame-only attributes."""
    sender = ZMQMailbox()
    receiver = ZMQMailbox()
    sender.add_publisher("receiver", receiver.consume_port)

    sender.send({"msg": "hello"})
    got = _drain(receiver, expected=1)

    assert got == [{"msg": "hello"}]
    sender.stop()
    receiver.stop()


def test_full_internal_queue_applies_backpressure_without_none() -> None:
    """With a tiny internal queue, no message is lost and no None is injected."""
    sender = ZMQMailbox()
    receiver = ZMQMailbox(message_queue_size=1)
    sender.add_publisher("receiver", receiver.consume_port)

    n = 5
    for i in range(n):
        sender.send({"i": i})

    received = _drain(receiver, expected=n)

    assert None not in received
    assert sorted(item["i"] for item in received) == list(range(n))
    sender.stop()
    receiver.stop()


def test_send_has_no_per_message_sleep() -> None:
    """Sending N messages must be far faster than the old 1ms-per-message floor."""
    sender = ZMQMailbox()
    receiver = ZMQMailbox()
    sender.add_publisher("receiver", receiver.consume_port)

    n = 300
    start = time.time()
    for i in range(n):
        sender.send({"i": i})
    send_elapsed = time.time() - start

    # The removed sleep alone would cost n * 1ms; assert we are well under half of it.
    assert send_elapsed < n * 0.001 * 0.5

    received = _drain(receiver, expected=n)
    assert len(received) == n
    sender.stop()
    receiver.stop()
