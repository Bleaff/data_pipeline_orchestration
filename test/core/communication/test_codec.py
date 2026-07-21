"""Tests for the message codec and fan-out serialize-once behaviour (Stage 5)."""

from __future__ import annotations

import time

import numpy as np

from neudc.core.communication.mailbox import zmq_mailbox
from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging import codec
from neudc.core.communication.messaging.types import Box, Frame


def _drain(mailbox: ZMQMailbox, expected: int, timeout: float = 5.0) -> list:
    received: list = []
    deadline = time.time() + timeout
    while len(received) < expected and time.time() < deadline:
        msg = mailbox.receive(timeout=0.1)
        if msg is not None:
            received.append(msg)
    return received


def test_roundtrip_plain_object() -> None:
    obj = {"a": 1, "b": [1, 2, 3], "c": "text"}
    assert codec.loads(codec.dumps(obj)) == obj


def test_roundtrip_frame_with_image_and_boxes() -> None:
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    frame = Frame(
        image=img,
        timestamp=1.5,
        source_frame="src.png",
        frame_id=7,
        boxes=[Box(x1=1, y1=2, x2=3, y2=4, score=0.5, class_id_name="cat", class_id_num=0)],
        frame_id_last=10,
        drop=False,
    )
    out = codec.loads(codec.dumps(frame))

    assert np.array_equal(out.image, img)
    assert out.frame_id == 7
    assert out.source_frame == "src.png"
    assert out.boxes[0].class_id_name == "cat"
    assert out.boxes[0].class_id_num == 0


def test_uses_pickle_protocol_5() -> None:
    # Wire framing: a leading plain tag byte, then a protocol-5 pickle stream (which
    # starts with the PROTO opcode for version 5).
    raw = codec.dumps({"x": 1})
    assert raw[0] == 0  # _TAG_PLAIN
    assert raw[1:3] == b"\x80\x05"


def test_fanout_serializes_once_and_delivers_to_all(monkeypatch) -> None:
    calls = {"n": 0}
    real_dumps = codec.dumps

    def counting_dumps(obj, use_shm=False):
        calls["n"] += 1
        return real_dumps(obj, use_shm=use_shm)

    monkeypatch.setattr(zmq_mailbox.codec, "dumps", counting_dumps)

    sender = ZMQMailbox()
    receiver_a = ZMQMailbox()
    receiver_b = ZMQMailbox()
    sender.add_publisher(receiver_a.consume_port)
    sender.add_publisher(receiver_b.consume_port)

    sender.send({"i": 42})

    # One serialization for a two-edge fan-out, not one per edge.
    assert calls["n"] == 1
    assert _drain(receiver_a, 1) == [{"i": 42}]
    assert _drain(receiver_b, 1) == [{"i": 42}]

    sender.stop()
    receiver_a.stop()
    receiver_b.stop()
