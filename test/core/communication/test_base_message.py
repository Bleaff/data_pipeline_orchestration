"""Tests for the BaseMessage envelope and the generic Batch (#33).

The point of these tests is that the graph is no longer tied to ``Frame``: a
non-CV payload must survive the codec, the shared-memory transport and a real
mailbox edge without any of them knowing what it is. ``Frame`` keeps working
unchanged — it is now just one subclass among others.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest
from pydantic import ValidationError

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging import codec
from neudc.core.communication.messaging.types import BaseMessage, Batch, Box, Frame

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shared memory only")


class SamplesMessage(BaseMessage):
    """Stand-in for a non-CV payload; the real Text/Audio/Token schemas land in #34."""

    samples: np.ndarray


def _make_frame(img: np.ndarray | None = None) -> Frame:
    if img is None:
        img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    return Frame(
        image=img,
        timestamp=1.5,
        source_frame="src.png",
        frame_id=7,
        boxes=[Box(x1=1, y1=2, x2=3, y2=4, score=0.5, class_id_name="cat", class_id_num=0)],
        frame_id_last=10,
        drop=False,
    )


def _drain(mailbox: ZMQMailbox, expected: int, timeout: float = 5.0) -> list:
    received: list = []
    deadline = time.time() + timeout
    while len(received) < expected and time.time() < deadline:
        msg = mailbox.receive(timeout=0.1)
        if msg is not None:
            received.append(msg)
    return received


# === The envelope itself ===


def test_frame_is_a_base_message_with_inherited_defaults() -> None:
    frame = _make_frame()

    assert isinstance(frame, BaseMessage)
    # Fields that moved up to the envelope keep working through Frame's own constructor.
    assert frame.timestamp == 1.5
    assert frame.drop is False
    # Streaming identity defaults to "standalone message", which is what the
    # labelling pipeline produces.
    assert frame.source == ""
    assert frame.session_id is None
    assert frame.turn_id is None
    assert frame.is_final is True


def test_base_message_requires_a_timestamp() -> None:
    with pytest.raises(ValidationError):
        BaseMessage()


def test_frame_still_requires_its_cv_fields() -> None:
    # Generalising the envelope must not make Frame lax: an image-less Frame is
    # still a broken frame, even though a payload-less BaseMessage is fine.
    with pytest.raises(ValidationError):
        Frame(timestamp=1.0, source_frame="a.png", frame_id=1, boxes=[], frame_id_last=1)


# === Codec ===


def test_base_message_roundtrip_carries_streaming_fields() -> None:
    msg = BaseMessage(timestamp=2.0, source="mic", session_id="s-1", turn_id=3, is_final=False)

    out = codec.loads(codec.dumps(msg))

    assert out.source == "mic"
    assert out.session_id == "s-1"
    assert out.turn_id == 3
    assert out.is_final is False


def test_roundtrip_preserves_the_concrete_subclass() -> None:
    # The wire must not flatten a subclass into the envelope: the receiving node
    # gets back exactly the type the sender built.
    out = codec.loads(codec.dumps(_make_frame()))

    assert type(out) is Frame
    assert out.source_frame == "src.png"
    assert out.boxes[0].class_id_name == "cat"


def test_non_frame_payload_roundtrips() -> None:
    samples = np.arange(128, dtype=np.int16)
    out = codec.loads(codec.dumps(SamplesMessage(timestamp=1.0, samples=samples)))

    assert type(out) is SamplesMessage
    assert np.array_equal(out.samples, samples)


# === Batch ===


def test_batch_keeps_message_subclass_instances() -> None:
    # A Batch typed by the envelope must not coerce its items down to BaseMessage —
    # that would silently drop the pixels on the way into batched inference.
    frame = _make_frame()

    batch = Batch(frames=[frame])

    assert type(batch[0]) is Frame
    assert np.array_equal(batch[0].image, frame.image)
    assert len(batch) == 1
    assert [type(item) for item in batch] == [Frame]


def test_parametrized_batch_validates_and_is_recognised_as_a_batch() -> None:
    # Batch[Frame] is what nodes annotate; the mailbox must still see it as a Batch
    # (that isinstance check is what makes it unroll instead of sending it whole).
    batch = Batch[Frame](frames=[_make_frame()])

    assert isinstance(batch, Batch)
    assert type(batch[0]) is Frame
    assert batch[0].frame_id == 7


def test_unparametrized_batch_roundtrips_through_the_codec() -> None:
    # Batches stay process-local — the mailbox unrolls them — so only the
    # unparametrized form needs to survive pickling. See the Batch docstring: a
    # parametrized alias built inside a function is not picklable by pydantic.
    out = codec.loads(codec.dumps(Batch(frames=[_make_frame()])))

    assert isinstance(out, Batch)
    assert type(out[0]) is Frame
    assert out[0].frame_id == 7


def test_batch_of_non_frame_messages() -> None:
    batch = Batch(frames=[SamplesMessage(timestamp=float(i), samples=np.zeros(4)) for i in range(3)])

    assert len(batch) == 3
    assert all(type(item) is SamplesMessage for item in batch)


# === Over a real mailbox edge ===


def test_mailbox_delivers_a_non_frame_message() -> None:
    sender = ZMQMailbox()
    receiver = ZMQMailbox()
    sender.add_publisher("receiver", receiver.consume_port)
    try:
        samples = np.arange(64, dtype=np.int16)
        sender.send(SamplesMessage(timestamp=1.0, source="mic", samples=samples))

        got = _drain(receiver, 1)
        assert len(got) == 1
        assert type(got[0]) is SamplesMessage
        assert got[0].source == "mic"
        assert np.array_equal(got[0].samples, samples)
    finally:
        sender.stop()
        receiver.stop()


def test_mailbox_unrolls_a_batch_of_non_frame_messages() -> None:
    sender = ZMQMailbox()
    receiver = ZMQMailbox()
    sender.add_publisher("receiver", receiver.consume_port)
    try:
        sender.send(Batch(frames=[SamplesMessage(timestamp=float(i), samples=np.full(4, i)) for i in range(3)]))

        got = _drain(receiver, 3)
        # A batch arrives as its items, one message per edge write.
        assert [type(m) for m in got] == [SamplesMessage] * 3
        assert sorted(m.timestamp for m in got) == [0.0, 1.0, 2.0]
    finally:
        sender.stop()
        receiver.stop()


# === Shared-memory transport (#10) needs no change for the new base type ===


@posix_only
def test_shm_transport_carries_a_non_frame_payload() -> None:
    # Comfortably above SHM_MIN_BYTES (64 KiB) so the buffer goes out-of-band.
    samples = np.arange(64 * 1024, dtype=np.int16)
    raw = codec.dumps(SamplesMessage(timestamp=1.0, samples=samples), use_shm=True)

    assert raw[0] == 1  # _TAG_SHM: the payload really took the shared-memory path
    out = codec.loads(raw)

    assert type(out) is SamplesMessage
    assert np.array_equal(out.samples, samples)
    # Same contract as Frame.image: nodes mutate buffers in place, so the copy
    # handed to the consumer must be writable.
    assert out.samples.flags.writeable
    out.samples[0] = 42
    assert out.samples[0] == 42
