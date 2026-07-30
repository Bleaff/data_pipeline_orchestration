"""Tests for the non-CV payload schemas (#34): TextChunk, TokenTensor, AudioChunk, VideoSegment.

These mirror the ``SamplesMessage``-based patterns in ``test_base_message.py`` (#33):
the codec, ``Batch`` and the mailbox are already payload-agnostic, so the only new
ground to cover is that each concrete schema round-trips correctly, including through
the shared-memory transport for the ones carrying large buffers.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging import codec
from neudc.core.communication.messaging.types import AudioChunk, Batch, TextChunk, TokenTensor, VideoSegment

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shared memory only")


def _drain(mailbox: ZMQMailbox, expected: int, timeout: float = 5.0) -> list:
    received: list = []
    deadline = time.time() + timeout
    while len(received) < expected and time.time() < deadline:
        msg = mailbox.receive(timeout=0.1)
        if msg is not None:
            received.append(msg)
    return received


# === TextChunk ===


def test_text_chunk_roundtrips() -> None:
    chunk = TextChunk(timestamp=1.0, source="asr", text="hello world", language="en")

    out = codec.loads(codec.dumps(chunk))

    assert type(out) is TextChunk
    assert out.text == "hello world"
    assert out.language == "en"


def test_batch_of_text_chunks() -> None:
    batch = Batch(frames=[TextChunk(timestamp=float(i), text=f"chunk {i}") for i in range(3)])

    assert len(batch) == 3
    assert [item.text for item in batch] == ["chunk 0", "chunk 1", "chunk 2"]


# === TokenTensor ===


def test_token_tensor_roundtrips() -> None:
    data = np.arange(128, dtype=np.int64)

    out = codec.loads(codec.dumps(TokenTensor(timestamp=1.0, data=data)))

    assert type(out) is TokenTensor
    assert np.array_equal(out.data, data)


@posix_only
def test_shm_transport_carries_a_token_tensor() -> None:
    data = np.arange(64 * 1024, dtype=np.int64)  # comfortably above SHM_MIN_BYTES
    raw = codec.dumps(TokenTensor(timestamp=1.0, data=data), use_shm=True)

    assert raw[0] == 1  # _TAG_SHM
    out = codec.loads(raw)

    assert np.array_equal(out.data, data)
    assert out.data.flags.writeable
    out.data[0] = 42
    assert out.data[0] == 42


# === AudioChunk ===


def test_audio_chunk_roundtrips() -> None:
    samples = np.zeros(256, dtype=np.float32)

    out = codec.loads(codec.dumps(AudioChunk(timestamp=1.0, samples=samples, sample_rate=16000, channels=2)))

    assert type(out) is AudioChunk
    assert np.array_equal(out.samples, samples)
    assert out.sample_rate == 16000
    assert out.channels == 2


@posix_only
def test_shm_transport_carries_an_audio_chunk() -> None:
    samples = np.arange(64 * 1024, dtype=np.int16)  # comfortably above SHM_MIN_BYTES
    raw = codec.dumps(AudioChunk(timestamp=1.0, samples=samples, sample_rate=16000), use_shm=True)

    assert raw[0] == 1  # _TAG_SHM
    out = codec.loads(raw)

    assert np.array_equal(out.samples, samples)
    assert out.samples.flags.writeable
    out.samples[0] = 42
    assert out.samples[0] == 42


def test_mailbox_delivers_an_audio_chunk() -> None:
    sender = ZMQMailbox()
    receiver = ZMQMailbox()
    sender.add_publisher("receiver", receiver.consume_port)
    try:
        samples = np.arange(64, dtype=np.int16)
        sender.send(AudioChunk(timestamp=1.0, source="mic", samples=samples, sample_rate=16000))

        got = _drain(receiver, 1)
        assert len(got) == 1
        assert type(got[0]) is AudioChunk
        assert got[0].source == "mic"
        assert np.array_equal(got[0].samples, samples)
    finally:
        sender.stop()
        receiver.stop()


# === VideoSegment ===


def test_video_segment_roundtrips() -> None:
    images = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]

    out = codec.loads(codec.dumps(VideoSegment(timestamp=1.0, images=images, start_frame_id=10, fps=30.0)))

    assert type(out) is VideoSegment
    assert len(out.images) == 2
    assert np.array_equal(out.images[0], images[0])
    assert np.array_equal(out.images[1], images[1])
    assert out.start_frame_id == 10
    assert out.fps == 30.0


@posix_only
def test_shm_transport_carries_a_video_segment() -> None:
    # Each image is comfortably above SHM_MIN_BYTES on its own, so both go out-of-band.
    big = (np.arange(256 * 256 * 3, dtype=np.uint8) % 255).reshape(256, 256, 3)
    images = [big, big.copy()]
    raw = codec.dumps(VideoSegment(timestamp=1.0, images=images, start_frame_id=0), use_shm=True)

    assert raw[0] == 1  # _TAG_SHM
    out = codec.loads(raw)

    assert np.array_equal(out.images[0], images[0])
    assert np.array_equal(out.images[1], images[1])
    for img in out.images:
        assert img.flags.writeable
        img[0, 0, 0] = 42
        assert img[0, 0, 0] == 42
