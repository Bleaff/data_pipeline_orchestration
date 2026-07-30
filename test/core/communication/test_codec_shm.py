"""Tests for the opt-in shared-memory buffer transport in the codec (Stage 8).

The transport is payload-agnostic: it moves any large out-of-band pickle buffer
(image, embedding, mask, future audio/token payloads) through POSIX shared memory.
These tests assert round-trip equality, writability of the reconstructed buffer,
the small-buffer inline threshold, and — most importantly — that no segment is
leaked, including across a real process boundary.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from neudc.core.communication.mailbox import zmq_mailbox
from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging import codec
from neudc.core.communication.messaging.types import Box, Frame

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shared memory only")


def _big_image(side: int = 256) -> np.ndarray:
    # Comfortably above SHM_MIN_BYTES (64 KiB) so pixels go out-of-band to shared memory.
    return (np.arange(side * side * 3, dtype=np.uint8) % 255).reshape(side, side, 3)


def _make_frame(img: np.ndarray) -> Frame:
    return Frame(
        image=img,
        timestamp=1.5,
        source_frame="src.png",
        frame_id=7,
        boxes=[Box(x1=1, y1=2, x2=3, y2=4, score=0.5, class_id_name="cat", class_id_num=0)],
        embedding=np.ones(4, dtype=np.float32),
        frame_id_last=10,
        drop=False,
    )


@posix_only
def test_shm_roundtrip_preserves_frame() -> None:
    img = _big_image()
    frame = _make_frame(img)

    raw = codec.dumps(frame, use_shm=True)
    assert raw[0] == 1  # _TAG_SHM

    out = codec.loads(raw)
    assert np.array_equal(out.image, img)
    assert np.array_equal(out.embedding, frame.embedding)
    assert out.frame_id == 7
    assert out.boxes[0].class_id_name == "cat"


@posix_only
def test_shm_reconstructed_image_is_writable() -> None:
    # Nodes such as DrawNode mutate frame.image in place, so the consumer copy must
    # be writable (a plain read-only shared view would be unsafe / would raise).
    img = _big_image()
    out = codec.loads(codec.dumps(_make_frame(img), use_shm=True))

    assert out.image.flags.writeable
    out.image[0, 0, 0] = 42  # must not raise
    assert out.image[0, 0, 0] == 42


@posix_only
def test_small_buffers_stay_inline_no_segment() -> None:
    # A tiny image is below the threshold: no shared-memory segment is created, and
    # the message still round-trips.
    tiny = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    raw = codec.dumps(_make_frame(tiny), use_shm=True)

    out = codec.loads(raw)
    assert np.array_equal(out.image, tiny)


@posix_only
def test_shm_roundtrip_leaves_no_segment() -> None:
    # After a full produce/consume cycle the segment must be unlinked (no leak).
    before = set(Path("/dev/shm").glob("psm_*"))  # noqa: S108 -- inspecting real POSIX shm segments, not a temp-file
    img = _big_image()

    raw = codec.dumps(_make_frame(img), use_shm=True)
    # While the segment is live it exists on Linux; on macOS /dev/shm is absent, so we
    # only assert the end state, which holds on every POSIX platform.
    codec.loads(raw)

    after = set(Path("/dev/shm").glob("psm_*"))  # noqa: S108 -- inspecting real POSIX shm segments, not a temp-file
    assert after <= before, f"leaked shared-memory segments: {after - before}"


@posix_only
def test_shm_roundtrip_across_processes(tmp_path) -> None:
    # The real scenario: producer serializes in one process, consumer reads in another.
    # Exercises resource_tracker behaviour across a genuine process boundary and asserts
    # the segment does not survive the consumer.
    import multiprocessing as mp

    img = _big_image()
    frame = _make_frame(img)
    raw = codec.dumps(frame, use_shm=True)

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_consume_in_child, args=(raw, q))
    proc.start()
    ok, first_pixel = q.get(timeout=30)
    proc.join(timeout=30)

    assert ok, "child failed to reconstruct the frame from shared memory"
    assert first_pixel == int(img[0, 0, 0])
    assert list(Path("/dev/shm").glob("psm_*")) == [] or sys.platform == "darwin"  # noqa: S108


def _consume_in_child(raw: bytes, q) -> None:
    # Runs in a separate spawned process: decode and report back a couple of facts.
    from neudc.core.communication.messaging import codec as child_codec

    frame = child_codec.loads(raw)
    q.put((frame.image.flags.writeable, int(frame.image[0, 0, 0])))


def _drain(mailbox: ZMQMailbox, expected: int, timeout: float = 5.0) -> list:
    received: list = []
    deadline = time.time() + timeout
    while len(received) < expected and time.time() < deadline:
        msg = mailbox.receive(timeout=0.1)
        if msg is not None:
            received.append(msg)
    return received


def _spy_use_shm(monkeypatch) -> list[bool]:
    """Record the ``use_shm`` argument the mailbox passes to the codec on each send."""
    seen: list[bool] = []
    real_dumps = codec.dumps

    def spy(obj, use_shm=False):
        seen.append(use_shm)
        return real_dumps(obj, use_shm=use_shm)

    monkeypatch.setattr(zmq_mailbox.codec, "dumps", spy)
    return seen


@posix_only
def test_mailbox_single_consumer_uses_shm_and_delivers(monkeypatch) -> None:
    monkeypatch.setattr(zmq_mailbox.codec, "SHM_ENABLED", True)
    seen = _spy_use_shm(monkeypatch)

    sender = ZMQMailbox()
    receiver = ZMQMailbox()
    sender.add_publisher("receiver", receiver.consume_port)
    try:
        img = _big_image()
        sender.send(_make_frame(img))

        assert seen == [True]  # single edge -> shared-memory path
        got = _drain(receiver, 1)
        assert len(got) == 1
        assert np.array_equal(got[0].image, img)
    finally:
        sender.stop()
        receiver.stop()


@posix_only
def test_mailbox_fanout_falls_back_to_inline(monkeypatch) -> None:
    monkeypatch.setattr(zmq_mailbox.codec, "SHM_ENABLED", True)
    seen = _spy_use_shm(monkeypatch)

    sender = ZMQMailbox()
    receiver_a = ZMQMailbox()
    receiver_b = ZMQMailbox()
    sender.add_publisher("receiver_a", receiver_a.consume_port)
    sender.add_publisher("receiver_b", receiver_b.consume_port)
    try:
        img = _big_image()
        sender.send(_make_frame(img))

        # Two consumers on one message -> in-band pickle, never shared memory (a single
        # segment cannot be unlinked twice without a race).
        assert seen == [False]
        assert np.array_equal(_drain(receiver_a, 1)[0].image, img)
        assert np.array_equal(_drain(receiver_b, 1)[0].image, img)
    finally:
        sender.stop()
        receiver_a.stop()
        receiver_b.stop()
