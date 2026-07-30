"""Tests for AudioReaderNode: a live, unbounded source paced by the device itself.

Hermetic — no real microphone/hardware, no network: a `FakeAudioDevice` stands in
for a real backend, generating deterministic samples and pacing itself with a short
real sleep proportional to `n_frames / sample_rate` (the same thing a real blocking
`read()` call on a device would do).
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging.types import AudioChunk
from neudc.core.node.readers.audio_reader import AudioReaderNode


class FakeAudioDevice:
    """Deterministic, dependency-free stand-in for a real microphone/stream.

    Generates a synthetic sine wave and blocks (via a short real sleep) for
    `n_frames / sample_rate` seconds per `read()` call — this is what makes the
    device, not the node, set the pace. `max_reads`, if given, makes the device
    "close"/error out after that many reads, for the error-path test.
    """

    def __init__(self, sample_rate: int, max_reads: int | None = None) -> None:
        self.sample_rate = sample_rate
        self.max_reads = max_reads
        self._reads = 0
        self._cursor = 0
        self.closed = False

    def read(self, n_frames: int) -> np.ndarray:
        if self.max_reads is not None and self._reads >= self.max_reads:
            msg = "fake device exhausted"
            raise RuntimeError(msg)
        self._reads += 1
        time.sleep(n_frames / self.sample_rate)
        t = np.arange(self._cursor, self._cursor + n_frames) / self.sample_rate
        self._cursor += n_frames
        return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    def close(self) -> None:
        self.closed = True


def _drain_with_timestamps(mailbox: ZMQMailbox, expected: int, timeout: float = 5.0) -> list[tuple[Any, float]]:
    received: list[tuple[Any, float]] = []
    deadline = time.time() + timeout
    while len(received) < expected and time.time() < deadline:
        msg = mailbox.receive(timeout=0.1)
        if msg is not None:
            received.append((msg, time.monotonic()))
    return received


def test_reader_emits_correctly_shaped_audio_chunks_paced_by_the_device() -> None:
    sample_rate = 8000
    chunk_duration = 0.05  # 50 ms chunks keep the test fast
    receiver = ZMQMailbox()
    sender_mailbox = ZMQMailbox()
    sender_mailbox.add_publisher("receiver", receiver.consume_port)
    device = FakeAudioDevice(sample_rate=sample_rate)
    reader = AudioReaderNode(
        device=device,
        mailbox=sender_mailbox,
        sample_rate=sample_rate,
        channels=1,
        chunk_duration=chunk_duration,
        source="fake-mic",
    )
    try:
        reader.start()
        got = _drain_with_timestamps(receiver, 4)

        assert len(got) == 4
        for msg, _ in got:
            assert isinstance(msg, AudioChunk)
            assert msg.sample_rate == sample_rate
            assert msg.channels == 1
            assert msg.source == "fake-mic"
            assert len(msg.samples) == int(sample_rate * chunk_duration)

        timestamps = [msg.timestamp for msg, _ in got]
        assert all(b > a for a, b in itertools.pairwise(timestamps))  # monotonically increasing

        gaps = [t2 - t1 for (_, t1), (_, t2) in itertools.pairwise(got)]
        assert all(gap >= chunk_duration * 0.5 for gap in gaps), gaps  # paced by the device, not a busy loop
    finally:
        reader.stop()
        receiver.stop()


def test_device_error_stops_reader_cleanly() -> None:
    """A device that errors out (closed/exhausted) makes the reader stop itself, not crash."""
    sample_rate = 8000
    device = FakeAudioDevice(sample_rate=sample_rate, max_reads=1)
    mailbox = ZMQMailbox()
    reader = AudioReaderNode(
        device=device,
        mailbox=mailbox,
        sample_rate=sample_rate,
        chunk_duration=0.01,
    )
    reader.start()

    deadline = time.time() + 5
    while reader.is_running and time.time() < deadline:
        time.sleep(0.05)

    assert reader.is_running is False
