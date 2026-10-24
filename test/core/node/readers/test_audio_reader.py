"""Tests for AudioReaderNode: a live, unbounded source paced by the device itself.

Hermetic — no real microphone/hardware, no network: a `FakeAudioDevice` stands in
for a real backend, generating deterministic samples and pacing itself with a short
real sleep proportional to `n_frames / sample_rate` (the same thing a real blocking
`read()` call on a device would do).
"""

from __future__ import annotations

import itertools
import sys
import time
import types
from typing import Any

import numpy as np
import pytest

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging.types import AudioChunk
from neudc.core.node.readers.audio_reader import AudioReaderNode, SoundDeviceAudioDevice


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


class _FakeInputStream:
    """Stand-in for `sounddevice.InputStream`: `.read()` returns a fixed (n, channels) array."""

    def __init__(self, samplerate: int, channels: int, device: object) -> None:
        del samplerate, device
        self.channels = channels
        self.started = False
        self.stopped = False
        self.closed = False
        # (n_frames, channels), like the real sd.InputStream.read() always returns —
        # even for mono, which is the whole point of the bug this covers.
        self._data = np.arange(100 * channels, dtype=np.float32).reshape(-1, channels)

    def start(self) -> None:
        self.started = True

    def read(self, n_frames: int) -> tuple[np.ndarray, bool]:
        return self._data[:n_frames], False

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_sounddevice(monkeypatch: pytest.MonkeyPatch) -> list[_FakeInputStream]:
    """Inject a fake `sounddevice` module into `sys.modules` — real hardware, not tested here."""
    created: list[_FakeInputStream] = []
    fake_module = types.ModuleType("sounddevice")

    def _input_stream(samplerate: int, channels: int, device: object) -> _FakeInputStream:
        stream = _FakeInputStream(samplerate, channels, device)
        created.append(stream)
        return stream

    fake_module.InputStream = _input_stream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_module)
    return created


def test_sounddevice_read_flattens_mono_capture_to_1d(fake_sounddevice: list[_FakeInputStream]) -> None:
    # sd.InputStream.read() always returns (n_frames, channels), even for mono — the
    # AudioDevice contract (see FakeAudioDevice above) is 1D regardless of channel
    # count, so mono capture must be flattened here, not left (n, 1).
    device = SoundDeviceAudioDevice(sample_rate=16000, channels=1)

    samples = device.read(10)

    assert samples.shape == (10,)
    assert np.array_equal(samples, np.arange(10, dtype=np.float32))


def test_sounddevice_read_keeps_multichannel_capture_2d(fake_sounddevice: list[_FakeInputStream]) -> None:
    device = SoundDeviceAudioDevice(sample_rate=16000, channels=2)

    samples = device.read(10)

    assert samples.shape == (10, 2)


def test_sounddevice_close_stops_and_closes_the_stream(fake_sounddevice: list[_FakeInputStream]) -> None:
    device = SoundDeviceAudioDevice(sample_rate=16000, channels=1)

    device.close()

    stream = fake_sounddevice[0]
    assert stream.stopped is True
    assert stream.closed is True
