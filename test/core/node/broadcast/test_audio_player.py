"""Tests for AudioPlayerNode: plays AudioChunks through a pluggable output device."""

from __future__ import annotations

import numpy as np
import pytest

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging.types import AudioChunk
from neudc.core.node.broadcast.audio_player import AudioPlayerNode


class FakeAudioOutputDevice:
    """Dependency-free stand-in for a real speaker/stream, recording every write."""

    def __init__(self) -> None:
        self.writes: list[np.ndarray] = []
        self.closed = False

    def write(self, samples: np.ndarray) -> None:
        self.writes.append(samples)

    def close(self) -> None:
        self.closed = True


def _chunk(samples: np.ndarray, *, drop: bool = False) -> AudioChunk:
    return AudioChunk(timestamp=1.0, source="tts", samples=samples, sample_rate=16000, drop=drop)


def test_process_writes_samples_to_the_device() -> None:
    device = FakeAudioOutputDevice()
    node = AudioPlayerNode(device=device, mailbox=ZMQMailbox())
    samples = np.full(4, 0.25, dtype=np.float32)

    node.process(_chunk(samples))

    assert len(device.writes) == 1
    assert np.array_equal(device.writes[0], samples)


def test_dropped_chunk_is_not_written() -> None:
    device = FakeAudioOutputDevice()
    node = AudioPlayerNode(device=device, mailbox=ZMQMailbox())

    node.process(_chunk(np.ones(4, dtype=np.float32), drop=True))

    assert device.writes == []


def test_from_config_requires_a_device() -> None:
    with pytest.raises(ValueError, match="requires an explicit 'device'"):
        AudioPlayerNode.from_config({"mailbox": ZMQMailbox()})


def test_from_config_builds_node_with_injected_device() -> None:
    device = FakeAudioOutputDevice()

    node = AudioPlayerNode.from_config({"mailbox": ZMQMailbox(), "device": device})

    assert node.device is device
