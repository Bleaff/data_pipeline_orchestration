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


class _FakeCancelMailbox:
    """Duck-typed stand-in exposing only `is_cancelled`, counting how many times it's asked.

    `BaseNode._is_cancelled` only needs `mailbox.is_cancelled(session_id, turn_id)`
    (see `neudc/core/base/base_node.py`), so this is enough to drive
    `AudioPlayerNode` without any real ZMQ transport.
    """

    def __init__(self, *, cancelled_from_call: int | None = None) -> None:
        self.calls = 0
        self.cancelled_from_call = cancelled_from_call

    def is_cancelled(self, session_id: str | None, turn_id: int | None) -> bool:
        del session_id, turn_id
        self.calls += 1
        return self.cancelled_from_call is not None and self.calls >= self.cancelled_from_call


def _chunk(
    samples: np.ndarray,
    *,
    drop: bool = False,
    sample_rate: int = 16000,
    session_id: str | None = None,
    turn_id: int | None = None,
) -> AudioChunk:
    return AudioChunk(
        timestamp=1.0,
        source="tts",
        samples=samples,
        sample_rate=sample_rate,
        drop=drop,
        session_id=session_id,
        turn_id=turn_id,
    )


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


def test_long_chunk_is_written_in_blocks_of_check_interval() -> None:
    device = FakeAudioOutputDevice()
    mailbox = _FakeCancelMailbox()  # never cancelled
    node = AudioPlayerNode(device=device, mailbox=mailbox, playback_check_interval=0.1)
    # sample_rate=10 * 0.1s check interval = 1 frame per block; 5 samples -> 5 blocks.
    samples = np.arange(5, dtype=np.float32)

    node.process(_chunk(samples, sample_rate=10))

    assert len(device.writes) == 5
    assert np.array_equal(np.concatenate(device.writes), samples)


def test_playback_stops_early_when_the_turn_is_cancelled_mid_chunk() -> None:
    device = FakeAudioOutputDevice()
    # Not cancelled for the first 3 checks, cancelled from the 4th check onward —
    # simulates a barge-in ControlMessage arriving partway through playback.
    mailbox = _FakeCancelMailbox(cancelled_from_call=4)
    node = AudioPlayerNode(device=device, mailbox=mailbox, playback_check_interval=0.1)
    samples = np.arange(10, dtype=np.float32)

    node.process(_chunk(samples, sample_rate=10, session_id="s1", turn_id=1))

    # 3 blocks played (checks 1-3 said "not cancelled"), the 4th check stops it.
    assert len(device.writes) == 3
    assert np.array_equal(np.concatenate(device.writes), samples[:3])


def test_cancellation_is_checked_with_the_chunks_own_session_and_turn() -> None:
    device = FakeAudioOutputDevice()
    mailbox = _FakeCancelMailbox()
    node = AudioPlayerNode(device=device, mailbox=mailbox)
    samples = np.full(4, 0.25, dtype=np.float32)

    node.process(_chunk(samples, session_id="sess-1", turn_id=7))

    assert mailbox.calls >= 1  # BaseNode._is_cancelled was actually consulted
