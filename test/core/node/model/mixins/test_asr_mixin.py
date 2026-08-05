"""Tests for AsrMixin: utterance segmentation over VAD-tagged AudioChunks."""

from __future__ import annotations

import numpy as np

from neudc.core.communication.messaging.types import AudioChunk
from neudc.core.node.model.mixins.asr_mixin import AsrMixin


class _FakeASRBackend:
    def __init__(self, text: str = "hello there") -> None:
        self.text = text
        self.calls: list[tuple[np.ndarray, int]] = []

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        self.calls.append((samples, sample_rate))
        return self.text


def _chunk(samples: np.ndarray, *, drop: bool, sample_rate: int = 16000) -> AudioChunk:
    return AudioChunk(timestamp=1.0, source="mic", samples=samples, sample_rate=sample_rate, drop=drop)


def _speech(n: int = 8000, sample_rate: int = 16000) -> AudioChunk:
    return _chunk(np.full(n, 0.5, dtype=np.float32), drop=False, sample_rate=sample_rate)


def _silence(n: int = 8000, sample_rate: int = 16000) -> AudioChunk:
    return _chunk(np.zeros(n, dtype=np.float32), drop=True, sample_rate=sample_rate)


def test_speech_chunks_alone_yield_nothing() -> None:
    backend = _FakeASRBackend()
    mixin = AsrMixin(backend)

    out = list(mixin.process(_speech()))

    assert out == []
    assert backend.calls == []


def test_silence_after_enough_speech_flushes_one_utterance() -> None:
    backend = _FakeASRBackend(text="turn the lights on")
    mixin = AsrMixin(backend, session_id="s1")

    list(mixin.process(_speech(8000)))  # 0.5s of speech at 16kHz
    out = list(mixin.process(_silence()))

    assert len(out) == 1
    result = out[0]
    assert result.text == "turn the lights on"
    assert result.session_id == "s1"
    assert result.turn_id == 1
    assert result.is_final is True
    assert len(backend.calls) == 1
    assert len(backend.calls[0][0]) == 8000
    assert backend.calls[0][1] == 16000


def test_multiple_speech_chunks_are_concatenated_into_one_call() -> None:
    backend = _FakeASRBackend()
    mixin = AsrMixin(backend)

    list(mixin.process(_speech(4000)))
    list(mixin.process(_speech(4000)))
    list(mixin.process(_silence()))

    assert len(backend.calls) == 1
    assert len(backend.calls[0][0]) == 8000


def test_sub_threshold_speech_is_discarded_without_a_call() -> None:
    backend = _FakeASRBackend()
    mixin = AsrMixin(backend, min_speech_duration=0.3)

    list(mixin.process(_speech(1000)))  # 0.0625s at 16kHz, below threshold
    out = list(mixin.process(_silence()))

    assert out == []
    assert backend.calls == []


def test_silence_with_no_buffered_speech_yields_nothing() -> None:
    backend = _FakeASRBackend()
    mixin = AsrMixin(backend)

    out = list(mixin.process(_silence()))

    assert out == []
    assert backend.calls == []


def test_turn_id_increments_across_consecutive_utterances() -> None:
    backend = _FakeASRBackend()
    mixin = AsrMixin(backend)

    list(mixin.process(_speech(8000)))
    first = list(mixin.process(_silence()))
    list(mixin.process(_speech(8000)))
    second = list(mixin.process(_silence()))

    assert first[0].turn_id == 1
    assert second[0].turn_id == 2
