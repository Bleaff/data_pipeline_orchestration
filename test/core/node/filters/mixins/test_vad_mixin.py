"""Tests for VadFilterMixin's RMS-energy voice-activity heuristic."""

from __future__ import annotations

import numpy as np

from neudc.core.communication.messaging.types import AudioChunk
from neudc.core.node.filters.mixins.vad_mixin import VadFilterMixin


def _chunk(samples: np.ndarray) -> AudioChunk:
    return AudioChunk(timestamp=1.0, source="mic", samples=samples, sample_rate=16000, channels=1)


def test_loud_chunk_passes_through() -> None:
    mixin = VadFilterMixin(energy_threshold=0.05)
    t = np.arange(1600) / 16000
    loud = (0.8 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    result = mixin.process(_chunk(loud))

    assert result.drop is False


def test_silent_chunk_is_dropped() -> None:
    mixin = VadFilterMixin(energy_threshold=0.05)
    silence = np.zeros(1600, dtype=np.float32)

    result = mixin.process(_chunk(silence))

    assert result.drop is True


def test_empty_samples_do_not_crash() -> None:
    mixin = VadFilterMixin(energy_threshold=0.05)
    empty = np.array([], dtype=np.float32)

    result = mixin.process(_chunk(empty))

    assert result.drop is True
