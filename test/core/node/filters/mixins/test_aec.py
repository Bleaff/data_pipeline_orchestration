"""Tests for the NLMS adaptive echo canceller and its reference buffer."""

from __future__ import annotations

import numpy as np
import pytest

from neudc.core.node.filters.mixins.aec import AdaptiveEchoCanceller, ReferenceBuffer


def _tone(freq: float, n: int, sample_rate: int, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestReferenceBuffer:
    def test_window_returns_none_until_enough_buffered(self) -> None:
        buf = ReferenceBuffer(target_sample_rate=16000, max_seconds=1.0)
        assert buf.window(filter_length=256, length=100) is None
        buf.push(np.zeros(200, dtype=np.float32), source_sample_rate=16000)
        assert buf.window(filter_length=256, length=100) is None
        buf.push(np.zeros(200, dtype=np.float32), source_sample_rate=16000)
        window = buf.window(filter_length=256, length=100)
        assert window is not None
        assert len(window) == 356

    def test_push_resamples_to_target_rate(self) -> None:
        buf = ReferenceBuffer(target_sample_rate=16000, max_seconds=1.0)
        buf.push(np.zeros(22050, dtype=np.float32), source_sample_rate=22050)
        # 22050 samples at 22050Hz is 1s -> resampled to 16000 samples at 16000Hz.
        assert len(buf._samples) == 16000

    def test_push_trims_to_max_seconds(self) -> None:
        buf = ReferenceBuffer(target_sample_rate=16000, max_seconds=0.5)
        buf.push(np.zeros(16000, dtype=np.float32), source_sample_rate=16000)
        assert len(buf._samples) == 8000

    def test_push_ignores_empty_input(self) -> None:
        buf = ReferenceBuffer(target_sample_rate=16000, max_seconds=1.0)
        buf.push(np.zeros(0, dtype=np.float32), source_sample_rate=16000)
        assert buf._last_push_time is None
        assert not buf.is_active()

    def test_is_active_true_immediately_after_push(self) -> None:
        buf = ReferenceBuffer(target_sample_rate=16000, max_seconds=1.0)
        assert not buf.is_active()
        buf.push(np.zeros(100, dtype=np.float32), source_sample_rate=16000)
        assert buf.is_active(max_staleness_sec=5.0)

    def test_is_active_false_when_stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        buf = ReferenceBuffer(target_sample_rate=16000, max_seconds=1.0)
        current_time = [1000.0]
        monkeypatch.setattr("neudc.core.node.filters.mixins.aec.time.time", lambda: current_time[0])
        buf.push(np.zeros(100, dtype=np.float32), source_sample_rate=16000)
        current_time[0] += 10.0
        assert not buf.is_active(max_staleness_sec=0.5)


class TestAdaptiveEchoCanceller:
    def test_rejects_non_positive_filter_length(self) -> None:
        with pytest.raises(ValueError, match="filter_length must be positive"):
            AdaptiveEchoCanceller(filter_length=0)

    def test_rejects_wrong_ref_history_length(self) -> None:
        canceller = AdaptiveEchoCanceller(filter_length=32)
        mic = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="ref_history must have length"):
            canceller.process(mic, np.zeros(10, dtype=np.float32))

    def test_suppresses_pure_echo(self) -> None:
        sample_rate = 16000
        filter_length = 256
        n = sample_rate  # 1 second
        rng = np.random.default_rng(0)
        reference = _tone(220.0, n, sample_rate) + 0.05 * rng.standard_normal(n).astype(np.float32)

        delay = 20
        echo = np.zeros(n, dtype=np.float32)
        echo[delay:] = 0.4 * reference[: n - delay]
        mic = echo + 0.01 * rng.standard_normal(n).astype(np.float32)

        canceller = AdaptiveEchoCanceller(filter_length=filter_length, step_size=0.3)
        ref_buf = ReferenceBuffer(target_sample_rate=sample_rate, max_seconds=2.0)

        chunk_size = 400
        cleaned = np.zeros(n, dtype=np.float32)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            ref_buf.push(reference[start:end], sample_rate)
            window = ref_buf.window(filter_length, end - start)
            cleaned[start:end] = canceller.process(mic[start:end], window) if window is not None else mic[start:end]

        # Skip the initial convergence period; compare RMS in the settled tail.
        tail = slice(int(0.5 * sample_rate), n)

        def rms(x: np.ndarray) -> float:
            return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

        assert rms(cleaned[tail]) < 0.5 * rms(mic[tail])

    def test_preserves_near_end_speech_during_double_talk(self) -> None:
        sample_rate = 16000
        filter_length = 256
        n = sample_rate * 2
        rng = np.random.default_rng(1)
        reference = _tone(220.0, n, sample_rate) + 0.05 * rng.standard_normal(n).astype(np.float32)

        delay = 20
        echo = np.zeros(n, dtype=np.float32)
        echo[delay:] = 0.4 * reference[: n - delay]

        near_end = np.zeros(n, dtype=np.float32)
        burst_start, burst_len = int(1.5 * sample_rate), int(0.2 * sample_rate)
        near_end[burst_start : burst_start + burst_len] = _tone(300.0, burst_len, sample_rate, amplitude=0.6)

        mic = echo + near_end + 0.01 * rng.standard_normal(n).astype(np.float32)

        canceller = AdaptiveEchoCanceller(filter_length=filter_length, step_size=0.3)
        ref_buf = ReferenceBuffer(target_sample_rate=sample_rate, max_seconds=3.0)

        chunk_size = 400
        cleaned = np.zeros(n, dtype=np.float32)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            ref_buf.push(reference[start:end], sample_rate)
            window = ref_buf.window(filter_length, end - start)
            cleaned[start:end] = canceller.process(mic[start:end], window) if window is not None else mic[start:end]

        burst = slice(burst_start, burst_start + burst_len)

        def rms(x: np.ndarray) -> float:
            return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

        preservation_ratio = rms(cleaned[burst]) / rms(near_end[burst])
        assert 0.85 < preservation_ratio < 1.15
