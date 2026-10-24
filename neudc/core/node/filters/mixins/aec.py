"""Adaptive acoustic echo cancellation (AEC): estimates and subtracts the assistant's own TTS output from the microphone signal, using it as a reference.

Without this, a mic and speaker that aren't acoustically isolated (a laptop, no
headphones) pick up the assistant's own voice as "the user talking again" and
barge-in fires on itself. A fixed energy threshold (see `VadFilterMixin`) can only
trade off false triggers against missed real interruptions — it has no way to tell
the two apart. An adaptive filter that knows what's actually being played can.

Algorithm: NLMS (normalized least-mean-squares). `ReferenceBuffer` accumulates the
audio actually written to the speaker (see `AudioPlayerNode`, which now emits a copy
of every block it plays); `AdaptiveEchoCanceller` uses the most recent
`filter_length` reference samples before each mic sample as a sliding window,
predicts the echo component, and subtracts it. The filter adapts continuously, so it
tracks a given room/speaker/mic setup better the longer it runs.

Deliberately not sample-accurate: `ReferenceBuffer` aligns by arrival order, not by
reconstructing a shared wall-clock sample index between the two independent
(reader/vad and player) node threads. Reference chunks are emitted right after each
block is actually written to the output device (see `AudioPlayerNode.process`), so
arrival order is already a close proxy for real playback time; the adaptive filter's
own tap span absorbs the residual slop. This trades a bit of precision for an
implementation that doesn't depend on a precise cross-thread clock — good enough to
suppress echo, not a substitute for hardware-level AEC.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit
from scipy.signal import resample_poly

__all__ = ("REFERENCE_SOURCE", "AdaptiveEchoCanceller", "ReferenceBuffer")

#: `AudioChunk.source` value `AudioPlayerNode` stamps on every reference chunk it emits
#: back for AEC, and `VadFilterMixin` looks for to route a chunk into `ReferenceBuffer`
#: instead of treating it as mic input.
REFERENCE_SOURCE = "speaker-reference"


@njit(cache=True)
def _nlms_process(
    mic: np.ndarray,
    ref_history: np.ndarray,
    weights: np.ndarray,
    step_size: float,
    epsilon: float,
) -> np.ndarray:
    """Run NLMS over `mic`, updating `weights` in place.

    `ref_history` must have length `len(weights) + len(mic)`;
    `ref_history[i : i + len(weights)]` are the reference samples immediately
    preceding (and including) the instant aligned with `mic[i]`, oldest first.
    """
    n = mic.shape[0]
    filter_length = weights.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x_vec = ref_history[i : i + filter_length][::-1]  # most-recent-first, matches weights[0]
        y = np.dot(weights, x_vec)
        e = mic[i] - y
        norm = np.dot(x_vec, x_vec) + epsilon
        weights += (step_size / norm) * e * x_vec
        out[i] = e
    return out


class AdaptiveEchoCanceller:
    """Stateful NLMS filter: weights persist and keep adapting across calls.

    `filter_length` (samples, at the mic's rate) bounds the acoustic + buffering
    delay this can compensate for — e.g. 2048 samples at 16 kHz is ~128ms. Too short
    and echo outside that window leaks through uncancelled; too long slows
    convergence and costs more compute per sample.
    """

    def __init__(
        self,
        filter_length: int = 2048,
        step_size: float = 0.3,
        epsilon: float = 1e-6,
        double_talk_ratio: float = 1.5,
        double_talk_floor: float = 1e-3,
    ) -> None:
        """Initialize the canceller.

        Args:
        ----
            filter_length: Taps (samples, at the mic's rate) — bounds the acoustic +
                buffering delay this can compensate for.
            step_size: NLMS adaptation rate (mu).
            epsilon: Small constant preventing division by zero when the reference
                window is near-silent.
            double_talk_ratio: Freeze adaptation for a block whose *residual* RMS
                (`mic - current_echo_prediction`) exceeds the prediction's own RMS by
                more than this factor — an energy-based double-talk guard. Compared
                against the filter's own prediction (not the raw reference): a
                well-converged filter's prediction already reflects the room's real
                attenuation, so this is a like-for-like comparison, unlike comparing
                against the raw (unattenuated) reference. Without this, a block
                where the user's own speech dominates (barge-in, exactly the case
                that matters most) gets "explained away" by the adaptive filter same
                as echo would, corrupting the weights and partially cancelling real
                speech instead of just the echo. Frozen blocks still get the
                *current* echo estimate subtracted — only the weight update is
                skipped, not the cancellation itself.
            double_talk_floor: Minimum prediction RMS the guard trusts — below this
                (e.g. right after startup, before the filter has learned anything)
                the guard would divide near-zero by near-zero and freeze forever, so
                it's bypassed and adaptation proceeds until the filter has something
                to lose.

        """
        if filter_length <= 0:
            msg = f"filter_length must be positive, got {filter_length}"
            raise ValueError(msg)
        self.filter_length = filter_length
        self.step_size = step_size
        self.epsilon = epsilon
        self.double_talk_ratio = double_talk_ratio
        self.double_talk_floor = double_talk_floor
        self._weights = np.zeros(filter_length, dtype=np.float64)

    def process(self, mic: np.ndarray, ref_history: np.ndarray) -> np.ndarray:
        """Return `mic` with the estimated echo subtracted.

        Args:
        ----
            mic: Mic samples to clean.
            ref_history: Exactly `filter_length + len(mic)` reference samples (at the
                same sample rate as `mic`) ending at the same instant as `mic[-1]`
                (see `ReferenceBuffer.window`).

        Returns:
        -------
            `mic`-shaped array, same dtype, with the echo estimate subtracted.

        """
        expected_len = self.filter_length + len(mic)
        if len(ref_history) != expected_len:
            msg = f"ref_history must have length {expected_len} (filter_length + len(mic)), got {len(ref_history)}"
            raise ValueError(msg)

        mic64 = mic.astype(np.float64)
        ref64 = ref_history.astype(np.float64)
        effective_step = self.step_size
        if self._weights.any():
            # Vectorized "what would the *current* (pre-update) filter predict for
            # this whole block" — a sliding dot product, i.e. correlation. Purely
            # for the double-talk decision below; the real (adaptive, sample-by-
            # sample) prediction happens inside _nlms_process.
            windows = np.lib.stride_tricks.sliding_window_view(ref64, self.filter_length)[: len(mic)]
            y_pred = windows[:, ::-1] @ self._weights
            pred_rms = float(np.sqrt(np.mean(y_pred**2)))
            if pred_rms > self.double_talk_floor:
                residual_rms = float(np.sqrt(np.mean((mic64 - y_pred) ** 2)))
                if residual_rms > self.double_talk_ratio * pred_rms:
                    effective_step = 0.0

        out = _nlms_process(mic64, ref64, self._weights, effective_step, self.epsilon)
        return out.astype(mic.dtype)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    return resample_poly(samples, up=target_rate, down=source_rate).astype(samples.dtype)


class ReferenceBuffer:
    """Rolling buffer of recently-played reference (speaker) audio, resampled to a fixed target rate (the mic's).

    Trims itself to `max_seconds` and reports staleness (`is_active`) so a mic chunk
    arriving long after the assistant stopped talking doesn't get "cancelled"
    against leftover reference audio that isn't actually playing anymore.
    """

    def __init__(self, target_sample_rate: int, max_seconds: float = 3.0) -> None:
        """Initialize an empty buffer resampling pushed audio to `target_sample_rate`."""
        self.target_sample_rate = target_sample_rate
        self._max_samples = int(target_sample_rate * max_seconds)
        self._samples = np.zeros(0, dtype=np.float32)
        self._last_push_time: float | None = None

    def push(self, samples: np.ndarray, source_sample_rate: int) -> None:
        """Append newly-played reference samples, resampling if needed."""
        if samples.size == 0:
            return
        resampled = _resample(samples, source_sample_rate, self.target_sample_rate).astype(np.float32)
        self._samples = np.concatenate([self._samples, resampled])
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples :]
        self._last_push_time = time.time()

    def is_active(self, max_staleness_sec: float = 0.5) -> bool:
        """Whether reference audio was pushed recently enough to trust for cancellation."""
        return self._last_push_time is not None and (time.time() - self._last_push_time) <= max_staleness_sec

    def window(self, filter_length: int, length: int) -> np.ndarray | None:
        """Return the most recent `filter_length + length` reference samples, or `None` if not enough is buffered."""
        needed = filter_length + length
        if len(self._samples) < needed:
            return None
        return self._samples[-needed:]
