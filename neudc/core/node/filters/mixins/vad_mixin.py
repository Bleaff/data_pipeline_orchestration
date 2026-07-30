"""Voice-Activity-Detection Filter Mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import AudioChunk


class VadFilterMixin:
    """Mixin for marking silent `AudioChunk`s as droppable, by RMS energy.

    A simple, dependency-free voice-activity heuristic: compute the RMS energy of
    the chunk's samples and compare it against `energy_threshold`. No ML model is
    needed for this — matches the level of the existing `HashFilterMixin` (a plain
    signal-level heuristic, not an inference node).

    Attributes
    ----------
        energy_threshold: RMS energy below which a chunk is considered silence.

    """

    def __init__(self, energy_threshold: float = 0.01) -> None:
        """Initialize the mixin.

        Args:
        ----
            energy_threshold: RMS energy below which a chunk is marked as dropped.

        """
        self.energy_threshold = energy_threshold

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Mark `chunk` as dropped if its RMS energy is below `energy_threshold`.

        Args:
        ----
            chunk: Incoming AudioChunk to inspect.

        Returns:
        -------
            The same AudioChunk, with `drop` set to True when it looks like silence.

        """
        samples = chunk.samples
        if samples.size == 0:
            chunk.drop = True
            return chunk

        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if rms < self.energy_threshold:
            chunk.drop = True
        return chunk
