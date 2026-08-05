"""ASR mixin: turns VAD-segmented speech AudioChunks into TextChunks via a remote ASR backend."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from neudc.core.communication.messaging.types import TextChunk
from neudc.utils import LOGGER

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neudc.core.communication.messaging.types import AudioChunk
    from neudc.nn.backends.asr.base import BaseASRBackend


class AsrMixin:
    """Mixin turning a stream of VAD-tagged `AudioChunk`s into ASR `TextChunk`s.

    `VadFilterMixin` already marks silent chunks via `chunk.drop`; this mixin reuses
    that as utterance segmentation: it buffers consecutive speech chunks and, on the
    next silent chunk, flushes the buffered utterance through `backend.transcribe`
    (one HTTP call per utterance, not per chunk). A buffered utterance shorter than
    `min_speech_duration` is discarded as noise, with no call made.
    """

    def __init__(
        self,
        backend: BaseASRBackend,
        *,
        session_id: str = "default",
        min_speech_duration: float = 0.3,
    ) -> None:
        """Initialize the mixin.

        Args:
        ----
            backend: ASR backend used to transcribe buffered utterances.
            session_id: Stamped onto every emitted `TextChunk` for downstream turn tracking.
            min_speech_duration: Minimum buffered speech duration (seconds) worth transcribing.

        """
        self.backend = backend
        self.session_id = session_id
        self.min_speech_duration = min_speech_duration
        self._buffer: list[np.ndarray] = []
        self._buffer_sample_rate: int | None = None
        self._turn_id = 0

    def process(self, chunk: AudioChunk) -> Iterator[TextChunk]:
        """Buffer speech, transcribing the buffered utterance once silence follows it.

        Args:
        ----
            chunk: Incoming AudioChunk, already tagged by VAD via `chunk.drop`.

        Yields:
        ------
            One final `TextChunk` per completed utterance; nothing while speech is
            still ongoing or when a silent chunk arrives with nothing buffered.

        """
        if not chunk.drop:
            if chunk.samples.size:
                self._buffer.append(chunk.samples)
                self._buffer_sample_rate = chunk.sample_rate
            return

        if not self._buffer:
            return

        sample_rate = self._buffer_sample_rate
        assert sample_rate is not None  # invariant: set alongside every append to a non-empty _buffer
        samples = np.concatenate(self._buffer)
        self._buffer = []
        self._buffer_sample_rate = None

        duration = len(samples) / sample_rate
        if duration < self.min_speech_duration:
            LOGGER.debug(f"Discarding {duration:.2f}s speech buffer below min_speech_duration")
            return

        text = self.backend.transcribe(samples, sample_rate)
        self._turn_id += 1
        yield TextChunk(
            timestamp=time.time(),
            source=getattr(self, "id", "AsrNode"),
            text=text,
            session_id=self.session_id,
            turn_id=self._turn_id,
            is_final=True,
        )
