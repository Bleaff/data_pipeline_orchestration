"""TTS mixin: turns LLM TextChunks into AudioChunks via a pluggable speech backend."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from neudc.core.communication.messaging.types import AudioChunk

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neudc.core.communication.messaging.types import TextChunk
    from neudc.nn.backends.tts.base import BaseTTSBackend


class TtsMixin:
    """Mixin turning a stream of `TextChunk`s into synthesized speech `AudioChunk`s.

    Buffers `text` from every chunk (works whether the upstream `LlmNode` streams
    content deltas or emits one final reply) and synthesizes the buffered text as
    soon as an `is_final=True` chunk arrives — one HTTP call per reply, not per
    delta. An empty buffered utterance (e.g. an empty streaming trailer) is skipped.
    """

    def __init__(self, backend: BaseTTSBackend) -> None:
        """Initialize the mixin.

        Args:
        ----
            backend: TTS backend used to synthesize buffered text.

        """
        self.backend = backend
        self._buffer: list[str] = []

    def process(self, chunk: TextChunk) -> Iterator[AudioChunk]:
        """Buffer text, synthesizing the buffered reply once its final chunk arrives.

        Args:
        ----
            chunk: Incoming TextChunk (a content delta or a final reply).

        Yields:
        ------
            One `AudioChunk` per completed reply; nothing while the reply is still
            streaming or when the completed reply has no text.

        """
        if getattr(chunk, "drop", False):
            return

        if chunk.text:
            self._buffer.append(chunk.text)

        if not chunk.is_final:
            return

        text = "".join(self._buffer).strip()
        self._buffer = []
        if not text:
            return

        samples, sample_rate = self.backend.synthesize(text)
        yield AudioChunk(
            timestamp=time.time(),
            source=getattr(self, "id", "TtsNode"),
            samples=samples,
            sample_rate=sample_rate,
            session_id=chunk.session_id,
            turn_id=chunk.turn_id,
            is_final=True,
        )
