"""LLM mixin: wires a `BaseLLMBackend` into the graph as `TextChunk -> TextChunk`."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from neudc.core.communication.messaging.types import TextChunk
from neudc.nn.backends.llm.base import ChatMessage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neudc.nn.backends.llm.base import BaseLLMBackend


class LlmMixin:
    """Mixin turning ASR `TextChunk`s into LLM replies via a pluggable chat backend.

    Keeps a running conversation history (optionally seeded by `system_prompt`) so
    replies stay in context across turns. `process()` is always a generator: in
    non-streaming mode it yields exactly one final `TextChunk`; in streaming mode
    (`stream=True`) it yields one non-final `TextChunk` per content delta plus a
    trailing empty final one, matching the streaming convention documented in
    `docs/NODE_DEV_GUIDE.md`. A dropped or empty incoming chunk gets no reply.
    """

    def __init__(
        self,
        backend: BaseLLMBackend,
        *,
        system_prompt: str | None = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the mixin.

        Args:
        ----
            backend: LLM backend used to generate replies.
            system_prompt: Optional system message seeding the conversation history.
            stream: Whether to stream content deltas instead of one final reply.
            temperature: Sampling temperature forwarded to the backend.
            max_tokens: Upper bound on generated tokens, provider default if `None`.

        """
        self.backend = backend
        self.stream = stream
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._history: list[ChatMessage] = []
        if system_prompt:
            self._history.append(ChatMessage(role="system", content=system_prompt))

    def process(self, chunk: TextChunk) -> Iterator[TextChunk]:
        """Answer one user utterance, updating the running conversation history.

        Args:
        ----
            chunk: Incoming final `TextChunk` (typically an ASR transcription).

        Yields:
        ------
            One final `TextChunk` (non-streaming), or content-delta `TextChunk`s
            followed by a trailing empty final one (streaming). Nothing for a
            dropped or blank-text chunk.

        """
        if getattr(chunk, "drop", False) or not chunk.text.strip():
            return

        self._history.append(ChatMessage(role="user", content=chunk.text))

        if not self.stream:
            reply = self.backend.generate(self._history, temperature=self.temperature, max_tokens=self.max_tokens)
            self._history.append(ChatMessage(role="assistant", content=reply))
            yield self._make_chunk(chunk, reply, is_final=True)
            return

        pieces: list[str] = []
        for delta in self.backend.stream(self._history, temperature=self.temperature, max_tokens=self.max_tokens):
            pieces.append(delta)
            yield self._make_chunk(chunk, delta, is_final=False)

        self._history.append(ChatMessage(role="assistant", content="".join(pieces)))
        yield self._make_chunk(chunk, "", is_final=True)

    def _make_chunk(self, source_chunk: TextChunk, text: str, *, is_final: bool) -> TextChunk:
        return TextChunk(
            timestamp=time.time(),
            source=getattr(self, "id", "LlmNode"),
            text=text,
            session_id=source_chunk.session_id,
            turn_id=source_chunk.turn_id,
            is_final=is_final,
        )
