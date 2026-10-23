"""TTS mixin: turns LLM TextChunks into AudioChunks via a pluggable speech backend."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from neudc.core.communication.messaging.types import AudioChunk

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neudc.core.communication.messaging.types import TextChunk
    from neudc.nn.backends.tts.base import BaseTTSBackend

#: Markdown emphasis/formatting characters an LLM commonly reaches for out of habit
#: even in a plain-text conversation (bold/italic asterisks or underscores,
#: strikethrough tildes, inline-code backticks) and heading hashes — none of them
#: mean anything spoken aloud, so they're stripped rather than voiced literally
#: (a TTS engine has no notion of "this word is bold", it just reads the character).
#: This is a safety net, not the primary fix — `system_prompt` should already ask the
#: LLM not to use markdown in the first place.
_MARKDOWN_STRIP_RE = re.compile(r"[*_`~]+|^#{1,6}\s+", re.MULTILINE)


def _sanitize_for_speech(text: str) -> str:
    """Strip markdown formatting characters a TTS engine would otherwise read aloud literally.

    Markdown stripping runs first, while `^` still anchors on real line starts (for
    heading hashes); whitespace — including the newlines that stripping a heading
    marker or a list structure can leave behind — is then collapsed to single spaces,
    since a spoken reply has no use for line breaks.
    """
    stripped = _MARKDOWN_STRIP_RE.sub("", text)
    return re.sub(r"\s+", " ", stripped).strip()


class TtsMixin:
    """Mixin turning a stream of `TextChunk`s into synthesized speech `AudioChunk`s.

    Buffers `text` from every chunk (works whether the upstream `LlmNode` streams
    content deltas or emits one final reply), runs it through `_sanitize_for_speech`
    (strips markdown formatting characters — see that function) and synthesizes as
    soon as an `is_final=True` chunk arrives — one HTTP call per reply, not per
    delta. An empty buffered utterance (e.g. an empty streaming trailer) is skipped.

    The buffer is keyed to the (session_id, turn_id) it started accumulating for: a
    chunk from a different turn resets it first. This matters for barge-in — if a
    turn gets cancelled mid-stream (see `VadFilterMixin`), its trailing `is_final`
    chunk never arrives, so without this the next turn's reply would get its
    predecessor's leftover partial text spliced onto the front of it.
    """

    def __init__(self, backend: BaseTTSBackend) -> None:
        """Initialize the mixin.

        Args:
        ----
            backend: TTS backend used to synthesize buffered text.

        """
        self.backend = backend
        self._buffer: list[str] = []
        self._buffer_turn_key: tuple[str | None, int | None] | None = None

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

        turn_key = (chunk.session_id, chunk.turn_id)
        if turn_key != self._buffer_turn_key:
            self._buffer = []
            self._buffer_turn_key = turn_key

        if chunk.text:
            self._buffer.append(chunk.text)

        if not chunk.is_final:
            return

        text = _sanitize_for_speech("".join(self._buffer))
        self._buffer = []
        self._buffer_turn_key = None
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
