"""Mixin implementing whitespace/case normalization for TextChunk payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neudc.utils import LOGGER

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import TextChunk


class TextNormalizeMixin:
    """Mixin class for normalizing the text carried by a `TextChunk`.

    A minimal, genuinely non-CV processing step (ASR partials/finals, LLM token-stream
    chunks, ...): strips leading/trailing whitespace and lowercases `chunk.text`. Exists
    primarily to give the payload-compat check (#36) a real `TextChunk -> TextChunk` node
    pair, instead of relying only on `monkeypatch` in tests.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the mixin, forwarding all arguments to the next class in the MRO."""
        super().__init__(*args, **kwargs)

    def process(self, chunk: TextChunk) -> TextChunk:
        """Normalize the text in the TextChunk in place: strip whitespace, lowercase it.

        Follows the same in-place mutation convention as `ResizeLogicMixin`/`HashFilterMixin`.
        A chunk already marked `drop` is passed through unchanged, matching `CreateDatasetMixin`.

        Args:
        ----
            chunk: Incoming TextChunk to normalize.

        Returns:
        -------
            The same TextChunk, with `text` normalized (unless `chunk.drop` is set).

        """
        if getattr(chunk, "drop", False):
            LOGGER.debug("Skip normalizing dropped TextChunk")
            return chunk

        chunk.text = chunk.text.strip().lower()
        return chunk
