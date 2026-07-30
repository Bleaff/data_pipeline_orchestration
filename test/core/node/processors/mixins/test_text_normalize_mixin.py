"""Tests for TextNormalizeMixin.process() (#36 real non-CV fixture)."""

from __future__ import annotations

from neudc.core.communication.messaging.types import TextChunk
from neudc.core.node.processors.mixins.text_normalize_mixin import TextNormalizeMixin


def test_process_strips_and_lowercases_text() -> None:
    mixin = TextNormalizeMixin()
    chunk = TextChunk(timestamp=0.0, text="  Hello WORLD  ")

    result = mixin.process(chunk)

    assert result is chunk
    assert result.text == "hello world"


def test_process_handles_empty_string() -> None:
    mixin = TextNormalizeMixin()
    chunk = TextChunk(timestamp=0.0, text="")

    result = mixin.process(chunk)

    assert result.text == ""


def test_process_passes_through_already_dropped_chunk() -> None:
    # Matches CreateDatasetMixin's convention: a dropped message is passed through
    # unmodified rather than processed.
    mixin = TextNormalizeMixin()
    chunk = TextChunk(timestamp=0.0, text="  NOT Normalized  ", drop=True)

    result = mixin.process(chunk)

    assert result is chunk
    assert result.text == "  NOT Normalized  "
    assert result.drop is True
