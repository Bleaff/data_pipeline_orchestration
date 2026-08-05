"""Tests for TtsMixin: buffering TextChunks across a reply, synthesizing once it's final."""

from __future__ import annotations

import numpy as np

from neudc.core.communication.messaging.types import TextChunk
from neudc.core.node.model.mixins.tts_mixin import TtsMixin


class _FakeTTSBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str):
        self.calls.append(text)
        return np.full(4, 0.1, dtype=np.float32), 16000


def _chunk(text: str, *, is_final: bool, session_id: str = "s1", turn_id: int = 1, drop: bool = False) -> TextChunk:
    return TextChunk(
        timestamp=1.0,
        source="llm",
        text=text,
        session_id=session_id,
        turn_id=turn_id,
        is_final=is_final,
        drop=drop,
    )


def test_non_final_chunk_alone_yields_nothing() -> None:
    backend = _FakeTTSBackend()
    mixin = TtsMixin(backend)

    out = list(mixin.process(_chunk("partial", is_final=False)))

    assert out == []
    assert backend.calls == []


def test_final_chunk_synthesizes_buffered_text() -> None:
    backend = _FakeTTSBackend()
    mixin = TtsMixin(backend)

    list(mixin.process(_chunk("Hello", is_final=False)))
    list(mixin.process(_chunk(" world", is_final=False)))
    out = list(mixin.process(_chunk("", is_final=True)))

    assert backend.calls == ["Hello world"]
    assert len(out) == 1
    assert out[0].sample_rate == 16000
    assert out[0].session_id == "s1"
    assert out[0].turn_id == 1
    assert out[0].is_final is True


def test_single_final_chunk_non_streaming_reply_works_too() -> None:
    backend = _FakeTTSBackend()
    mixin = TtsMixin(backend)

    out = list(mixin.process(_chunk("full reply", is_final=True)))

    assert backend.calls == ["full reply"]
    assert len(out) == 1


def test_empty_final_reply_yields_nothing() -> None:
    backend = _FakeTTSBackend()
    mixin = TtsMixin(backend)

    out = list(mixin.process(_chunk("", is_final=True)))

    assert out == []
    assert backend.calls == []


def test_dropped_chunk_yields_nothing_and_does_not_buffer() -> None:
    backend = _FakeTTSBackend()
    mixin = TtsMixin(backend)

    list(mixin.process(_chunk("ignored", is_final=False, drop=True)))
    out = list(mixin.process(_chunk("", is_final=True)))

    assert out == []
    assert backend.calls == []


def test_buffer_resets_after_flush() -> None:
    backend = _FakeTTSBackend()
    mixin = TtsMixin(backend)

    list(mixin.process(_chunk("first", is_final=True)))
    list(mixin.process(_chunk("second", is_final=True)))

    assert backend.calls == ["first", "second"]


def test_interrupted_turn_does_not_leak_into_the_next_turns_reply() -> None:
    # Barge-in: turn 1 gets cancelled mid-stream, so it never gets an is_final chunk
    # (VadFilterMixin/BaseNode._handle stop delivering it, see AudioPlayerNode /
    # VadFilterMixin docstrings). Turn 2's deltas must not get turn 1's leftover
    # partial text spliced onto the front of them.
    backend = _FakeTTSBackend()
    mixin = TtsMixin(backend)

    list(mixin.process(_chunk("Turn one, never ", is_final=False, turn_id=1)))
    list(mixin.process(_chunk("finished", is_final=False, turn_id=1)))  # turn 1 cut off here, no final ever arrives

    out = list(mixin.process(_chunk("Turn two.", is_final=True, turn_id=2)))

    assert backend.calls == ["Turn two."]
    assert out[0].turn_id == 2


def test_different_session_id_also_resets_the_buffer() -> None:
    backend = _FakeTTSBackend()
    mixin = TtsMixin(backend)

    list(mixin.process(_chunk("leftover", is_final=False, session_id="s1", turn_id=1)))
    list(mixin.process(_chunk("fresh", is_final=True, session_id="s2", turn_id=1)))

    assert backend.calls == ["fresh"]
