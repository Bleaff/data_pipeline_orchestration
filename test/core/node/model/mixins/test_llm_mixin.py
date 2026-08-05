"""Tests for LlmMixin: conversation history, non-streaming and streaming replies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neudc.core.communication.messaging.types import TextChunk
from neudc.core.node.model.mixins.llm_mixin import LlmMixin

if TYPE_CHECKING:
    from neudc.nn.backends.llm.base import ChatMessage


class _FakeLLMBackend:
    def __init__(self, reply: str = "ok", stream_deltas: list[str] | None = None) -> None:
        self.reply = reply
        self.stream_deltas = stream_deltas or []
        self.generate_calls: list[list[ChatMessage]] = []
        self.stream_calls: list[list[ChatMessage]] = []

    def generate(self, messages, *, temperature=0.7, max_tokens=None):
        self.generate_calls.append(list(messages))
        return self.reply

    def stream(self, messages, *, temperature=0.7, max_tokens=None):
        self.stream_calls.append(list(messages))
        yield from self.stream_deltas


def _chunk(text: str, *, session_id: str = "s1", turn_id: int = 1, drop: bool = False) -> TextChunk:
    return TextChunk(timestamp=1.0, source="asr", text=text, session_id=session_id, turn_id=turn_id, drop=drop)


def test_non_streaming_reply_is_one_final_chunk() -> None:
    backend = _FakeLLMBackend(reply="hi there")
    mixin = LlmMixin(backend)

    out = list(mixin.process(_chunk("hello")))

    assert len(out) == 1
    assert out[0].text == "hi there"
    assert out[0].is_final is True
    assert out[0].session_id == "s1"
    assert out[0].turn_id == 1


def test_system_prompt_seeds_history() -> None:
    backend = _FakeLLMBackend()
    mixin = LlmMixin(backend, system_prompt="be nice")

    list(mixin.process(_chunk("hello")))

    sent = backend.generate_calls[0]
    assert sent[0].role == "system"
    assert sent[0].content == "be nice"
    assert sent[1].role == "user"
    assert sent[1].content == "hello"


def test_history_accumulates_across_turns() -> None:
    backend = _FakeLLMBackend(reply="reply-1")
    mixin = LlmMixin(backend)

    list(mixin.process(_chunk("first")))
    backend.reply = "reply-2"
    list(mixin.process(_chunk("second")))

    second_call_messages = backend.generate_calls[1]
    contents = [m.content for m in second_call_messages]
    assert contents == ["first", "reply-1", "second"]


def test_dropped_chunk_yields_no_reply() -> None:
    backend = _FakeLLMBackend()
    mixin = LlmMixin(backend)

    out = list(mixin.process(_chunk("hello", drop=True)))

    assert out == []
    assert backend.generate_calls == []


def test_blank_text_yields_no_reply() -> None:
    backend = _FakeLLMBackend()
    mixin = LlmMixin(backend)

    out = list(mixin.process(_chunk("   ")))

    assert out == []
    assert backend.generate_calls == []


def test_streaming_yields_deltas_then_trailing_final_chunk() -> None:
    backend = _FakeLLMBackend(stream_deltas=["ab", "cd"])
    mixin = LlmMixin(backend, stream=True)

    out = list(mixin.process(_chunk("hello")))

    assert [c.text for c in out] == ["ab", "cd", ""]
    assert [c.is_final for c in out] == [False, False, True]
    assert all(c.session_id == "s1" and c.turn_id == 1 for c in out)


def test_streaming_appends_full_reply_to_history() -> None:
    backend = _FakeLLMBackend(stream_deltas=["ab", "cd"])
    mixin = LlmMixin(backend, stream=True)

    list(mixin.process(_chunk("hello")))
    list(mixin.process(_chunk("again")))

    second_call_messages = backend.stream_calls[1]
    contents = [m.content for m in second_call_messages]
    assert contents == ["hello", "abcd", "again"]
