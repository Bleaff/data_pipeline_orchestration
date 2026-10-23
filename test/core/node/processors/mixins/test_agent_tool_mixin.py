"""Tests for AgentToolMixin: tool-call marker parsing, execution, and substitution."""

from __future__ import annotations

from neudc.core.communication.messaging.types import TextChunk
from neudc.core.node.processors.mixins.agent_tool_mixin import AgentToolMixin, tool_system_prompt
from neudc.core.node.processors.mixins.tools.base import ToolError


class _FakeTool:
    name = "fake_tool"
    description = 'fake_tool({"x": <number>}) - test tool.'

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return f"fake-result-{kwargs.get('x')}"


class _FailingTool:
    name = "failing_tool"
    description = "failing_tool({}) - always fails."

    def __call__(self, **kwargs) -> str:
        del kwargs
        msg = "always fails"
        raise ToolError(msg)


def _chunk(text: str, *, drop: bool = False) -> TextChunk:
    return TextChunk(timestamp=1.0, source="llm", text=text, drop=drop)


def test_text_without_markers_passes_through_unchanged() -> None:
    mixin = AgentToolMixin(tools={})

    result = mixin.process(_chunk("just a normal reply"))

    assert result.text == "just a normal reply"


def test_dropped_chunk_is_returned_unresolved() -> None:
    tool = _FakeTool()
    mixin = AgentToolMixin(tools={tool.name: tool})

    result = mixin.process(_chunk('call: [[fake_tool({"x": 1})]]', drop=True))

    assert result.text == 'call: [[fake_tool({"x": 1})]]'
    assert tool.calls == []


def test_resolves_a_single_tool_call() -> None:
    tool = _FakeTool()
    mixin = AgentToolMixin(tools={tool.name: tool})

    result = mixin.process(_chunk('Sure, one sec: [[fake_tool({"x": 42})]]'))

    assert result.text == "Sure, one sec: fake-result-42"
    assert tool.calls == [{"x": 42}]


def test_resolves_multiple_tool_calls_in_order() -> None:
    tool = _FakeTool()
    mixin = AgentToolMixin(tools={tool.name: tool})

    result = mixin.process(_chunk('[[fake_tool({"x": 1})]] then [[fake_tool({"x": 2})]]'))

    assert result.text == "fake-result-1 then fake-result-2"
    assert tool.calls == [{"x": 1}, {"x": 2}]


def test_unknown_tool_name_becomes_an_error_message() -> None:
    mixin = AgentToolMixin(tools={})

    result = mixin.process(_chunk('[[nonexistent({"x": 1})]]'))

    assert "unknown tool" in result.text
    assert "nonexistent" in result.text


def test_malformed_json_arguments_become_an_error_message() -> None:
    tool = _FakeTool()
    mixin = AgentToolMixin(tools={tool.name: tool})

    result = mixin.process(_chunk("[[fake_tool({not valid json})]]"))

    assert "malformed arguments" in result.text
    assert tool.calls == []


def test_tool_error_becomes_a_short_spoken_error_message() -> None:
    tool = _FailingTool()
    mixin = AgentToolMixin(tools={tool.name: tool})

    result = mixin.process(_chunk("[[failing_tool({})]]"))

    assert "tool error" in result.text
    assert "always fails" in result.text


def test_bad_arguments_raise_type_error_and_become_an_error_message() -> None:
    tool = _FakeTool()
    mixin = AgentToolMixin(tools={tool.name: tool})

    # _FakeTool.__call__ accepts **kwargs so it never actually raises TypeError on
    # its own; use a tool with a strict signature to exercise the TypeError path.
    class _StrictTool:
        name = "strict_tool"
        description = "strict_tool({}) - no args allowed."

        def __call__(self) -> str:
            return "ok"

    strict = _StrictTool()
    mixin = AgentToolMixin(tools={strict.name: strict})

    result = mixin.process(_chunk('[[strict_tool({"unexpected": 1})]]'))

    assert "bad arguments" in result.text


def test_empty_text_passes_through() -> None:
    mixin = AgentToolMixin(tools={})

    result = mixin.process(_chunk(""))

    assert result.text == ""


def test_tool_system_prompt_lists_descriptions() -> None:
    tool = _FakeTool()

    prompt = tool_system_prompt({tool.name: tool})

    assert "fake_tool" in prompt
    assert tool.description in prompt


def test_tool_system_prompt_empty_for_no_tools() -> None:
    assert tool_system_prompt({}) == ""


def test_resolves_a_call_with_incidental_whitespace() -> None:
    # An LLM asked to follow an exact textual format doesn't always reproduce it
    # byte-for-byte — stray spaces around parens/braces shouldn't leak as raw text.
    tool = _FakeTool()
    mixin = AgentToolMixin(tools={tool.name: tool})

    result = mixin.process(_chunk('[[ fake_tool( {"x": 7} ) ]]'))

    assert result.text == "fake-result-7"


def test_strips_leftover_unresolved_marker_shaped_text() -> None:
    # Malformed call (invalid JSON) that _TOOL_CALL_RE can't match at all — must not
    # reach TTS as raw "[[...]]" syntax read aloud literally.
    tool = _FakeTool()
    mixin = AgentToolMixin(tools={tool.name: tool})

    result = mixin.process(_chunk("Sure: [[fake_tool(not json at all)]] done."))

    assert "[[" not in result.text
    assert "]]" not in result.text
    assert result.text == "Sure:  done."


def test_leftover_marker_not_stripped_when_no_tools_configured() -> None:
    # With no tools offered, "[[" in a reply was never advertised as meaning
    # anything — stripping it would be an unrelated, surprising text mutation.
    mixin = AgentToolMixin(tools={})

    result = mixin.process(_chunk("some [[literal]] brackets in conversation"))

    assert result.text == "some [[literal]] brackets in conversation"
