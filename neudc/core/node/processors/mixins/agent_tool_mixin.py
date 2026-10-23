"""Agent tool-use mixin: resolves inline tool-call markers in LLM output.

Protocol: the LLM is instructed (via a system-prompt fragment, see
`tool_system_prompt()`) to emit `[[tool_name({"arg": "value"})]]` inline wherever it
wants to invoke a tool. This mixin scans each incoming `TextChunk.text` for such
markers, executes each matched tool synchronously and in order, and replaces the
marker with the tool's returned text — or a short `[tool error: ...]` message if the
tool raised `ToolError`, the name is unknown, or the arguments didn't parse — so the
reply that reaches `TtsMixin` downstream is always plain, speakable text, never raw
tool-call syntax.

Deliberately a single pass, not a multi-turn "ReAct" loop: a tool's result is spliced
into the same reply the LLM already generated rather than fed back for a second LLM
call. This keeps the wiring a straight line (`llm -> tool -> tts`, no cycle back into
`LlmNode`) at the cost of the LLM never seeing a tool's result before it finishes
composing the sentence around the call — fine for confirmation-style tools (a timer
was set, a file was saved) where the LLM doesn't need the result to phrase a sensible
reply, weaker for a tool whose result should shape the wording (e.g. summarizing a
fetched page in different words than the page itself uses). A second pass through
`LlmNode` is a natural follow-up, not built here.

Works with a streaming `LlmNode` (`stream: true`): each delta is one token or so, so
a marker almost always arrives split across several chunks. Text from an unclosed
`[[` onward is held back and prepended to the next chunk of the same turn, then
flushed (resolved or stripped) on the turn's `is_final` chunk — a marker is only
ever matched against complete text, never a fragment.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from neudc.core.node.processors.mixins.tools.base import ToolError
from neudc.utils import LOGGER

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import TextChunk
    from neudc.core.node.processors.mixins.tools.base import Tool

__all__ = ("AgentToolMixin", "tool_system_prompt")

#: Non-greedy match on a flat JSON object argument — deliberately simple (no nested
#: parens/braces tracking); good enough for the small, flat argument shapes the
#: shipped tools take (numbers/strings, no nested objects). Tolerates incidental
#: whitespace around the parens/braces (`[[ name( {...} ) ]]`) since an LLM asked to
#: follow an exact textual format doesn't always reproduce it byte-for-byte.
_TOOL_CALL_RE = re.compile(r"\[\[\s*(\w+)\s*\(\s*(\{.*?\})\s*\)\s*\]\]", re.DOTALL)

#: Catches anything left over that still *looks* like a tool-call marker (`[[...]]`)
#: after `_TOOL_CALL_RE` has already resolved every well-formed one — a malformed
#: call (bad JSON, extra text inside the brackets) that would otherwise reach `TtsMixin`
#: and get read aloud literally, exactly the "double brackets" failure mode this guards.
_LEFTOVER_MARKER_RE = re.compile(r"\[\[.*?\]\]", re.DOTALL)


def _incomplete_marker_start(text: str) -> int:
    """Index where a possibly-unfinished tool-call marker starts in `text`, or `len(text)` if none.

    The last `[[` with no `]]` after it is unfinished; so is a lone trailing `[`,
    which the next streamed delta may turn into `[[`.
    """
    start = text.rfind("[[")
    if start != -1 and "]]" not in text[start + 2 :]:
        return start
    if text.endswith("["):
        return len(text) - 1
    return len(text)


def tool_system_prompt(tools: dict[str, Tool]) -> str:
    """Build a system-prompt fragment describing available tools and how to call them.

    Meant to be appended to `LlmNode`'s `system_prompt` — an LLM given no such
    instruction has no reason to ever emit the `[[...]]` marker `AgentToolMixin`
    looks for.

    Args:
    ----
        tools: The same tool registry passed to `AgentToolMixin`.

    Returns:
    -------
        A prompt fragment, or `""` if `tools` is empty.

    """
    if not tools:
        return ""
    lines = [
        (
            "You can use tools by writing exactly one marker per call, anywhere in your reply, "
            'in the form [[tool_name({"arg": "value"})]] — the arguments must be valid JSON. '
            "Available tools:"
        ),
    ]
    lines.extend(f"- {tool.description}" for tool in tools.values())
    return "\n".join(lines)


class AgentToolMixin:
    """Mixin resolving `[[tool_name({...})]]` markers in `TextChunk.text` via a tool registry."""

    def __init__(self, tools: dict[str, Tool]) -> None:
        """Initialize the mixin.

        Args:
        ----
            tools: Tool name -> callable tool (see `neudc/core/node/processors/mixins/tools/`),
                as built by `AgentToolNode.from_config`.

        """
        self.tools = tools
        self._pending = ""
        self._pending_turn_key: tuple[str | None, int | None] | None = None

    def process(self, chunk: TextChunk) -> TextChunk:
        """Resolve every tool-call marker in `chunk.text`, passthrough otherwise.

        A non-final chunk whose text ends inside a possible marker (an unclosed `[[`,
        or a trailing `[`) has that tail held back until the next chunk of the same
        (session_id, turn_id) completes it; the `is_final` chunk flushes whatever is
        held. A chunk from a different turn discards held text first — that turn was
        cancelled before its final chunk arrived (see `VadFilterMixin`).

        Args:
        ----
            chunk: Incoming `TextChunk`, typically `LlmNode`'s reply or one delta of it.

        Returns:
        -------
            The same `TextChunk`, with every recognized marker replaced by that
            tool's result (or a short error message) and any held-back tail removed;
            text outside markers is untouched, and a dropped chunk passes through as-is.

        """
        if getattr(chunk, "drop", False):
            return chunk

        turn_key = (chunk.session_id, chunk.turn_id)
        if turn_key != self._pending_turn_key:
            self._pending = ""
            self._pending_turn_key = turn_key

        text = self._pending + chunk.text
        if chunk.is_final:
            ready, self._pending = text, ""
            self._pending_turn_key = None
        else:
            split = _incomplete_marker_start(text)
            ready, self._pending = text[:split], text[split:]

        chunk.text = self._resolve_text(ready)
        return chunk

    def _resolve_text(self, text: str) -> str:
        if not text:
            return text
        text = _TOOL_CALL_RE.sub(self._resolve_call, text)
        # Only when tools are actually offered: with none configured, the LLM was
        # never told the [[...]] syntax means anything, so a literal "[[" in a reply
        # is just text, not a botched tool call — stripping it there would be an
        # unrelated, surprising mutation of ordinary conversation.
        if self.tools and _LEFTOVER_MARKER_RE.search(text):
            LOGGER.warning(f"Stripping unresolved tool-call-shaped text: {text!r}")
            text = _LEFTOVER_MARKER_RE.sub("", text)
        return text

    def _resolve_call(self, match: re.Match[str]) -> str:
        tool_name, raw_args = match.group(1), match.group(2)
        tool = self.tools.get(tool_name)
        if tool is None:
            LOGGER.warning(f"Unknown tool call {tool_name!r}, leaving unresolved")
            return f"[tool error: unknown tool {tool_name!r}]"

        try:
            kwargs = json.loads(raw_args)
        except json.JSONDecodeError as e:
            LOGGER.warning(f"Malformed arguments for tool {tool_name!r}: {e}")
            return f"[tool error: malformed arguments for {tool_name!r}]"

        try:
            return tool(**kwargs)
        except ToolError as e:
            LOGGER.warning(f"Tool {tool_name!r} failed: {e}")
            return f"[tool error: {e}]"
        except TypeError as e:
            LOGGER.warning(f"Tool {tool_name!r} called with bad arguments: {e}")
            return f"[tool error: bad arguments for {tool_name!r}]"
