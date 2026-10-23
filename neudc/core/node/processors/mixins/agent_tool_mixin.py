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

    def process(self, chunk: TextChunk) -> TextChunk:
        """Resolve every tool-call marker in `chunk.text`, passthrough otherwise.

        Args:
        ----
            chunk: Incoming `TextChunk`, typically `LlmNode`'s reply.

        Returns:
        -------
            The same `TextChunk`, with every recognized marker replaced by that
            tool's result (or a short error message); text outside markers is
            untouched, and a dropped or empty-text chunk passes through as-is.

        """
        if getattr(chunk, "drop", False) or not chunk.text:
            return chunk
        chunk.text = _TOOL_CALL_RE.sub(self._resolve_call, chunk.text)
        # Only when tools are actually offered: with none configured, the LLM was
        # never told the [[...]] syntax means anything, so a literal "[[" in a reply
        # is just text, not a botched tool call — stripping it there would be an
        # unrelated, surprising mutation of ordinary conversation.
        if self.tools and _LEFTOVER_MARKER_RE.search(chunk.text):
            LOGGER.warning(f"Stripping unresolved tool-call-shaped text: {chunk.text!r}")
            chunk.text = _LEFTOVER_MARKER_RE.sub("", chunk.text)
        return chunk

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
