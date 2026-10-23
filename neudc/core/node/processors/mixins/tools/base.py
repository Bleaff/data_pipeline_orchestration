"""Shared contract for tools invoked by `AgentToolMixin`."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ("Tool", "ToolError")


class ToolError(Exception):
    """Raised by a tool's `__call__` when it can't complete.

    Caught by `AgentToolMixin`, which turns it into a short spoken error message
    instead of letting it crash the node — a tool call came from LLM-generated text,
    which can be malformed or ask for something the tool legitimately can't do (a
    path outside the sandbox, an unreachable URL), not a bug in our own code.
    """


@runtime_checkable
class Tool(Protocol):
    """Structural interface every tool in the registry satisfies.

    `name` is the marker keyword `AgentToolMixin` matches (`[[name(...)]]`);
    `description` is a one-line usage string folded into `tool_system_prompt()` so
    the LLM knows the tool exists and how to call it.
    """

    name: str
    description: str

    def __call__(self, **kwargs: object) -> str:
        """Execute the tool and return its result as text, or raise `ToolError`."""
        ...
