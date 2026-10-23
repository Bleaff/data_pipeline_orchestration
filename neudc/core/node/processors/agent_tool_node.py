"""Threaded node wiring `AgentToolMixin` tool-call resolution into the pipeline."""

from __future__ import annotations

from typing import Any, ClassVar

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.messaging.types import BaseMessage, TextChunk
from neudc.core.node.processors.mixins.agent_tool_mixin import AgentToolMixin
from neudc.core.node.processors.mixins.tools.file_tools import CreateFileTool, ReadFileTool
from neudc.core.node.processors.mixins.tools.timer_tool import TimerTool
from neudc.core.node.processors.mixins.tools.web_fetch_tool import FetchUrlTool


class AgentToolNode(AgentToolMixin, BaseThreadedNode):
    """A node that resolves inline tool-call markers in `TextChunk` text (see `AgentToolMixin`).

    Sits between `LlmNode` and `TtsNode`: `LlmNode -> AgentToolNode -> TtsNode`. Tools
    run autonomously, with no per-call confirmation, but side-effecting ones
    (`create_file`/`read_file`) are confined to a fixed `sandbox_dir`, never touching
    the filesystem outside it — network (`fetch_url`) and timers have their own,
    narrower guards (see each tool's docstring) instead of a sandbox directory.
    """

    accepts: ClassVar[tuple[type[BaseMessage], ...]] = (TextChunk,)
    emits: ClassVar[tuple[type[BaseMessage], ...]] = (TextChunk,)

    def __init__(self, tools: dict[str, Any], mailbox: Any, *, id: str = "AgentToolNode") -> None:
        """Initialize the AgentToolNode.

        Args:
        ----
            tools: Tool name -> callable tool, as built by `from_config`.
            mailbox: Mailbox for sending/receiving `TextChunk` objects.
            id: Identifier for this node instance.

        """
        AgentToolMixin.__init__(self, tools=tools)
        BaseThreadedNode.__init__(self, mailbox, id=id)

    @staticmethod
    def from_config(config: dict[str, Any]) -> AgentToolNode:
        """Build an AgentToolNode with the tools enabled by config.

        Args:
        ----
            config (dict): Configuration with key `mailbox` and, optionally:
                `enable_timers` (default True), `enable_web_fetch` (default True),
                `enable_file_io` (default True — requires `sandbox_dir`),
                `sandbox_dir` (root directory `create_file`/`read_file` are confined
                to, created if missing), `session_id` (stamped on timer reminders,
                should match `VadNode`'s so a reminder lands on the active turn),
                `fetch_timeout` (default 10.0), `fetch_max_chars` (default 1500).

        Returns:
        -------
            AgentToolNode: Instantiated AgentToolNode.

        Raises:
        ------
            ValueError: If `enable_file_io` is set without a `sandbox_dir`.

        """
        mailbox = config["mailbox"]
        tools: dict[str, Any] = {}

        if config.get("enable_timers", True):
            timer = TimerTool(mailbox=mailbox, session_id=config.get("session_id", "default"))
            tools[timer.name] = timer

        if config.get("enable_web_fetch", True):
            fetch = FetchUrlTool(
                timeout=config.get("fetch_timeout", 10.0),
                max_chars=config.get("fetch_max_chars", 1500),
            )
            tools[fetch.name] = fetch

        if config.get("enable_file_io", True):
            sandbox_dir = config.get("sandbox_dir")
            if not sandbox_dir:
                msg = "AgentToolNode.from_config requires 'sandbox_dir' when 'enable_file_io' is true"
                raise ValueError(msg)
            create_file = CreateFileTool(sandbox_dir=sandbox_dir)
            read_file = ReadFileTool(sandbox_dir=sandbox_dir)
            tools[create_file.name] = create_file
            tools[read_file.name] = read_file

        return AgentToolNode(tools=tools, mailbox=mailbox)
