"""Tests for AgentToolNode.from_config: config dict -> wired tool registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.node.processors.agent_tool_node import AgentToolNode
from neudc.core.node.processors.mixins.tools.file_tools import CreateFileTool, ReadFileTool
from neudc.core.node.processors.mixins.tools.timer_tool import TimerTool
from neudc.core.node.processors.mixins.tools.web_fetch_tool import FetchUrlTool

if TYPE_CHECKING:
    from pathlib import Path


def test_default_config_enables_all_three_tool_groups(tmp_path: Path) -> None:
    node = AgentToolNode.from_config(
        {"mailbox": ZMQMailbox(), "sandbox_dir": str(tmp_path)},
    )

    assert set(node.tools) == {"set_timer", "fetch_url", "create_file", "read_file"}
    assert isinstance(node.tools["set_timer"], TimerTool)
    assert isinstance(node.tools["fetch_url"], FetchUrlTool)
    assert isinstance(node.tools["create_file"], CreateFileTool)
    assert isinstance(node.tools["read_file"], ReadFileTool)


def test_file_io_requires_sandbox_dir() -> None:
    with pytest.raises(ValueError, match="requires 'sandbox_dir'"):
        AgentToolNode.from_config({"mailbox": ZMQMailbox()})


def test_file_io_can_be_disabled_without_sandbox_dir() -> None:
    node = AgentToolNode.from_config(
        {"mailbox": ZMQMailbox(), "enable_file_io": False},
    )

    assert "create_file" not in node.tools
    assert "read_file" not in node.tools
    assert "set_timer" in node.tools
    assert "fetch_url" in node.tools


def test_individual_tool_groups_can_be_disabled(tmp_path: Path) -> None:
    node = AgentToolNode.from_config(
        {
            "mailbox": ZMQMailbox(),
            "sandbox_dir": str(tmp_path),
            "enable_timers": False,
            "enable_web_fetch": False,
        },
    )

    assert set(node.tools) == {"create_file", "read_file"}


def test_timer_tool_uses_the_node_mailbox_and_session_id(tmp_path: Path) -> None:
    mailbox = ZMQMailbox()
    node = AgentToolNode.from_config(
        {"mailbox": mailbox, "sandbox_dir": str(tmp_path), "session_id": "sess-42"},
    )

    timer = node.tools["set_timer"]
    assert timer._mailbox is mailbox
    assert timer._session_id == "sess-42"


def test_fetch_tool_uses_configured_timeout_and_max_chars(tmp_path: Path) -> None:
    node = AgentToolNode.from_config(
        {
            "mailbox": ZMQMailbox(),
            "sandbox_dir": str(tmp_path),
            "fetch_timeout": 5.0,
            "fetch_max_chars": 100,
        },
    )

    fetch = node.tools["fetch_url"]
    assert fetch._timeout == 5.0
    assert fetch._max_chars == 100
