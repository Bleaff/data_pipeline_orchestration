"""Tests for CreateFileTool/ReadFileTool: sandbox confinement is the whole point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from neudc.core.node.processors.mixins.tools.base import ToolError
from neudc.core.node.processors.mixins.tools.file_tools import (
    MAX_FILE_CONTENT_CHARS,
    MAX_READ_CHARS,
    CreateFileTool,
    ReadFileTool,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_create_then_read_round_trips(tmp_path: Path) -> None:
    create = CreateFileTool(sandbox_dir=str(tmp_path))
    read = ReadFileTool(sandbox_dir=str(tmp_path))

    result = create(path="notes.txt", content="hello sandbox")

    assert "notes.txt" in result
    assert read(path="notes.txt") == "hello sandbox"


def test_create_makes_intermediate_directories(tmp_path: Path) -> None:
    create = CreateFileTool(sandbox_dir=str(tmp_path))

    create(path="nested/dir/notes.txt", content="deep")

    assert (tmp_path / "nested" / "dir" / "notes.txt").read_text() == "deep"


def test_sandbox_dir_is_created_if_missing(tmp_path: Path) -> None:
    sandbox = tmp_path / "does" / "not" / "exist"

    CreateFileTool(sandbox_dir=str(sandbox))

    assert sandbox.is_dir()


@pytest.mark.parametrize("path", ["../escape.txt", "../../etc/passwd", "/etc/passwd"])
def test_create_rejects_paths_escaping_the_sandbox(tmp_path: Path, path: str) -> None:
    create = CreateFileTool(sandbox_dir=str(tmp_path))

    with pytest.raises(ToolError, match="escapes the sandbox"):
        create(path=path, content="pwned")


@pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd"])
def test_read_rejects_paths_escaping_the_sandbox(tmp_path: Path, path: str) -> None:
    read = ReadFileTool(sandbox_dir=str(tmp_path))

    with pytest.raises(ToolError, match="escapes the sandbox"):
        read(path=path)


def test_read_missing_file_raises_tool_error(tmp_path: Path) -> None:
    read = ReadFileTool(sandbox_dir=str(tmp_path))

    with pytest.raises(ToolError, match="does not exist"):
        read(path="nope.txt")


def test_create_rejects_content_over_the_limit(tmp_path: Path) -> None:
    create = CreateFileTool(sandbox_dir=str(tmp_path))

    with pytest.raises(ToolError, match="exceeds"):
        create(path="big.txt", content="x" * (MAX_FILE_CONTENT_CHARS + 1))


def test_read_truncates_content_over_the_limit(tmp_path: Path) -> None:
    create = CreateFileTool(sandbox_dir=str(tmp_path))
    read = ReadFileTool(sandbox_dir=str(tmp_path))
    long_content = "x" * (MAX_READ_CHARS + 500)
    (tmp_path / "long.txt").write_text(long_content)
    del create

    result = read(path="long.txt")

    assert len(result) == MAX_READ_CHARS + 1  # + ellipsis
    assert result.endswith("…")


def test_create_overwrites_existing_file(tmp_path: Path) -> None:
    create = CreateFileTool(sandbox_dir=str(tmp_path))
    create(path="notes.txt", content="first")

    create(path="notes.txt", content="second")

    assert (tmp_path / "notes.txt").read_text() == "second"
