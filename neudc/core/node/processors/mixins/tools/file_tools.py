"""Sandboxed file I/O tools: create/read files confined to a fixed sandbox directory."""

from __future__ import annotations

from pathlib import Path

from neudc.core.node.processors.mixins.tools.base import ToolError

__all__ = ("CreateFileTool", "ReadFileTool")

#: Guards against an LLM asked to "write a file" dumping something unbounded.
MAX_FILE_CONTENT_CHARS = 20_000
#: Kept short enough for a `read_file` result to fold into a spoken reply.
MAX_READ_CHARS = 4_000


def _resolve_in_sandbox(sandbox_root: Path, relative_path: str) -> Path:
    """Resolve `relative_path` against `sandbox_root`, refusing anything that escapes it.

    Both `../` traversal and an absolute path are caught the same way: resolve first,
    then check the result is still inside `sandbox_root` — there is no path shape
    that can bypass this by construction.
    """
    candidate = (sandbox_root / relative_path).resolve()
    if not candidate.is_relative_to(sandbox_root):
        msg = f"path {relative_path!r} escapes the sandbox directory"
        raise ToolError(msg)
    return candidate


class CreateFileTool:
    """`create_file`: writes text content to a path confined to `sandbox_dir`."""

    name = "create_file"
    description = (
        'create_file({"path": <string>, "content": <string>}) - creates or overwrites a file '
        "inside the assistant's sandbox directory."
    )

    def __init__(self, sandbox_dir: str) -> None:
        """Initialize the tool, creating `sandbox_dir` if it doesn't exist yet."""
        self._sandbox_root = Path(sandbox_dir).resolve()
        self._sandbox_root.mkdir(parents=True, exist_ok=True)

    def __call__(self, path: str, content: str) -> str:
        """Write `content` to `path`, confined to the sandbox.

        Args:
        ----
            path: Path relative to the sandbox root. Absolute paths and `../`
                traversal outside the sandbox are rejected.
            content: Text to write.

        Returns:
        -------
            A confirmation string.

        Raises:
        ------
            ToolError: If `path` escapes the sandbox, `content` is too long, or the
                write fails.

        """
        if len(content) > MAX_FILE_CONTENT_CHARS:
            msg = f"content exceeds {MAX_FILE_CONTENT_CHARS} characters"
            raise ToolError(msg)
        target = _resolve_in_sandbox(self._sandbox_root, path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            msg = f"failed to write {path!r}: {e}"
            raise ToolError(msg) from e
        return f"Created {path}."


class ReadFileTool:
    """`read_file`: reads text content from a path confined to `sandbox_dir`."""

    name = "read_file"
    description = 'read_file({"path": <string>}) - reads a file from the assistant\'s sandbox directory.'

    def __init__(self, sandbox_dir: str) -> None:
        """Initialize the tool, creating `sandbox_dir` if it doesn't exist yet."""
        self._sandbox_root = Path(sandbox_dir).resolve()
        self._sandbox_root.mkdir(parents=True, exist_ok=True)

    def __call__(self, path: str) -> str:
        """Read and return the text content of `path`, confined to the sandbox.

        Args:
        ----
            path: Path relative to the sandbox root.

        Returns:
        -------
            The file's content, truncated to `MAX_READ_CHARS`.

        Raises:
        ------
            ToolError: If `path` escapes the sandbox, doesn't exist, or can't be read.

        """
        target = _resolve_in_sandbox(self._sandbox_root, path)
        if not target.is_file():
            msg = f"{path!r} does not exist"
            raise ToolError(msg)
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as e:
            msg = f"failed to read {path!r}: {e}"
            raise ToolError(msg) from e
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + "…"
        return text
