"""Tests for FetchUrlTool: hermetic via a fake session, no real network access."""

from __future__ import annotations

import pytest
import requests

from neudc.core.node.processors.mixins.tools.base import ToolError
from neudc.core.node.processors.mixins.tools.web_fetch_tool import FetchUrlTool


class _FakeRaw:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def read(self, size: int, decode_content: bool = True) -> bytes:
        del decode_content
        return self._content[:size]


class _FakeResponse:
    def __init__(self, content: bytes = b"", status_error: Exception | None = None) -> None:
        self.raw = _FakeRaw(content)
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.last_call = None

    def get(self, url, timeout, stream):
        self.last_call = {"url": url, "timeout": timeout, "stream": stream}
        return self.response

    def close(self) -> None:
        pass


def _tool_with(response: _FakeResponse, **kwargs) -> FetchUrlTool:
    tool = FetchUrlTool(**kwargs)
    tool._session = _FakeSession(response)
    return tool


def test_extracts_visible_text_and_skips_script_style() -> None:
    html = b"<html><head><style>.a{}</style></head><body><script>evil()</script><p>Hello world</p></body></html>"
    tool = _tool_with(_FakeResponse(content=html))

    result = tool("http://example.test/page")

    assert result == "Hello world"


def test_rejects_non_http_scheme() -> None:
    tool = _tool_with(_FakeResponse())

    with pytest.raises(ToolError, match="unsupported URL scheme"):
        tool("file:///etc/passwd")


def test_rejects_url_without_scheme() -> None:
    tool = _tool_with(_FakeResponse())

    with pytest.raises(ToolError, match="unsupported URL scheme"):
        tool("example.test/page")


def test_raises_tool_error_on_request_exception() -> None:
    tool = FetchUrlTool()

    class _FailingSession:
        def get(self, url, timeout, stream):
            del url, timeout, stream
            msg = "boom"
            raise requests.ConnectionError(msg)

        def close(self) -> None:
            pass

    tool._session = _FailingSession()

    with pytest.raises(ToolError, match="failed to fetch"):
        tool("http://example.test/page")


def test_raises_tool_error_on_http_error_status() -> None:
    tool = _tool_with(_FakeResponse(status_error=requests.HTTPError("404")))

    with pytest.raises(ToolError):
        tool("http://example.test/missing")


def test_rejects_response_larger_than_max_bytes() -> None:
    tool = _tool_with(_FakeResponse(content=b"<p>" + b"x" * 100 + b"</p>"), max_response_bytes=10)

    with pytest.raises(ToolError, match="exceeds"):
        tool("http://example.test/big")


def test_truncates_text_to_max_chars() -> None:
    html = ("<p>" + "word " * 100 + "</p>").encode()
    tool = _tool_with(_FakeResponse(content=html), max_chars=20)

    result = tool("http://example.test/long")

    assert len(result) <= 21  # 20 + ellipsis
    assert result.endswith("…")


def test_empty_page_returns_placeholder() -> None:
    tool = _tool_with(_FakeResponse(content=b"<html><body></body></html>"))

    result = tool("http://example.test/empty")

    assert result == "(page had no readable text)"
