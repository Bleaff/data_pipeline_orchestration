"""Tests for OpenAICompatibleBackend request/response handling, hermetic via a fake session (#25)."""

from __future__ import annotations

import pytest

from neudc.nn.backends.llm.base import ChatMessage
from neudc.nn.backends.llm.openai_compatible import OpenAICompatibleBackend


class _FakeResponse:
    def __init__(self, json_data=None, lines=None, raw_lines=None):
        self._json = json_data
        self._lines = lines or []
        # Encoding-sensitive lines, as raw bytes — mirrors real `requests` behavior
        # (see test_stream_forces_utf8_encoding_before_decoding_sse_lines below) where
        # `iter_lines(decode_unicode=True)` decodes each raw line using `.encoding`.
        # `None` here matches `requests`' own default before anything sets it.
        self._raw_lines = raw_lines
        self.encoding = None

    def raise_for_status(self):
        pass

    def json(self):
        return self._json

    def iter_lines(self, decode_unicode=True):
        if self._raw_lines is not None:
            codec = self.encoding or "ISO-8859-1"
            for raw in self._raw_lines:
                yield raw.decode(codec, errors="replace")
            return
        yield from self._lines


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.last_call = None

    def post(self, url, json, headers, timeout, stream=False):
        self.last_call = {"url": url, "json": json, "headers": headers, "timeout": timeout, "stream": stream}
        return self.response

    def close(self):
        pass


def test_requires_base_url():
    with pytest.raises(ValueError, match="requires a base_url"):
        OpenAICompatibleBackend("some-model")


def test_headers_include_bearer_token_when_api_key_set():
    backend = OpenAICompatibleBackend("m", base_url="http://fake.test/v1", api_key="secret")

    assert backend._headers()["Authorization"] == "Bearer secret"


def test_headers_omit_auth_when_no_api_key():
    backend = OpenAICompatibleBackend("m", base_url="http://fake.test/v1")

    assert "Authorization" not in backend._headers()


def test_generate_returns_message_content():
    backend = OpenAICompatibleBackend("test-model", base_url="http://fake.test/v1")
    backend._session = _FakeSession(_FakeResponse(json_data={"choices": [{"message": {"content": "hello"}}]}))

    out = backend.generate([ChatMessage(role="user", content="hi")])

    assert out == "hello"
    assert backend._session.last_call["json"]["stream"] is False
    assert backend._session.last_call["json"]["model"] == "test-model"
    assert backend._session.last_call["url"] == "http://fake.test/v1/chat/completions"


def test_generate_includes_max_tokens_when_given():
    backend = OpenAICompatibleBackend("test-model", base_url="http://fake.test/v1")
    backend._session = _FakeSession(_FakeResponse(json_data={"choices": [{"message": {"content": "hi"}}]}))

    backend.generate([ChatMessage(role="user", content="hi")], max_tokens=42)

    assert backend._session.last_call["json"]["max_tokens"] == 42


def test_stream_yields_content_deltas_and_stops_at_done():
    backend = OpenAICompatibleBackend("test-model", base_url="http://fake.test/v1")
    lines = [
        "",  # keep-alive blank lines must be skipped
        'data: {"choices": [{"delta": {"content": "ab"}}]}',
        'data: {"choices": [{"delta": {"content": "cd"}}]}',
        "data: [DONE]",
        'data: {"choices": [{"delta": {"content": "should not appear"}}]}',
    ]
    backend._session = _FakeSession(_FakeResponse(lines=lines))

    out = list(backend.stream([ChatMessage(role="user", content="hi")]))

    assert out == ["ab", "cd"]
    assert backend._session.last_call["json"]["stream"] is True


def test_stream_skips_malformed_chunks():
    backend = OpenAICompatibleBackend("test-model", base_url="http://fake.test/v1")
    lines = ["data: not-json", 'data: {"choices": [{"delta": {"content": "ok"}}]}']
    backend._session = _FakeSession(_FakeResponse(lines=lines))

    out = list(backend.stream([ChatMessage(role="user", content="hi")]))

    assert out == ["ok"]


def test_stream_forces_utf8_encoding_before_decoding_sse_lines():
    # Regression: `requests` defaults `.encoding` to ISO-8859-1 for any `text/*`
    # response whose Content-Type omits an explicit charset — `text/event-stream`
    # from llama-server (and most OpenAI-compatible servers) does exactly that.
    # Left alone, `iter_lines(decode_unicode=True)` decodes every SSE line with that
    # wrong codec: any multi-byte UTF-8 character (Cyrillic among others) comes out
    # as mojibake instead of the real text. `stream()` must force `.encoding = "utf-8"`
    # itself, since the server never declares it and OpenAI-compatible SSE is always
    # UTF-8 regardless. Verified directly against the real deployed llama-server
    # during manual testing (response.encoding was indeed "ISO-8859-1" by default).
    backend = OpenAICompatibleBackend("test-model", base_url="http://fake.test/v1")
    raw_lines = [
        'data: {"choices": [{"delta": {"content": "Привет"}}]}'.encode(),
        b"data: [DONE]",
    ]
    response = _FakeResponse(raw_lines=raw_lines)
    backend._session = _FakeSession(response)

    out = list(backend.stream([ChatMessage(role="user", content="hi")]))

    assert out == ["Привет"]
    assert response.encoding == "utf-8"
