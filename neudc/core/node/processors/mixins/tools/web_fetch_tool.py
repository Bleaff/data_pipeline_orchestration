"""Web-page fetch/parse tool: downloads a URL and extracts its visible text."""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

import requests

from neudc.core.node.processors.mixins.tools.base import ToolError

__all__ = ("FetchUrlTool",)

_ALLOWED_SCHEMES = ("http", "https")
_SKIP_TAGS = frozenset({"script", "style", "noscript"})


class _TextExtractor(HTMLParser):
    """Collects visible text nodes, skipping `<script>`/`<style>`/`<noscript>` content."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self.chunks.append(stripped)


def _extract_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    return " ".join(extractor.chunks)


class FetchUrlTool:
    """`fetch_url`: downloads a page and returns its extracted, truncated visible text.

    Uses the stdlib `html.parser` rather than a new dependency (e.g. BeautifulSoup) —
    good enough to strip markup for a short spoken summary, not a general-purpose
    scraper. Only `http`/`https` schemes are allowed. Response size and extracted
    text length are both capped, but this is a page-size guard, not a security
    boundary: the LLM is the only caller of this tool (invoked via
    `AgentToolMixin`'s marker protocol), not an untrusted third party reaching it
    directly.
    """

    name = "fetch_url"
    description = 'fetch_url({"url": <string>}) - fetches a web page and returns its visible text (truncated).'

    def __init__(self, *, timeout: float = 10.0, max_response_bytes: int = 2_000_000, max_chars: int = 1500) -> None:
        """Initialize the tool.

        Args:
        ----
            timeout: HTTP request timeout, in seconds.
            max_response_bytes: Response bodies larger than this are rejected rather
                than parsed in full.
            max_chars: Extracted text is truncated to this many characters — short
                enough for the LLM to fold into a spoken reply.

        """
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._max_chars = max_chars
        self._session = requests.Session()

    def __call__(self, url: str) -> str:
        """Fetch `url` and return its extracted, truncated visible text.

        Args:
        ----
            url: The page to fetch. Must be `http://` or `https://`.

        Returns:
        -------
            The page's visible text, truncated to `max_chars`.

        Raises:
        ------
            ToolError: If the scheme isn't allowed, the request fails, or the
                response exceeds `max_response_bytes`.

        """
        scheme = url.split("://", 1)[0].lower() if "://" in url else ""
        if scheme not in _ALLOWED_SCHEMES:
            msg = f"unsupported URL scheme {scheme!r} — only {_ALLOWED_SCHEMES} are allowed"
            raise ToolError(msg)

        content = self._download(url)
        if len(content) > self._max_response_bytes:
            msg = f"response from {url} exceeds {self._max_response_bytes} bytes, refusing to parse"
            raise ToolError(msg)

        text = _extract_text(content.decode("utf-8", errors="replace"))
        if len(text) > self._max_chars:
            text = text[: self._max_chars].rsplit(" ", 1)[0] + "…"
        return text or "(page had no readable text)"

    def _download(self, url: str) -> bytes:
        try:
            response = self._session.get(url, timeout=self._timeout, stream=True)
            response.raise_for_status()
            return response.raw.read(self._max_response_bytes + 1, decode_content=True)
        except requests.RequestException as e:
            msg = f"failed to fetch {url}: {e}"
            raise ToolError(msg) from e

    def __del__(self) -> None:
        """Release the underlying HTTP session."""
        session: Any = getattr(self, "_session", None)
        if session is not None:
            session.close()
