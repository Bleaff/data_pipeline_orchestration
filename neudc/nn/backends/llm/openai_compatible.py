"""HTTP client shared by every provider that speaks the OpenAI chat-completions wire format."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import requests

from neudc.nn.backends.llm.base import BaseLLMBackend
from neudc.utils import LOGGER

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neudc.nn.backends.llm.base import ChatMessage

__all__ = ("OpenAICompatibleBackend",)

_CHAT_COMPLETIONS_PATH = "/chat/completions"
_SSE_DATA_PREFIX = "data: "
_SSE_DONE_MARKER = "[DONE]"


class OpenAICompatibleBackend(BaseLLMBackend):
    """Talks to any `/chat/completions` endpoint following the OpenAI wire format.

    OpenRouter and local runtimes (Ollama, vLLM, llama.cpp) all expose this same
    protocol, so this class holds the request/response handling once; provider
    subclasses only fix the endpoint and credential resolution.
    """

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the OpenAICompatibleBackend class.

        Args:
        ----
            model_id (str): Provider-specific model identifier.
            base_url (str | None): Root URL of the OpenAI-compatible API (no trailing slash needed).
            api_key (str | None): Bearer token sent as `Authorization`, when the provider requires one.
            timeout (float): Request timeout in seconds.

        """
        if not base_url:
            msg = f"{type(self).__name__} requires a base_url"
            raise ValueError(msg)

        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> dict:
        payload = {
            "model": self.model_id,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Run a single, non-streaming chat completion.

        Args:
        ----
            messages (list[ChatMessage]): The chat history to complete.
            temperature (float): Sampling temperature.
            max_tokens (int | None): Upper bound on generated tokens, provider default if `None`.

        Returns:
        -------
            str: The assistant's full reply.

        """
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens, stream=False)
        response = self._session.post(
            f"{self.base_url}{_CHAT_COMPLETIONS_PATH}",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Run a streaming chat completion, yielding incremental content deltas.

        Args:
        ----
            messages (list[ChatMessage]): The chat history to complete.
            temperature (float): Sampling temperature.
            max_tokens (int | None): Upper bound on generated tokens, provider default if `None`.

        Yields:
        ------
            str: Successive content chunks as the provider streams them (SSE `data:` frames).

        """
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        response = self._session.post(
            f"{self.base_url}{_CHAT_COMPLETIONS_PATH}",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith(_SSE_DATA_PREFIX):
                continue
            data_str = raw_line[len(_SSE_DATA_PREFIX) :].strip()
            if data_str == _SSE_DONE_MARKER:
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                LOGGER.warning(f"WARNING ⚠️ Skipping malformed SSE chunk from {self.base_url}: {data_str!r}")
                continue
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta

    def __del__(self) -> None:
        """Release the underlying HTTP session."""
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()
