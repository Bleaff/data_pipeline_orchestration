"""LLM backend abstraction: the common interface pluggable LLM/VLM providers implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ("BaseLLMBackend", "ChatMessage", "LLMBackendType")


class LLMBackendType(StrEnum):
    """Identifiers for the LLM providers supported by `get_llm_backend`."""

    OPENROUTER = "OpenRouterBackend"
    LOCAL = "LocalLLMBackend"


class ChatMessage(BaseModel):
    """A single turn in a chat completion request, mirroring the OpenAI wire format."""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        """Return the OpenAI-compatible wire representation of this message."""
        return {"role": self.role, "content": self.content}


class BaseLLMBackend(ABC):
    """Common interface every concrete LLM provider (OpenRouter, local OpenAI-compatible, ...) implements."""

    model_id: str

    @abstractmethod
    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Init of BaseLLMBackend.

        Args:
        ----
            model_id (str): Provider-specific model identifier (e.g. `"openai/gpt-4o"`).
            base_url (str | None): Provider endpoint, when it is not fixed by the provider itself.
            api_key (str | None): Credential for the provider, when required.
            timeout (float): Request timeout in seconds.

        """

    @abstractmethod
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

    @abstractmethod
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
            str: Successive content chunks as the provider streams them.

        """
