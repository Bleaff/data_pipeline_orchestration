"""Pluggable LLM/VLM provider backends (OpenRouter, local OpenAI-compatible) and their factory (#25).

Base for the future VLM node (#17) and copilot (#26): a single `BaseLLMBackend`
interface, config-driven `model_id`, and provider selection through
`LLMBackendConfig.provider` instead of importing a specific adapter directly.
"""

from __future__ import annotations

from neudc.nn.backends.llm.base import BaseLLMBackend, ChatMessage, LLMBackendType
from neudc.nn.backends.llm.config import LLMBackendConfig, build_llm_backend
from neudc.nn.backends.llm.factory import get_llm_backend

__all__ = (
    "BaseLLMBackend",
    "ChatMessage",
    "LLMBackendConfig",
    "LLMBackendType",
    "build_llm_backend",
    "get_llm_backend",
)
