"""OpenRouter LLM backend: HTTP access to hosted top-tier models."""

from __future__ import annotations

import os

from neudc.nn.backends.llm.openai_compatible import OpenAICompatibleBackend

__all__ = ("OpenRouterBackend",)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"


class OpenRouterBackend(OpenAICompatibleBackend):
    """Chat completions through OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the OpenRouterBackend class.

        Args:
        ----
            model_id (str): OpenRouter model identifier (e.g. `"openai/gpt-4o"`).
            base_url (str | None): Override for OpenRouter's API root; defaults to the public endpoint.
            api_key (str | None): OpenRouter API key. Falls back to the `OPENROUTER_API_KEY` env var.
            timeout (float): Request timeout in seconds.

        """
        api_key = api_key or os.environ.get(_API_KEY_ENV_VAR)
        if not api_key:
            msg = f"OpenRouterBackend requires an api_key (config `api_key` or {_API_KEY_ENV_VAR} env var)"
            raise ValueError(msg)
        super().__init__(model_id, base_url=base_url or _DEFAULT_BASE_URL, api_key=api_key, timeout=timeout)
