"""Local LLM backend: Ollama / vLLM / llama.cpp behind their OpenAI-compatible endpoint."""

from __future__ import annotations

from neudc.nn.backends.llm.openai_compatible import OpenAICompatibleBackend

__all__ = ("LocalLLMBackend",)


class LocalLLMBackend(OpenAICompatibleBackend):
    """Chat completions through a self-hosted OpenAI-compatible server (Ollama, vLLM, llama.cpp).

    Unlike OpenRouter, there is no fixed endpoint: `base_url` must name the caller's
    own server. Most local runtimes accept any (or no) `Authorization` header, so
    `api_key` stays optional here.
    """

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the LocalLLMBackend class.

        Args:
        ----
            model_id (str): Model name as served by the local runtime (e.g. `"llama3.1"`).
            base_url (str): URL of the local OpenAI-compatible server (e.g. `"http://localhost:11434/v1"`).
            api_key (str | None): Optional bearer token, only if the local server enforces one.
            timeout (float): Request timeout in seconds.

        """
        if not base_url:
            msg = (
                "LocalLLMBackend requires a base_url pointing at your Ollama/vLLM/llama.cpp OpenAI-compatible endpoint"
            )
            raise ValueError(msg)
        super().__init__(model_id, base_url=base_url, api_key=api_key, timeout=timeout)
