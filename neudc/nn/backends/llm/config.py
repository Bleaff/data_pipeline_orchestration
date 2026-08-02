"""Config-driven construction of `BaseLLMBackend` instances (#25)."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from neudc.nn.backends.llm.base import BaseLLMBackend, LLMBackendType
from neudc.nn.backends.llm.factory import get_llm_backend

__all__ = ("LLMBackendConfig", "build_llm_backend")


class LLMBackendConfig(BaseModel):
    """Provider-agnostic configuration for an `LLMBackend` (`provider` picks the adapter)."""

    provider: LLMBackendType
    model_id: str
    base_url: str | None = None
    api_key: str | None = None
    stream: bool = False
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.7
    max_tokens: Annotated[int, Field(gt=0)] | None = None
    timeout: Annotated[float, Field(gt=0.0)] = 60.0

    # A model_validator (unlike field_validator) always runs, even when base_url is
    # left at its `None` default — which is exactly the case this constraint must catch.
    @model_validator(mode="after")
    def _check_base_url_for_local_provider(self) -> LLMBackendConfig:
        if self.provider == LLMBackendType.LOCAL and not self.base_url:
            msg = "base_url is required when provider is 'LocalLLMBackend'"
            raise ValueError(msg)
        return self


def build_llm_backend(config: LLMBackendConfig) -> BaseLLMBackend:
    """Instantiate the concrete backend named by `config.provider`.

    Args:
    ----
        config (LLMBackendConfig): Validated provider-agnostic configuration.

    Returns:
    -------
        BaseLLMBackend: The constructed provider backend.

    """
    backend_cls = get_llm_backend(config.provider)
    return backend_cls(
        config.model_id,
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
    )
