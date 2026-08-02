"""Factory resolving an `LLMBackendType` to its concrete backend class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neudc.nn.backends.llm.base import LLMBackendType

if TYPE_CHECKING:
    from neudc.nn.backends.llm.base import BaseLLMBackend

__all__ = ("get_llm_backend",)


def get_llm_backend(backend_type: LLMBackendType) -> type[BaseLLMBackend]:
    """Return the backend class for `backend_type`, importing it lazily.

    Args:
    ----
        backend_type (LLMBackendType): Which provider adapter to resolve.

    Returns:
    -------
        type[BaseLLMBackend]: The concrete backend class (not yet instantiated).

    """
    if backend_type == LLMBackendType.OPENROUTER:
        from neudc.nn.backends.llm.openrouter import OpenRouterBackend

        return OpenRouterBackend
    if backend_type == LLMBackendType.LOCAL:
        from neudc.nn.backends.llm.local import LocalLLMBackend

        return LocalLLMBackend
    msg = f"Unsupported LLM backend type: {backend_type}"
    raise ValueError(msg)
