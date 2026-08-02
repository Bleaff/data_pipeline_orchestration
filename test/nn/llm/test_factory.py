"""Tests for the LLMBackendType -> backend class factory (#25)."""

from __future__ import annotations

import pytest

from neudc.nn.backends.llm import LLMBackendType, get_llm_backend
from neudc.nn.backends.llm.local import LocalLLMBackend
from neudc.nn.backends.llm.openrouter import OpenRouterBackend


def test_get_llm_backend_resolves_openrouter():
    assert get_llm_backend(LLMBackendType.OPENROUTER) is OpenRouterBackend


def test_get_llm_backend_resolves_local():
    assert get_llm_backend(LLMBackendType.LOCAL) is LocalLLMBackend


def test_get_llm_backend_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported LLM backend type"):
        get_llm_backend("NotARealBackend")
