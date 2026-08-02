"""Tests for LLMBackendConfig validation and build_llm_backend wiring (#25)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neudc.nn.backends.llm import LLMBackendConfig, LLMBackendType, build_llm_backend
from neudc.nn.backends.llm.local import LocalLLMBackend
from neudc.nn.backends.llm.openrouter import OpenRouterBackend


def test_local_provider_requires_base_url():
    with pytest.raises(ValidationError, match="base_url is required"):
        LLMBackendConfig(provider=LLMBackendType.LOCAL, model_id="llama3.1")


def test_openrouter_provider_does_not_require_base_url():
    config = LLMBackendConfig(provider=LLMBackendType.OPENROUTER, model_id="openai/gpt-4o")

    assert config.base_url is None


def test_temperature_out_of_range_rejected():
    with pytest.raises(ValidationError):
        LLMBackendConfig(provider=LLMBackendType.OPENROUTER, model_id="openai/gpt-4o", temperature=3.0)


def test_build_llm_backend_local():
    config = LLMBackendConfig(
        provider=LLMBackendType.LOCAL,
        model_id="llama3.1",
        base_url="http://localhost:11434/v1",
    )

    backend = build_llm_backend(config)

    assert isinstance(backend, LocalLLMBackend)
    assert backend.model_id == "llama3.1"
    assert backend.base_url == "http://localhost:11434/v1"


def test_build_llm_backend_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    config = LLMBackendConfig(provider=LLMBackendType.OPENROUTER, model_id="openai/gpt-4o")

    backend = build_llm_backend(config)

    assert isinstance(backend, OpenRouterBackend)
    assert backend.model_id == "openai/gpt-4o"
