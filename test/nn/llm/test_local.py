"""Tests for LocalLLMBackend (Ollama/vLLM/llama.cpp OpenAI-compatible endpoint) (#25)."""

from __future__ import annotations

import pytest

from neudc.nn.backends.llm.local import LocalLLMBackend


def test_requires_base_url():
    with pytest.raises(ValueError, match="requires a base_url"):
        LocalLLMBackend("llama3.1")


def test_accepts_base_url_without_api_key():
    backend = LocalLLMBackend("llama3.1", base_url="http://localhost:11434/v1")

    assert backend.base_url == "http://localhost:11434/v1"
    assert backend.api_key is None


def test_optional_api_key_is_kept_when_given():
    backend = LocalLLMBackend("llama3.1", base_url="http://localhost:8000/v1", api_key="local-secret")

    assert backend.api_key == "local-secret"
