"""Tests for OpenRouterBackend credential/endpoint resolution (#25)."""

from __future__ import annotations

import pytest

from neudc.nn.backends.llm.openrouter import OpenRouterBackend


def test_defaults_to_public_base_url_and_env_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

    backend = OpenRouterBackend("openai/gpt-4o")

    assert backend.base_url == "https://openrouter.ai/api/v1"
    assert backend.api_key == "env-key"  # pragma: allowlist secret


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

    backend = OpenRouterBackend("openai/gpt-4o", api_key="explicit-key")  # pragma: allowlist secret

    assert backend.api_key == "explicit-key"  # pragma: allowlist secret


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="requires an api_key"):
        OpenRouterBackend("openai/gpt-4o")


def test_base_url_can_be_overridden(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

    backend = OpenRouterBackend("openai/gpt-4o", base_url="https://proxy.internal/v1")

    assert backend.base_url == "https://proxy.internal/v1"
