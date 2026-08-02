"""Tests for the BaseLLMBackend interface and ChatMessage wire format (#25)."""

from __future__ import annotations

import pytest

from neudc.nn.backends.llm import BaseLLMBackend, ChatMessage


def test_chat_message_to_dict_matches_openai_wire_format():
    msg = ChatMessage(role="user", content="hi")

    assert msg.to_dict() == {"role": "user", "content": "hi"}


def test_chat_message_rejects_unknown_role():
    with pytest.raises(ValueError, match="role"):
        ChatMessage(role="tool", content="hi")


def test_base_llm_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseLLMBackend("some-model")
