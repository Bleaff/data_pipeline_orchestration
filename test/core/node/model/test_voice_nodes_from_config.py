"""Tests for AsrNode/LlmNode/TtsNode `from_config`: config dict -> wired backend instance."""

from __future__ import annotations

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.node.model.asr_node import AsrNode
from neudc.core.node.model.llm_node import LlmNode
from neudc.core.node.model.tts_node import TtsNode
from neudc.nn.backends.asr.openai_compatible import OpenAICompatibleASRBackend
from neudc.nn.backends.llm.local import LocalLLMBackend
from neudc.nn.backends.tts.openai_compatible import OpenAICompatibleTTSBackend


def test_asr_node_from_config_builds_backend_and_wires_options() -> None:
    node = AsrNode.from_config(
        {
            "mailbox": ZMQMailbox(),
            "asr_config": {"model_id": "whisper-large-v3", "base_url": "http://asr.test/v1"},
            "session_id": "sess-1",
            "min_speech_duration": 0.5,
        },
    )

    assert isinstance(node.backend, OpenAICompatibleASRBackend)
    assert node.backend.model_id == "whisper-large-v3"
    assert node.backend.base_url == "http://asr.test/v1"
    assert node.session_id == "sess-1"
    assert node.min_speech_duration == 0.5


def test_llm_node_from_config_builds_local_backend_and_seeds_system_prompt() -> None:
    node = LlmNode.from_config(
        {
            "mailbox": ZMQMailbox(),
            "llm_config": {
                "provider": "LocalLLMBackend",
                "model_id": "gemma-2-9b-it",
                "base_url": "http://llm.test/v1",
                "stream": True,
                "temperature": 0.3,
            },
            "system_prompt": "be brief",
        },
    )

    assert isinstance(node.backend, LocalLLMBackend)
    assert node.backend.model_id == "gemma-2-9b-it"
    assert node.stream is True
    assert node.temperature == 0.3
    assert node._history[0].role == "system"
    assert node._history[0].content == "be brief"


def test_tts_node_from_config_builds_backend_with_voice() -> None:
    node = TtsNode.from_config(
        {
            "mailbox": ZMQMailbox(),
            "tts_config": {"model_id": "tts-1", "voice": "alloy", "base_url": "http://tts.test/v1"},
        },
    )

    assert isinstance(node.backend, OpenAICompatibleTTSBackend)
    assert node.backend.model_id == "tts-1"
    assert node.backend.voice == "alloy"
    assert node.backend.base_url == "http://tts.test/v1"
