"""Tests for TextNormalizeNode: from_config wiring and its declared payload contract (#36)."""

from __future__ import annotations

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging.types import TextChunk
from neudc.core.node.node_factory import NodeFactory
from neudc.core.node.processors.text_normalize_node import TextNormalizeNode


def test_from_config_builds_a_working_node() -> None:
    mailbox = ZMQMailbox()
    node = TextNormalizeNode.from_config({"mailbox": mailbox})

    assert isinstance(node, TextNormalizeNode)
    chunk = TextChunk(timestamp=0.0, text="  Loud Noises  ")
    assert node.process(chunk).text == "loud noises"


def test_node_declares_a_real_textchunk_contract() -> None:
    # No monkeypatch: TextNormalizeNode genuinely overrides the (Frame,) default (#35).
    assert NodeFactory.get_accepts("TextNormalizeNode") == (TextChunk,)
    assert NodeFactory.get_emits("TextNormalizeNode") == (TextChunk,)
