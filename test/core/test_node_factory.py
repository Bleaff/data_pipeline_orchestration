"""Tests for NodeFactory's payload-contract lookup (#35): get_accepts / get_emits."""

from __future__ import annotations

from neudc.core.communication.messaging.types import Frame
from neudc.core.node.node_factory import NodeFactory


def test_get_accepts_defaults_to_frame() -> None:
    assert NodeFactory.get_accepts("FolderImageNode") == (Frame,)


def test_get_emits_defaults_to_frame() -> None:
    assert NodeFactory.get_emits("SaveImageNode") == (Frame,)


def test_get_accepts_reflects_a_node_class_override(monkeypatch) -> None:
    from neudc.core.communication.messaging.types import TextChunk

    node_cls = NodeFactory._resolve("HashNode")
    monkeypatch.setattr(node_cls, "accepts", (TextChunk,))

    assert NodeFactory.get_accepts("HashNode") == (TextChunk,)
