"""Tests for NodeFactory's payload-contract lookup (#35): get_accepts / get_emits."""

from __future__ import annotations

import sys

import pytest

from neudc.core.communication.messaging.types import Frame
from neudc.core.node.node_factory import NodeFactory
from neudc.utils.extras import MissingExtraError

# Every module that (transitively) runs the `cv` guard in base_process_inference.
_INFERENCE_MODULES = tuple(
    f"neudc.core.node.model.{name}"
    for name in (
        "base_process_inference",
        "base_batch_process_inference",
        "proc_det_inference",
        "proc_blur_inference",
        "proc_det_batch_inference",
        "proc_embedding_inference",
    )
)


@pytest.mark.cv
def test_get_accepts_defaults_to_frame() -> None:
    assert NodeFactory.get_accepts("FolderImageNode") == (Frame,)


@pytest.mark.cv
def test_get_emits_defaults_to_frame() -> None:
    assert NodeFactory.get_emits("SaveImageNode") == (Frame,)


@pytest.mark.cv
def test_get_accepts_reflects_a_node_class_override(monkeypatch) -> None:
    from neudc.core.communication.messaging.types import TextChunk

    node_cls = NodeFactory._resolve("HashNode")
    monkeypatch.setattr(node_cls, "accepts", (TextChunk,))

    assert NodeFactory.get_accepts("HashNode") == (TextChunk,)


def test_audio_reader_emits_audio_chunk() -> None:
    from neudc.core.communication.messaging.types import AudioChunk

    assert NodeFactory.get_emits("AudioReaderNode") == (AudioChunk,)


def test_vad_node_accepts_and_emits_audio_chunk() -> None:
    from neudc.core.communication.messaging.types import AudioChunk

    assert NodeFactory.get_accepts("VadNode") == (AudioChunk,)
    assert NodeFactory.get_emits("VadNode") == (AudioChunk,)


def test_resolve_names_the_node_type_and_extra_when_an_extra_is_missing(without_cv2) -> None:
    without_cv2("neudc.core.node.readers.image_reader")

    with pytest.raises(MissingExtraError, match=r"Node type 'FolderImageNode' needs the `cv` extra"):
        NodeFactory.get_emits("FolderImageNode")


@pytest.mark.parametrize("node_type", ["ProcessDetInference", "ProcessBlurInference", "ProcessDetBatchInference"])
def test_model_inference_nodes_need_the_cv_extra(without_cv2, node_type: str) -> None:
    # Their modules don't import cv2 themselves (models load lazily in the child process),
    # so without the guard they'd resolve here and only crash at `ModelFactory.create`.
    without_cv2(*_INFERENCE_MODULES)

    with pytest.raises(MissingExtraError, match=rf"Node type '{node_type}' needs the `cv` extra"):
        NodeFactory.get_emits(node_type)


@pytest.mark.cv
@pytest.mark.parametrize("node_type", ["ProcessDetInference", "ProcessBlurInference", "ProcessDetBatchInference"])
def test_model_inference_nodes_need_torch_even_with_cv2_installed(monkeypatch, node_type: str) -> None:
    # A partial install (cv2 pulled in by another package, no torch) must still be caught
    # here, not later in the child process at `get_backend`/model import.
    monkeypatch.setitem(sys.modules, "torch", None)
    for module in _INFERENCE_MODULES:
        monkeypatch.delitem(sys.modules, module, raising=False)

    with pytest.raises(MissingExtraError, match=rf"Node type '{node_type}' needs the `cv` extra.*'torch'"):
        NodeFactory.get_emits(node_type)


def test_resolve_does_not_blame_an_extra_for_an_unrelated_import_failure(monkeypatch) -> None:
    # `None` for the node's own module: a ModuleNotFoundError that has nothing to do with an extra.
    monkeypatch.setitem(sys.modules, "neudc.core.node.readers.image_reader", None)

    with pytest.raises(ModuleNotFoundError) as exc:
        NodeFactory.get_emits("FolderImageNode")

    assert not isinstance(exc.value, MissingExtraError)
    assert exc.value.name == "neudc.core.node.readers.image_reader"


def test_resolve_unknown_type_is_still_a_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown node type"):
        NodeFactory.get_emits("NopeNode")
