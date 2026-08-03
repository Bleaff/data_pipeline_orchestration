"""Tests for the node-type catalog powering the visual config builder (#24)."""

from __future__ import annotations

from neudc.core.node.node_factory import NodeFactory
from neudc.service.api.node_catalog import get_node_catalog


def test_every_catalog_entry_is_a_known_node_factory_type():
    known_types = set(NodeFactory.NODE_IMPORTS)
    catalog = get_node_catalog()

    for entry in catalog:
        assert entry.type in known_types


def test_audio_reader_node_is_excluded():
    # Its `device` field is a live Python object, not YAML/JSON-expressible (#24).
    catalog = get_node_catalog()

    assert "AudioReaderNode" not in {entry.type for entry in catalog}


def test_accepts_and_emits_are_populated_from_node_factory():
    catalog = get_node_catalog()
    folder_image = next(entry for entry in catalog if entry.type == "FolderImageNode")

    assert folder_image.accepts == [t.__name__ for t in NodeFactory.get_accepts("FolderImageNode")]
    assert folder_image.emits == [t.__name__ for t in NodeFactory.get_emits("FolderImageNode")]


def test_detection_nodes_expose_model_types_excluding_sahi():
    catalog = get_node_catalog()
    det_node = next(entry for entry in catalog if entry.type == "ProcessDetInference")

    assert det_node.model_types is not None
    model_type_names = {mt.type for mt in det_node.model_types}
    assert model_type_names == {"YOLOv8", "BlurClassification", "EmbeddingFilter"}


def test_nodes_without_a_model_config_have_no_model_types():
    catalog = get_node_catalog()
    save_image = next(entry for entry in catalog if entry.type == "SaveImageNode")

    assert save_image.model_types is None


def test_required_fields_have_no_default_value_semantics_conflict():
    # A field marked required shouldn't also claim a meaningful default (None is fine,
    # anything else would be a confusing spec: "required" but "pre-filled").
    catalog = get_node_catalog()

    for entry in catalog:
        for field in entry.fields:
            if field.required:
                assert field.default is None, f"{entry.type}.{field.name} is required but has a default"
