"""Tests for pipeline config validation (Stage 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neudc.core.utils.config_loader import load_config
from neudc.core.utils.config_schema import ConfigError, validate_pipeline_config

_VALID = {
    "nodes": [
        {"id": "reader", "type": "FolderImageNode", "folder_path": "x", "outputs": ["saver"]},
        {"id": "saver", "type": "SaveImageNode", "save_dir": "y", "outputs": []},
    ],
}


def test_valid_config_passes() -> None:
    parsed = validate_pipeline_config(_VALID)
    assert [n.id for n in parsed.nodes] == ["reader", "saver"]


def test_missing_nodes_key() -> None:
    with pytest.raises(ConfigError, match="nodes"):
        validate_pipeline_config({"prometheus": {"enable": False}})


def test_non_mapping_config() -> None:
    with pytest.raises(ConfigError, match="mapping"):
        validate_pipeline_config([1, 2, 3])


def test_unknown_node_type() -> None:
    cfg = {"nodes": [{"id": "a", "type": "NopeNode", "outputs": []}]}
    with pytest.raises(ConfigError, match="unknown type"):
        validate_pipeline_config(cfg)


def test_dangling_output_reference() -> None:
    cfg = {"nodes": [{"id": "reader", "type": "FolderImageNode", "outputs": ["ghost"]}]}
    with pytest.raises(ConfigError, match="does not reference"):
        validate_pipeline_config(cfg)


def test_duplicate_ids() -> None:
    cfg = {
        "nodes": [
            {"id": "dup", "type": "FolderImageNode", "outputs": []},
            {"id": "dup", "type": "SaveImageNode", "outputs": []},
        ],
    }
    with pytest.raises(ConfigError, match="Duplicate node ids"):
        validate_pipeline_config(cfg)


def test_valid_error_policy_passes() -> None:
    cfg = {
        "nodes": [
            {
                "id": "reader",
                "type": "FolderImageNode",
                "outputs": [],
                "error_policy": {"on_error": "retry", "max_retries": 3, "dead_letter_dir": "./dead"},
            },
        ],
    }
    assert validate_pipeline_config(cfg).nodes[0].id == "reader"


def test_invalid_error_policy_names_the_node() -> None:
    # Fails at config load, not deep in a running pipeline, and says which node.
    cfg = {
        "nodes": [
            {"id": "reader", "type": "FolderImageNode", "outputs": []},
            {"id": "detector", "type": "ProcessDetInference", "outputs": [], "error_policy": {"on_error": "sometimes"}},
        ],
    }
    with pytest.raises(ConfigError, match="Node 'detector': invalid 'error_policy'"):
        validate_pipeline_config(cfg)


def test_error_policy_retry_without_retries_is_rejected() -> None:
    cfg = {"nodes": [{"id": "a", "type": "FolderImageNode", "outputs": [], "error_policy": {"on_error": "retry"}}]}
    with pytest.raises(ConfigError, match="max_retries"):
        validate_pipeline_config(cfg)


def test_error_policy_must_be_a_mapping() -> None:
    cfg = {"nodes": [{"id": "a", "type": "FolderImageNode", "outputs": [], "error_policy": "retry"}]}
    with pytest.raises(ConfigError, match="must be a mapping"):
        validate_pipeline_config(cfg)


def test_example_cpu_pipeline_config_is_valid() -> None:
    config_path = Path("assets/configs/example_cpu_pipeline.yaml")
    parsed = validate_pipeline_config(load_config(config_path))
    assert {n.id for n in parsed.nodes} == {"reader", "resize", "saver"}


def test_default_frame_to_frame_edges_pass_payload_compat() -> None:
    # Every registered node type defaults to accepts=emits=(Frame,) (#35), so an
    # untouched config must keep validating exactly as before.
    assert validate_pipeline_config(_VALID).nodes[0].id == "reader"


def test_incompatible_payload_types_are_rejected(monkeypatch) -> None:
    from neudc.core.communication.messaging.types import TextChunk
    from neudc.core.node.node_factory import NodeFactory

    reader_cls = NodeFactory._resolve("FolderImageNode")
    monkeypatch.setattr(reader_cls, "emits", (TextChunk,))

    cfg = {
        "nodes": [
            {"id": "reader", "type": "FolderImageNode", "outputs": ["saver"]},
            {"id": "saver", "type": "SaveImageNode", "outputs": []},
        ],
    }
    with pytest.raises(ConfigError, match=r"Node 'reader' emits .* but 'saver' only accepts"):
        validate_pipeline_config(cfg)
