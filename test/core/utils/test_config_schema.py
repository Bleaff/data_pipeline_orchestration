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


def test_example_cpu_pipeline_config_is_valid() -> None:
    config_path = Path("assets/configs/example_cpu_pipeline.yaml")
    parsed = validate_pipeline_config(load_config(config_path))
    assert {n.id for n in parsed.nodes} == {"reader", "resize", "saver"}
