"""Tests for the config validate/read routes (#22)."""

from __future__ import annotations

import yaml


def test_validate_config_accepts_a_good_pipeline(client) -> None:
    nodes = [{"id": "reader", "type": "FolderImageNode", "folder_path": "unused", "outputs": []}]

    resp = client.post("/config/validate", json={"nodes": nodes})

    assert resp.status_code == 200
    assert resp.json() == {"valid": True, "error": None}


def test_validate_config_reports_the_offending_node(client) -> None:
    nodes = [{"id": "reader", "type": "NotARealNodeType", "outputs": []}]

    resp = client.post("/config/validate", json={"nodes": nodes})

    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert "reader" in body["error"]


def test_read_config_returns_parsed_yaml(client, tmp_path) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.dump({"nodes": [{"id": "reader", "type": "FolderImageNode"}]}))

    resp = client.get("/config", params={"path": str(config_path)})

    assert resp.status_code == 200
    assert resp.json() == {"nodes": [{"id": "reader", "type": "FolderImageNode"}]}


def test_read_config_missing_file_is_404(client, tmp_path) -> None:
    resp = client.get("/config", params={"path": str(tmp_path / "missing.yaml")})

    assert resp.status_code == 404
