"""Tests for the GET /node-types route (#24)."""

from __future__ import annotations


def test_list_node_types_returns_the_full_catalog(client) -> None:
    resp = client.get("/node-types")

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) > 0
    types = {entry["type"] for entry in body}
    assert "FolderImageNode" in types
    assert "AudioReaderNode" not in types


def test_node_type_entries_have_the_expected_shape(client) -> None:
    resp = client.get("/node-types")

    folder_image = next(e for e in resp.json() if e["type"] == "FolderImageNode")
    assert folder_image["runs_as"] == "thread"
    assert "Frame" in folder_image["accepts"]
    assert "Frame" in folder_image["emits"]
    field_names = {f["name"] for f in folder_image["fields"]}
    assert {"folder_path", "mode", "frame_delay"} <= field_names
