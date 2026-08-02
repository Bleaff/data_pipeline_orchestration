"""Tests for the frame-preview route and its `find_latest_frame` helper (#22)."""

from __future__ import annotations

import time

from neudc.service.api.preview import find_latest_frame


def _save_image_node(save_dir) -> dict:
    return {"id": "saver", "type": "SaveImageNode", "save_dir": str(save_dir), "outputs": []}


def test_find_latest_frame_returns_none_without_a_save_image_node(tmp_path) -> None:
    nodes_config = [{"id": "reader", "type": "FolderImageNode", "folder_path": str(tmp_path), "outputs": []}]

    assert find_latest_frame(nodes_config) is None


def test_find_latest_frame_returns_none_when_no_frames_written_yet(tmp_path) -> None:
    assert find_latest_frame([_save_image_node(tmp_path)]) is None


def test_find_latest_frame_picks_the_most_recently_written_file(tmp_path) -> None:
    older = tmp_path / "frame_1.jpg"
    newer = tmp_path / "frame_2.jpg"
    older.write_bytes(b"old")
    time.sleep(0.01)
    newer.write_bytes(b"new")

    assert find_latest_frame([_save_image_node(tmp_path)]) == newer


def test_preview_of_unknown_pipeline_is_404(client) -> None:
    resp = client.get("/pipelines/ghost/preview")

    assert resp.status_code == 404


def test_preview_before_any_frame_is_written_is_404(client, tmp_path) -> None:
    nodes = [_save_image_node(tmp_path)]
    client.post("/pipelines/p1/start", json={"nodes": nodes})

    resp = client.get("/pipelines/p1/preview")

    assert resp.status_code == 404
    assert "no SaveImageNode output" in resp.json()["detail"]


def test_preview_serves_the_latest_written_frame(client, tmp_path) -> None:
    nodes = [_save_image_node(tmp_path)]
    client.post("/pipelines/p1/start", json={"nodes": nodes})
    (tmp_path / "frame_1.jpg").write_bytes(b"fake-jpeg-bytes")

    resp = client.get("/pipelines/p1/preview")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == b"fake-jpeg-bytes"
    # Lets a polling client tell two preview responses apart without diffing bytes.
    assert resp.headers["x-frame-name"] == "frame_1.jpg"
