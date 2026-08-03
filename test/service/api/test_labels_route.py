"""Tests for the pre-label review routes (#24)."""

from __future__ import annotations

_VALID_NODES_TEMPLATE = [{"id": "reader", "type": "FolderImageNode", "folder_path": "unused", "outputs": ["ds"]}]


def _create_dataset_node(save_dir) -> dict:
    return {"id": "ds", "type": "CreateDataset", "save_dir": str(save_dir), "outputs": []}


def _write_frame(root, folder: str, stem: str, boxes_text: str) -> None:
    images_dir = root / folder / "images"
    labels_dir = root / folder / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / f"{stem}.jpg").write_bytes(b"fake-jpeg-bytes")
    (labels_dir / f"{stem}.txt").write_text(boxes_text)


def test_list_labels_of_unknown_pipeline_is_404(client) -> None:
    resp = client.get("/pipelines/ghost/labels")

    assert resp.status_code == 404


def test_list_labels_without_a_create_dataset_node_is_404(client) -> None:
    nodes = [{"id": "reader", "type": "FolderImageNode", "folder_path": "unused", "outputs": []}]
    client.post("/pipelines/p1/start", json={"nodes": nodes})

    resp = client.get("/pipelines/p1/labels")

    assert resp.status_code == 404
    assert "CreateDataset" in resp.json()["detail"]


def test_list_labels_returns_frames_from_the_dataset_dir(client, tmp_path) -> None:
    _write_frame(tmp_path, "in", "frame_0", "0 0.5 0.5 0.2 0.2\n")
    nodes = [*_VALID_NODES_TEMPLATE, _create_dataset_node(tmp_path)]
    client.post("/pipelines/p1/start", json={"nodes": nodes})

    resp = client.get("/pipelines/p1/labels")

    assert resp.status_code == 200
    assert resp.json() == [{"frame_id": "in:frame_0", "box_count": 1, "reviewed": False}]


def test_get_label_detail_returns_boxes(client, tmp_path) -> None:
    _write_frame(tmp_path, "in", "frame_0", "0 0.5 0.5 0.2 0.2\n")
    nodes = [*_VALID_NODES_TEMPLATE, _create_dataset_node(tmp_path)]
    client.post("/pipelines/p1/start", json={"nodes": nodes})

    resp = client.get("/pipelines/p1/labels/in:frame_0")

    assert resp.status_code == 200
    body = resp.json()
    assert body["boxes"] == [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}]


def test_get_label_detail_of_unknown_frame_is_404(client, tmp_path) -> None:
    nodes = [*_VALID_NODES_TEMPLATE, _create_dataset_node(tmp_path)]
    client.post("/pipelines/p1/start", json={"nodes": nodes})

    resp = client.get("/pipelines/p1/labels/in:missing")

    assert resp.status_code == 404


def test_get_label_image_serves_the_frame(client, tmp_path) -> None:
    _write_frame(tmp_path, "in", "frame_0", "")
    nodes = [*_VALID_NODES_TEMPLATE, _create_dataset_node(tmp_path)]
    client.post("/pipelines/p1/start", json={"nodes": nodes})

    resp = client.get("/pipelines/p1/labels/in:frame_0/image")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == b"fake-jpeg-bytes"


def test_update_label_persists_corrected_boxes_and_marks_reviewed(client, tmp_path) -> None:
    _write_frame(tmp_path, "in", "frame_0", "0 0.5 0.5 0.2 0.2\n")
    nodes = [*_VALID_NODES_TEMPLATE, _create_dataset_node(tmp_path)]
    client.post("/pipelines/p1/start", json={"nodes": nodes})

    resp = client.put(
        "/pipelines/p1/labels/in:frame_0",
        json={"boxes": [{"class_id": 5, "x_center": 0.1, "y_center": 0.1, "width": 0.05, "height": 0.05}]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["reviewed"] is True
    assert body["boxes"][0]["class_id"] == 5

    listing = client.get("/pipelines/p1/labels").json()
    assert listing[0]["reviewed"] is True


def test_update_label_of_unknown_frame_is_404(client, tmp_path) -> None:
    nodes = [*_VALID_NODES_TEMPLATE, _create_dataset_node(tmp_path)]
    client.post("/pipelines/p1/start", json={"nodes": nodes})

    resp = client.put("/pipelines/p1/labels/in:missing", json={"boxes": []})

    assert resp.status_code == 404
