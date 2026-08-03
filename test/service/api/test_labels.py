"""Tests for the CreateDataset label read/write helpers (#24)."""

from __future__ import annotations

from neudc.service.api.labels import (
    find_dataset_root,
    get_frame_detail,
    get_frame_image_path,
    list_label_frames,
    save_frame_boxes,
)
from neudc.service.api.schemas import BoxSpec


def _write_frame(root, folder: str, stem: str, boxes_text: str) -> None:
    images_dir = root / folder / "images"
    labels_dir = root / folder / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / f"{stem}.jpg").write_bytes(b"fake-jpeg-bytes")
    (labels_dir / f"{stem}.txt").write_text(boxes_text)


def test_find_dataset_root_locates_the_create_dataset_node(tmp_path):
    nodes = [
        {"id": "reader", "type": "FolderImageNode", "outputs": ["ds"]},
        {"id": "ds", "type": "CreateDataset", "save_dir": str(tmp_path), "outputs": []},
    ]

    assert find_dataset_root(nodes) == tmp_path


def test_find_dataset_root_returns_none_without_a_create_dataset_node():
    nodes = [{"id": "reader", "type": "FolderImageNode", "outputs": []}]

    assert find_dataset_root(nodes) is None


def test_list_label_frames_reads_box_count_and_unreviewed_by_default(tmp_path):
    _write_frame(tmp_path, "in", "frame_0", "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")

    frames = list_label_frames(tmp_path)

    assert len(frames) == 1
    assert frames[0].frame_id == "in:frame_0"
    assert frames[0].box_count == 2
    assert frames[0].reviewed is False


def test_list_label_frames_skips_images_without_a_label_file(tmp_path):
    images_dir = tmp_path / "in" / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "frame_0.jpg").write_bytes(b"fake")

    assert list_label_frames(tmp_path) == []


def test_list_label_frames_spans_multiple_source_folders(tmp_path):
    _write_frame(tmp_path, "in", "frame_0", "")
    _write_frame(tmp_path, "other", "frame_0", "")

    frame_ids = {f.frame_id for f in list_label_frames(tmp_path)}

    assert frame_ids == {"in:frame_0", "other:frame_0"}


def test_get_frame_detail_parses_yolo_format_boxes(tmp_path):
    _write_frame(tmp_path, "in", "frame_0", "3 0.1 0.2 0.3 0.4\n")

    detail = get_frame_detail(tmp_path, "in:frame_0")

    assert detail is not None
    assert detail.boxes == [BoxSpec(class_id=3, x_center=0.1, y_center=0.2, width=0.3, height=0.4)]


def test_get_frame_detail_returns_none_for_unknown_frame(tmp_path):
    assert get_frame_detail(tmp_path, "in:missing") is None


def test_get_frame_detail_rejects_path_traversal_attempts(tmp_path):
    (tmp_path / "secret.txt").write_text("nope")

    assert get_frame_detail(tmp_path, "..:secret") is None
    assert get_frame_detail(tmp_path, "in:../../secret") is None
    assert get_frame_image_path(tmp_path, "in:../../../etc/passwd") is None


def test_save_frame_boxes_overwrites_the_label_file_and_marks_reviewed(tmp_path):
    _write_frame(tmp_path, "in", "frame_0", "0 0.5 0.5 0.2 0.2\n")
    new_boxes = [BoxSpec(class_id=7, x_center=0.1, y_center=0.1, width=0.05, height=0.05)]

    result = save_frame_boxes(tmp_path, "in:frame_0", new_boxes)

    assert result is not None
    assert result.boxes == new_boxes
    assert result.reviewed is True
    # Persisted, not just returned: a fresh read sees the same thing.
    assert get_frame_detail(tmp_path, "in:frame_0").boxes == new_boxes


def test_save_frame_boxes_reflects_reviewed_state_in_list_label_frames(tmp_path):
    _write_frame(tmp_path, "in", "frame_0", "0 0.5 0.5 0.2 0.2\n")

    save_frame_boxes(tmp_path, "in:frame_0", [])

    frames = list_label_frames(tmp_path)
    assert frames[0].reviewed is True
    assert frames[0].box_count == 0


def test_save_frame_boxes_returns_none_for_unknown_frame(tmp_path):
    assert save_frame_boxes(tmp_path, "in:missing", []) is None
