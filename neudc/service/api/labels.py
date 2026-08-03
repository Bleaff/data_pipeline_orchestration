"""Pre-label review: reads/writes `CreateDataset`'s YOLO-format output for human review (#24).

Layout written by `CreateDatasetMixin`: `{save_dir}/{folder}/images/frame_{id}.jpg` and
`{save_dir}/{folder}/labels/frame_{id}.txt` (one `class_id x_center y_center width
height` line per box, all normalized 0-1). `folder` is the name of whatever directory
the source frames came from, so one dataset can span several folders.

Review state (has a human looked at this frame yet) has nowhere else to live — there's
no database anywhere in this codebase, everything is files — so it's a small JSON
manifest (`_REVIEW_MANIFEST_NAME`) written alongside the dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neudc.service.api.schemas import BoxSpec, LabelFrameDetail, LabelFrameSummary

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = (
    "find_dataset_root",
    "get_frame_detail",
    "list_label_frames",
    "save_frame_boxes",
)

_FRAME_ID_SEP = ":"
_REVIEW_MANIFEST_NAME = ".review_state.json"


def find_dataset_root(nodes_config: list[dict[str, Any]]) -> Path | None:
    """Return the `save_dir` of this pipeline's `CreateDataset` node, if it has one."""
    for node in nodes_config:
        if node.get("type") == "CreateDataset" and "save_dir" in node:
            return Path(node["save_dir"])
    return None


def _frame_id_for(folder: str, stem: str) -> str:
    return f"{folder}{_FRAME_ID_SEP}{stem}"


def _iter_labeled_frames(dataset_root: Path) -> Iterator[tuple[str, str, Path, Path]]:
    """Yield `(folder, stem, image_path, label_path)` for every image with a label file."""
    if not dataset_root.is_dir():
        return
    for images_dir in sorted(dataset_root.glob("*/images")):
        folder = images_dir.parent.name
        labels_dir = images_dir.parent / "labels"
        for image_path in sorted(images_dir.glob("*.jpg")):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                yield folder, image_path.stem, image_path, label_path


def _resolve_frame_paths(dataset_root: Path, frame_id: str) -> tuple[Path, Path] | None:
    """Resolve `frame_id` to `(image_path, label_path)`, or `None` if invalid/missing.

    `frame_id` is attacker-controlled (a URL path parameter), so this rejects anything
    that could escape `dataset_root` (path separators, `..`) before touching the
    filesystem, then re-checks the resolved path is still inside `dataset_root` as a
    second line of defense.
    """
    folder, sep, stem = frame_id.partition(_FRAME_ID_SEP)
    if not sep or not folder or not stem:
        return None
    if any(part in ("..", "", ".") or "/" in part or "\\" in part for part in (folder, stem)):
        return None

    dataset_root_resolved = dataset_root.resolve()
    image_path = (dataset_root / folder / "images" / f"{stem}.jpg").resolve()
    label_path = (dataset_root / folder / "labels" / f"{stem}.txt").resolve()
    if dataset_root_resolved not in image_path.parents or dataset_root_resolved not in label_path.parents:
        return None
    if not image_path.is_file() or not label_path.is_file():
        return None
    return image_path, label_path


def _read_boxes(label_path: Path) -> list[BoxSpec]:
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        class_id, x, y, w, h = parts
        boxes.append(
            BoxSpec(class_id=int(class_id), x_center=float(x), y_center=float(y), width=float(w), height=float(h))
        )
    return boxes


def _write_boxes(label_path: Path, boxes: list[BoxSpec]) -> None:
    lines = [f"{b.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}" for b in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _manifest_path(dataset_root: Path) -> Path:
    return dataset_root / _REVIEW_MANIFEST_NAME


def _load_reviewed(dataset_root: Path) -> set[str]:
    path = _manifest_path(dataset_root)
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return set()
    return set(data.get("reviewed", []))


def _mark_reviewed(dataset_root: Path, frame_id: str) -> None:
    reviewed = _load_reviewed(dataset_root)
    reviewed.add(frame_id)
    _manifest_path(dataset_root).write_text(json.dumps({"reviewed": sorted(reviewed)}))


def list_label_frames(dataset_root: Path) -> list[LabelFrameSummary]:
    """List every frame with pre-labels under `dataset_root`, across all its source folders."""
    reviewed = _load_reviewed(dataset_root)
    summaries = []
    for folder, stem, _image_path, label_path in _iter_labeled_frames(dataset_root):
        frame_id = _frame_id_for(folder, stem)
        summaries.append(
            LabelFrameSummary(frame_id=frame_id, box_count=len(_read_boxes(label_path)), reviewed=frame_id in reviewed)
        )
    return summaries


def get_frame_detail(dataset_root: Path, frame_id: str) -> LabelFrameDetail | None:
    """Return one frame's boxes, or `None` if `frame_id` doesn't resolve to a real labeled frame."""
    paths = _resolve_frame_paths(dataset_root, frame_id)
    if paths is None:
        return None
    _image_path, label_path = paths
    return LabelFrameDetail(
        frame_id=frame_id, boxes=_read_boxes(label_path), reviewed=frame_id in _load_reviewed(dataset_root)
    )


def get_frame_image_path(dataset_root: Path, frame_id: str) -> Path | None:
    """Return the frame's image path, or `None` if `frame_id` doesn't resolve to a real labeled frame."""
    paths = _resolve_frame_paths(dataset_root, frame_id)
    return paths[0] if paths else None


def save_frame_boxes(dataset_root: Path, frame_id: str, boxes: list[BoxSpec]) -> LabelFrameDetail | None:
    """Overwrite a frame's label file with `boxes` and mark it reviewed.

    Returns `None` if `frame_id` doesn't resolve to a real labeled frame (nothing is
    written in that case).
    """
    paths = _resolve_frame_paths(dataset_root, frame_id)
    if paths is None:
        return None
    _image_path, label_path = paths
    _write_boxes(label_path, boxes)
    _mark_reviewed(dataset_root, frame_id)
    return LabelFrameDetail(frame_id=frame_id, boxes=boxes, reviewed=True)
