"""Multipipe manager test (Stage 2).

Runs a threaded reader -> saver pipeline inside a BasePipeline child process via
PipelineServiceManager. This validates the previously-broken multipipe path (the
NodeFactory.create call no longer passes a `logger` kwarg) across a real process
boundary, plus status()/stop_pipeline lifecycle.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from neudc.service.pipeline_manager import PipelineServiceManager

if TYPE_CHECKING:
    from pathlib import Path


def _make_images(folder: Path, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = np.full((16, 16, 3), (i * 30) % 255, dtype=np.uint8)
        cv2.imwrite(str(folder / f"img_{i}.png"), img)


def test_manager_runs_threaded_pipeline(tmp_path: Path) -> None:
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _make_images(in_dir, count=3)

    nodes_config = [
        {
            "id": "reader",
            "type": "FolderImageNode",
            "folder_path": str(in_dir),
            "mode": "loop",
            "frame_delay": 0.05,
            "outputs": ["saver"],
        },
        {
            "id": "saver",
            "type": "SaveImageNode",
            "save_dir": str(out_dir),
            "outputs": [],
        },
    ]

    manager = PipelineServiceManager()
    manager.start_pipeline(name="t", nodes_config=nodes_config)
    try:
        deadline = time.time() + 20
        while time.time() < deadline and not list(out_dir.glob("*.jpg")):
            time.sleep(0.2)

        assert list(out_dir.glob("*.jpg")), "child pipeline produced no output"
        assert manager.status().get("t") == "running"
    finally:
        manager.stop_pipeline("t")

    assert "t" not in manager.pipelines
