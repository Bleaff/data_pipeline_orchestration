"""End-to-end test: real `PipelineServiceManager` + real subprocess pipeline through the API (#22).

Everything else in `test/service/api/` exercises the routes against `FakePipelineServiceManager`
for speed; this one test proves the full stack actually wires together, mirroring
`test/service/test_pipeline_manager.py`'s real-process pattern.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
from fastapi.testclient import TestClient

from neudc.service.api import create_app
from neudc.service.pipeline_manager import PipelineServiceManager

if TYPE_CHECKING:
    from pathlib import Path


def _make_images(folder: Path, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = np.full((16, 16, 3), (i * 30) % 255, dtype=np.uint8)
        cv2.imwrite(str(folder / f"img_{i}.png"), img)


def test_start_status_preview_stop_through_the_real_manager(tmp_path: Path) -> None:
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _make_images(in_dir, count=3)

    nodes = [
        {
            "id": "reader",
            "type": "FolderImageNode",
            "folder_path": str(in_dir),
            "mode": "loop",
            "frame_delay": 0.05,
            "outputs": ["saver"],
        },
        {"id": "saver", "type": "SaveImageNode", "save_dir": str(out_dir), "outputs": []},
    ]

    client = TestClient(create_app(manager=PipelineServiceManager()))
    try:
        start_resp = client.post("/pipelines/t/start", json={"nodes": nodes})
        assert start_resp.status_code == 201
        assert start_resp.json() == {"name": "t", "status": "running"}

        deadline = time.time() + 20
        while time.time() < deadline and not list(out_dir.glob("*.jpg")):
            time.sleep(0.2)
        assert list(out_dir.glob("*.jpg")), "child pipeline produced no output"

        status_resp = client.get("/pipelines/t/status")
        assert status_resp.json() == {"name": "t", "status": "running"}

        preview_resp = client.get("/pipelines/t/preview")
        assert preview_resp.status_code == 200
        assert preview_resp.headers["content-type"] == "image/jpeg"
    finally:
        stop_resp = client.post("/pipelines/t/stop")
        assert stop_resp.status_code == 204

    assert client.get("/pipelines/t/status").status_code == 404
