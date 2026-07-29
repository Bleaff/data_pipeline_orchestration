"""End-to-end pipeline test over threaded nodes (Stage 2).

Builds a real reader -> resize -> saver pipeline with RoutingFactory + NodeFactory
(the same wiring main.py uses) and asserts images flow through and are saved at the
resized resolution. This exercises routing, mailboxes, threaded-node lifecycle and
the ResizeNode fix — without needing a GPU or model.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from neudc.core.communication.messaging.routing_factory import RoutingFactory
from neudc.core.node.node_factory import NodeFactory

if TYPE_CHECKING:
    from pathlib import Path


def _make_images(folder: Path, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = np.full((16, 16, 3), (i * 30) % 255, dtype=np.uint8)
        cv2.imwrite(str(folder / f"img_{i}.png"), img)


def _build(config: dict) -> list:
    router = RoutingFactory(config)
    mailboxes = router.create_mailboxes()
    nodes = []
    for nc in config["nodes"]:
        replica_mailboxes = mailboxes[nc["id"]]
        for i, mailbox in enumerate(replica_mailboxes):
            node = NodeFactory.create(nc, mailbox=mailbox)
            if len(replica_mailboxes) > 1:
                node.id = f"{nc['id']}#{i}"
            nodes.append(node)
    return nodes


def test_reader_resize_saver_pipeline(tmp_path: Path) -> None:
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _make_images(in_dir, count=3)

    config = {
        "nodes": [
            {
                "id": "reader",
                "type": "FolderImageNode",
                "folder_path": str(in_dir),
                "mode": "only_one",
                "frame_delay": 0.0,
                "outputs": ["resize"],
            },
            {
                "id": "resize",
                "type": "ResizeNode",
                "target_width": 8,
                "target_height": 8,
                "outputs": ["saver"],
            },
            {
                "id": "saver",
                "type": "SaveImageNode",
                "save_dir": str(out_dir),
                "outputs": [],
            },
        ],
    }

    nodes = _build(config)
    # Start consumers before producers so nothing is missed.
    for node in reversed(nodes):
        node.start()

    deadline = time.time() + 15
    while time.time() < deadline and len(list(out_dir.glob("*.jpg"))) < 3:
        time.sleep(0.1)

    for node in nodes:
        node.stop()

    saved = sorted(out_dir.glob("*.jpg"))
    assert len(saved) == 3, f"expected 3 saved frames, got {len(saved)}"

    # The resize node must have taken effect end-to-end.
    img = cv2.imread(str(saved[0]))
    assert img.shape[:2] == (8, 8)
