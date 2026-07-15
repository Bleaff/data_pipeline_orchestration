"""Tests for FolderImageNode robustness (Stage 2)."""

from __future__ import annotations

import time
from pathlib import Path

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.node.readers.image_reader import FolderImageNode


def test_empty_folder_stops_without_crash(tmp_path: Path) -> None:
    """An empty input folder must not raise ZeroDivisionError; the reader stops itself."""
    empty = tmp_path / "empty"
    empty.mkdir()

    mailbox = ZMQMailbox()
    reader = FolderImageNode(folder_path=str(empty), mailbox=mailbox, mode="loop", frame_delay=0.0)
    reader.start()

    deadline = time.time() + 5
    while reader.is_running and time.time() < deadline:
        time.sleep(0.05)

    # The guard turned the node off instead of dividing by zero on `frame_id % 0`.
    assert reader.is_running is False
