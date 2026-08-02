"""Latest-frame preview, reusing `SaveImageNode`'s existing on-disk output (#22).

There is no live tap into a running pipeline's frames yet (that needs a new node
type — out of scope here); this only serves whatever a `SaveImageNode` in the
pipeline's own config has already written to its `save_dir`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ("find_latest_frame",)

_SAVE_IMAGE_NODE_TYPE = "SaveImageNode"


def find_latest_frame(nodes_config: list[dict[str, Any]]) -> Path | None:
    """Return the most recently written frame from the pipeline's `SaveImageNode`, if any.

    Args:
    ----
        nodes_config (list[dict[str, Any]]): The pipeline's node configs, as passed to
            `PipelineServiceManager.start_pipeline`.

    Returns:
    -------
        Path | None: The newest `frame_*.jpg` on disk, or `None` if the pipeline has no
            `SaveImageNode` or that node hasn't written a frame yet.

    """
    save_dirs = [node["save_dir"] for node in nodes_config if node.get("type") == _SAVE_IMAGE_NODE_TYPE]
    if not save_dirs:
        return None

    frames = [f for save_dir in save_dirs for f in Path(save_dir).glob("frame_*.jpg")]
    if not frames:
        return None

    return max(frames, key=lambda f: f.stat().st_mtime)
