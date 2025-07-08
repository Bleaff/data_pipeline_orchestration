from __future__ import annotations

from typing import Any

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.node.processors.mixins.draw_boxes_mixin import DrawBoxesLogicMixin


class DrawNode(DrawBoxesLogicMixin, BaseThreadedNode):
    """A node that draws boxes on incoming Frame objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> DrawNode:
        """Initialize the DrawNode with target dimensions."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_config(config: dict[str, Any]) -> DrawNode:
        """Create ResizeNode from configuration dictionary."""
        return DrawNode(
            mailbox=config["mailbox"],
        )
