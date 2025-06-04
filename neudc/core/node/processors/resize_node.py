"""A node that resizes incoming Frame objects to a target resolution.

This node is useful for cases where processing needs to be done on images of a specific size.
The node will resize any incoming Frame object to the target resolution.

The target resolution is specified in the configuration dictionary as 'target_width' and 'target_height'.

Example:
-------
    >>> config = {
    ...     "mailbox": mailbox,
    ...     "logger": logger,
    ...     "target_width": 640,
    ...     "target_height": 480
    ... }
    >>> resize_node = ResizeNode.from_config(config)

"""

from __future__ import annotations

from typing import Any

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.node.processors.mixins.resize_mixin import ResizeLogicMixin


class ResizeNode(ResizeLogicMixin, BaseThreadedNode):
    """A node that resizes incoming Frame objects to a target resolution."""

    def __init__(self, target_width: int, target_height: int, *args: Any, **kwargs: Any) -> None:
        """Initialize the ResizeNode with target dimensions."""
        self.target_width = target_width
        self.target_height = target_height
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_config(config: dict[str, Any]) -> ResizeNode:
        """Create ResizeNode from configuration dictionary.

        Args:
        ----
            config (dict): Configuration dict containing 'mailbox', 'logger', 'target_width', and 'target_height'.

        Returns:
        -------
            ResizeNode: Instantiated ResizeNode.

        """
        return ResizeNode(
            mailbox=config["mailbox"],
            logger=config["logger"],
            target_width=config["target_width"],
            target_height=config["target_height"],
        )
