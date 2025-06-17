"""A process node that resizes incoming Frame objects to a target resolution.

This node is useful for cases where processing needs to be done on images of a specific size.
The node will resize any incoming Frame object to the target resolution.

The target resolution is specified in the configuration dictionary as 'target_width' and 'target_height'.
"""

from __future__ import annotations

from typing import Any

from neudc.core.base.base_process import BaseProcessNode
from neudc.core.node.processors.mixins.resize_mixin import ResizeLogicMixin


class ResizeProcessNode(ResizeLogicMixin, BaseProcessNode):
    """A process node that resizes incoming Frame objects to a target resolution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the ResizeProcessNode with target dimensions."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_config(config: dict[str, Any]) -> ResizeProcessNode:
        """Create ResizeProcessNode from configuration dictionary.

        Args:
        ----
            config (dict): Configuration dict containing 'mailbox', 'logger', 'target_width', and 'target_height'.

        Returns:
        -------
            ResizeProcessNode: Instantiated ResizeProcessNode.

        """
        return ResizeProcessNode(
            mailbox=config["mailbox"],
            logger=config["logger"],
            target_width=config["target_width"],
            target_height=config["target_height"],
        )
