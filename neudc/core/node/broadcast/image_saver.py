"""Image Saver Node.

This node saves incoming Frame objects to disk in the specified directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from neudc.core.base.base_thread import BaseThreadedNode

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import Frame


class SaveImageNode(BaseThreadedNode):
    """A node that saves incoming Frame images to disk in the specified directory."""

    def __init__(self, mailbox: Any, logger: Any, save_dir: str) -> None:
        """Initialize the SaveImageNode."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(mailbox, logger)

    @staticmethod
    def from_config(config: dict[str, Any]) -> SaveImageNode:
        """Create SaveImageNode from configuration.

        Args:
        ----
            config (dict): Dictionary containing 'mailbox', 'logger', 'save_dir'.

        Returns:
        -------
            SaveImageNode: Instantiated node.

        """
        return SaveImageNode(
            mailbox=config["mailbox"],
            logger=config["logger"],
            save_dir=config["save_dir"],
        )

    def process(self, frame: Frame) -> Frame:
        """Save the image from a Frame to disk.

        Args:
        ----
            frame (Frame): Input frame with image.

        Returns:
        -------
            Frame: The same frame, unmodified.

        """
        filename = self.save_dir / f"frame_{frame.frame_id}.jpg"
        success = cv2.imwrite(str(filename), frame.image)

        if success:
            self.logger.debug(f"Saved frame {frame.frame_id} to {filename}")
        else:
            self.logger.warning(f"Failed to save frame {frame.frame_id} to {filename}")

        return frame
