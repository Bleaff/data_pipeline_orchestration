"""ImageReader class.

This class reads images from a given path and puts them in a queue.
The class is designed to be used as a node in a dataflow graph.
The class is thread-safe and can be used with asyncio.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.messaging.types import Frame  # Frame class as given


class FolderImageNode(BaseThreadedNode):
    """A node that reads images from a folder and sends Frame objects via mailbox."""

    def __init__(
        self,
        folder_path: str,
        mailbox: Any,
        logger: Any,
        mode: str = "loop",
        frame_delay: float = 0.01,
    ) -> None:
        """Initialize FolderImageNode.

        Args:
        ----
            folder_path (str): Path to the folder containing images.
            mailbox (Any): Mailbox for sending Frame objects.
            logger (Any): Logger for debug/info messages.
            mode (str): Mode of reading images ("loop" or "only_one").
            frame_delay (float): Delay between sending frames (in seconds).

        """
        self.folder_path = Path(folder_path)
        self.mode = mode
        self.frame_delay = frame_delay
        self.image_files = sorted(f.name for f in self.folder_path.iterdir() if f.is_file())
        self.current_index = 0
        self.frame_id = 0
        super().__init__(mailbox, logger)

    @staticmethod
    def from_config(config: dict[str, Any]) -> FolderImageNode:
        """Create a FolderImageNode from configuration dictionary.

        Args:
        ----
            config (dict): Configuration with keys 'folder_path', 'mailbox', 'logger', etc.

        Returns:
        -------
            FolderImageNode: Instantiated FolderImageNode.

        """
        return FolderImageNode(
            folder_path=config["folder_path"],
            mailbox=config["mailbox"],
            logger=config["logger"],
            mode=config.get("mode", "loop"),
            frame_delay=config.get("frame_delay", 0.01),
        )

    def _collect_data(self) -> bool:
        return True

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Process method that reads an image, wraps it into a Frame, and returns it.

        Returns
        -------
            Frame or None: Frame object or None if reading fails or finished.

        """
        del args, kwargs

        if self.current_index >= len(self.image_files):
            if self.mode == "only_one":
                self.logger.info("All images processed in 'only_one' mode.")
                self.stop()
                return None
            self.current_index = 0

        # Load image from disk
        image_path = self.folder_path / self.image_files[self.current_index]
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.warning(f"Failed to read image: {image_path}")
            return None

        # Create Frame object
        timestamp = time.time()
        frame = Frame(
            image=image,
            timestamp=timestamp,
            source_frame=str(image_path),
            frame_id=self.frame_id,
            boxes=[],
        )

        self.logger.debug(f"PID#({os.getpid()}) Sending Frame(id={self.frame_id}) from {image_path}")
        self.current_index += 1
        self.frame_id += 1

        time.sleep(self.frame_delay)
        # self.logger.debug(f"Sleeping for {self.frame_delay:.2f} seconds")
        return frame
