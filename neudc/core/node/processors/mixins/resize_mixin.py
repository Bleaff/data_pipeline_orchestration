"""Provide Mixin class with logic for resizing Frame objects.

This class is designed to be used as a mixin to add resizing functionality to nodes.
It resizes the `image` attribute of incoming Frame objects to the target dimensions
specified during initialization.

Attributes
----------
target_width (int): The target width for resizing frames.
target_height (int): The target height for resizing frames.

Methods
-------
process(frame):
    Resizes the `image` of the given Frame object to the target dimensions.
    Logs the resizing operation along with the process ID and frame ID.

"""

import os
from typing import Any

import cv2

from neudc.core.communication.messaging.types import Frame


class ResizeLogicMixin:
    """Mixin class for resizing Frame objects."""

    def __init__(self, target_width: int, target_height: int, *args: Any, **kwargs: Any) -> None:
        """Initialize the mixin with target dimensions."""
        super().__init__(*args, **kwargs)
        self.target_width = target_width
        self.target_height = target_height

    def process(self, frame: Frame) -> Frame:
        """Resize the image in the Frame object to the target dimensions."""
        resized_image = cv2.resize(frame.image, (self.target_width, self.target_height))
        frame.image = resized_image
        self.logger.debug(  # type: ignore[attr-defined]
            f"[{self.name}] Running in PID: {os.getpid()}\nResized frame {frame.frame_id} to {self.target_width}x{self.target_height}",  # type: ignore[attr-defined]
        )
        return frame
