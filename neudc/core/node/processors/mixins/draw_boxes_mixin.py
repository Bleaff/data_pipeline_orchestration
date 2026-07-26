"""Mixin class for drawing boxes on Frame objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import Frame


class DrawBoxesLogicMixin:
    """Mixin class for drawing boxes on Frame objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the mixin, forwarding all arguments to the next class in the MRO."""
        super().__init__(*args, **kwargs)

    def draw_boxes(self, frame: Frame) -> Frame:
        """Draw boxes on the image in the Frame object."""
        for box in frame.boxes:
            cv2.rectangle(frame.image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 255, 0), 2)
        return frame

    def process(self, frame: Frame) -> Frame:
        """Draw boxes on the image in the Frame object."""
        return self.draw_boxes(frame)
