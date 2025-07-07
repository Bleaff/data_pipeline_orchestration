"""Mixin class for drawing boxes on Frame objects."""
from __future__ import annotations
from typing import Any
import cv2
from neudc.core.communication.messaging.types import Frame


class DrawBoxesLogicMixin:
    """Mixin class for drawing boxes on Frame objects."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def draw_boxes(self, frame: Frame) -> Frame:
        """Draw boxes on the image in the Frame object."""
        for box in frame.boxes:
            cv2.rectangle(frame.image, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
        return frame

    def process(self, frame: Frame) -> Frame:
        """Draw boxes on the image in the Frame object."""
        return self.draw_boxes(frame)
