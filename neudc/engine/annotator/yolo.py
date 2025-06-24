import os
from typing import Optional

import numpy as np

from neudc.utils import LOGGER

from .base import BaseAnnotator

__all__ = ("YOLOAnnotator",)


class YOLOAnnotator(BaseAnnotator):
    """YOLO format annotation handler."""

    def add_detections(
        self,
        image_id: int,
        image_name: str,
        frame: np.ndarray,
        dets: np.ndarray,
    ) -> None:
        """Add detections to the YOLO annotations.

        Args:
        ----
            image_id (int): Unique ID of the image.
            image_name (str): Name of the image.
            frame (np.ndarray): Original image.
            dets (np.ndarray): Detections in the format [{"xmin": float, "ymin": float, "xmax": float, "ymax": float, "score": float, "class_id": int}, ...].

        """
        height, width = frame.shape[:2]
        txt_path = os.path.join(self.save_dir, os.path.splitext(image_name)[0] + ".txt")

        with open(txt_path, "w") as f:
            for bbox in dets:
                xmin, ymin, xmax, ymax, _, cls_id = bbox

                # Convert to YOLO format
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height

                f.write(f"{int(cls_id)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    def finalize(self, _: Optional[str] = None) -> None:
        """Finalize the YOLO annotations."""
        LOGGER.info(f"YOLO annotations saved to {self.save_dir}")
