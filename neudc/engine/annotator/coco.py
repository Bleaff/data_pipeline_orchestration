import json
import os
from typing import Optional

import numpy as np

from neudc.utils import LOGGER

from .base import BaseAnnotator

__all__ = ("COCOAnnotator",)


class COCOAnnotator(BaseAnnotator):
    """COCO format annotation handler."""

    def __init__(
        self,
        save_dir: str,
        categories: Optional[list[dict]] = None,
    ) -> "COCOAnnotator":
        """
        Initialize the COCOAnnotator.

        Args:
            save_dir (str): Directory to save the JSON file.
            categories (Optional[List[Dict]]): List of categories in the format [{"id": int, "name": str}, ...].
        """
        super().__init__(save_dir, categories)
        self.images = []
        self.annotations = []
        self.ann_id = 0

    def add_detections(
        self,
        image_id: int,
        image_name: str,
        frame: np.ndarray,
        dets: np.ndarray,
    ) -> None:
        """
        Add detections to the COCO annotations.

        Args:
            image_id (int): Unique ID of the image.
            image_name (str): Name of the image.
            frame (np.ndarray): Original image.
            dets (np.ndarray): Detections in the format [{"xmin": float, "ymin": float, "xmax": float, "ymax": float, "score": float, "class_id": int}, ...].
        """
        height, width = frame.shape[:2]
        # Add image metadata if not exists
        if not any(img["id"] == image_id for img in self.images):
            self.images.append({"id": image_id, "file_name": image_name, "width": width, "height": height})

        # Add annotations
        for bbox in dets:
            xmin, ymin, xmax, ymax, score, cls_id = bbox
            w = xmax - xmin
            h = ymax - ymin

            self.annotations.append(
                {
                    "id": self.ann_id,
                    "image_id": image_id,
                    "category_id": int(cls_id),
                    "bbox": [float(xmin), float(ymin), float(w), float(h)],
                    "score": float(score),
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            self.ann_id += 1

    def finalize(self, filename: str = "annotations_coco.json") -> None:
        """
        Finalize the COCO annotations.

        Args:
            filename (str): Name of the file to save the annotations.
        """
        output_path = os.path.join(self.save_dir, filename)
        coco_data = {"images": self.images, "annotations": self.annotations, "categories": self.categories}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
        LOGGER.info(f"COCO annotations saved to {output_path}")
