from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING
from xml.dom import minidom

from neudc.utils import LOGGER

from .base import BaseAnnotator

if TYPE_CHECKING:
    import numpy as np

__all__ = ("PascalVOCAnnotator",)


class PascalVOCAnnotator(BaseAnnotator):
    """Pascal VOC format annotation handler."""

    def __init__(
        self,
        save_dir: str,
        categories: list[dict] | None = None,
    ) -> PascalVOCAnnotator:
        """Initialize the PascalVOCAnnotator.

        Args:
        ----
            save_dir (str): Directory to save the annotations.
            categories (Optional[List[Dict]]): List of categories in the format [{"id": int, "name": str}, ...].

        """
        super().__init__(save_dir, categories)
        self.id_to_name = {cat["id"]: cat["name"] for cat in self.categories}

    def add_detections(
        self,
        image_id: int,
        image_name: str,
        frame: np.ndarray,
        dets: np.ndarray,
    ) -> None:
        """Add detections to the PascalVOC annotations.

        Args:
        ----
            image_id (int): Unique ID of the image.
            image_name (str): Name of the image.
            frame (np.ndarray): Original image.
            dets (np.ndarray): Detections in the format [{"xmin": float, "ymin": float, "xmax": float, "ymax": float, "score": float, "class_id": int}, ...].

        """
        height, width = frame.shape[:2]
        depth = frame.shape[2] if len(frame.shape) > 2 else 3
        xml_path = os.path.join(self.save_dir, os.path.splitext(image_name)[0] + ".xml")

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = os.path.basename(self.save_dir)
        ET.SubElement(root, "filename").text = image_name
        ET.SubElement(root, "path").text = os.path.abspath(os.path.join(self.save_dir, image_name))

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)

        ET.SubElement(root, "segmented").text = "0"

        for bbox in dets:
            xmin, ymin, xmax, ymax, _, cls_id = bbox
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = self.id_to_name[int(cls_id)]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(xmin))
            ET.SubElement(bndbox, "ymin").text = str(int(ymin))
            ET.SubElement(bndbox, "xmax").text = str(int(xmax))
            ET.SubElement(bndbox, "ymax").text = str(int(ymax))

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(xml_path, "w") as f:
            f.write(xml_str)

    def finalize(self, _: str | None = None) -> None:
        """Finalize the PascalVOC annotations."""
        LOGGER.info(f"PascalVOC annotations saved to {self.save_dir}")
