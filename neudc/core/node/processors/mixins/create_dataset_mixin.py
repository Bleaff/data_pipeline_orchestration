from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from neudc.utils import LOGGER

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import Frame


class CreateDatasetMixin:
    """
    Mixin class for saving frames and their corresponding YOLO-format label files
    to a specified dataset directory. Intended to be combined with processing nodes
    that receive frames during runtime.

    Attributes:
        save_dir (Path): The base directory where dataset images and label files will be stored.
    """

    def __init__(self, save_dir: str | Path, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the mixin with the base save directory.

        Args:
            save_dir (str | Path): Base path where images and labels should be saved.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(*args, **kwargs)

    def process(self, frame: Frame) -> Frame:
        """
        Processes a single frame: saves the image and writes the associated
        label file in YOLO format.

        Args:
            frame (Frame): A frame containing image data and detection boxes.

        Returns:
            Frame: The same frame, unmodified.
        """
        save_dir_images, save_dir_labels = self.get_output_dirs(frame)

        image_path = save_dir_images / f"frame_{frame.frame_id}.jpg"
        self.save_image(frame, image_path)

        label_path = save_dir_labels / f"frame_{frame.frame_id}.txt"
        self.save_labels(frame, label_path)

        return frame

    def get_output_dirs(self, frame: Frame) -> tuple[Path, Path]:
        """
        Constructs image and label directory paths for a given frame,
        based on its source folder name.

        Args:
            frame (Frame): Frame from which the source path is extracted.

        Returns:
            tuple[Path, Path]: A tuple containing the image directory and label directory paths.
        """
        folder_name = Path(frame.source_frame).parent.name
        save_dir_images = self.save_dir / folder_name / "images"
        save_dir_labels = self.save_dir / folder_name / "labels"

        save_dir_images.mkdir(parents=True, exist_ok=True)
        save_dir_labels.mkdir(parents=True, exist_ok=True)

        return save_dir_images, save_dir_labels

    def save_image(self, frame: Frame, image_path: Path) -> None:
        """
        Saves the image data from the frame to disk.

        Args:
            frame (Frame): The frame containing the image to save.
            image_path (Path): The full path where the image should be saved.
        """
        success = cv2.imwrite(str(image_path), frame.image)
        if success:
            LOGGER.debug(f"Saved frame {frame.frame_id} to {image_path}")
        else:
            LOGGER.warning(f"Failed to save frame {frame.frame_id} to {image_path}")

    def save_labels(self, frame: Frame, label_path: Path) -> None:
        """
        Saves detection boxes from the frame to a label file in YOLO format.

        If no boxes are present, an empty file is created.

        Args:
            frame (Frame): The frame containing detection boxes.
            label_path (Path): The full path to the output `.txt` label file.
        """
        height, width = frame.image.shape[:2]

        with open(label_path, "w") as f:
            if frame.boxes:
                for box in frame.boxes:
                    x_center = (box.x1 + box.x2) / 2 / width
                    y_center = (box.y1 + box.y2) / 2 / height
                    w = (box.x2 - box.x1) / width
                    h = (box.y2 - box.y1) / height
                    class_id = box.class_id_num

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            else:
                LOGGER.debug(f"No boxes for frame {frame.frame_id}, created empty label file.")
