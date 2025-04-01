import os
from abc import ABC, abstractmethod
from typing import Optional

__all__ = ("BaseAnnotator",)


class BaseAnnotator(ABC):
    """Base class for different annotation formats."""

    def __init__(
        self,
        save_dir: str,
        categories: Optional[list[dict]] = None,
    ) -> "BaseAnnotator":
        """
        Initialize the BaseAnnotator.

        Args:
            save_dir (str): Directory to save the annotations.
            categories (Optional[List[Dict]]): List of categories in the format [{"id": int, "name": str}, ...].
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.categories = categories or [{"id": 0, "name": "class_0"}]

    @abstractmethod
    def add_detections(self, image_id: int, image_name: str, frame, dets) -> None:
        """Add detections for a specific image.

        Args:
            image_id (int): Unique ID of the image.
            image_name (str): Name of the image.
            frame (np.ndarray): Original image.
            dets (np.ndarray): Detections in the format [{"xmin": float, "ymin": float, "xmax": float, "ymax": float, "score": float, "class_id": int}, ...].
        """
        ...

    @abstractmethod
    def finalize(self, filename: Optional[str] = None) -> None:
        """Finalize and save annotations.

        Args:
            filename (Optional[str]): Name of the file to save the annotations.
        """
        ...
