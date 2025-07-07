from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import cv2

from neudc.utils import LOGGER, NoProfile

if TYPE_CHECKING:
    from neudc.nn.backends import BaseBackend
    from neudc.utils.types import FloatFeaturesBatch, FloatImagesBatch, ImageShape, UInt8HWC

__all__ = ("BaseClsModel",)


class BaseClsModel(ABC):

    @abstractmethod
    def __init__(
        self,
        path: str,
        backend: BaseBackend,
        device_id: int = 0,
        imgsz: ImageShape | None = None,
        names: list | dict | None = None,
        conf: float = 0.2,
    ) -> BaseClsModel:
        """Initialize the BaseClsModel for inference.

        Args:
        ----
            path (str): Path to the model weights file.
            backend (BaseBackend): Backend for inference the model
            device_id (int): device id, -1 for cpu device
            imgsz: (int | tuple[int, int]): Image size for the inference height x width.
            names: (list): Class names for the visualisation.
            conf (float): Confidence threshold

        """

    @property
    def get_metadata(self) -> dict[str, Any]:
        return self.metadata

    @abstractmethod
    def pre_transform(
        self,
        ims: list[UInt8HWC],
    ) -> FloatImagesBatch:
        """Pre-transform input image before inference.

        Args:
        ----
            ims (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
        -------
            (list): A list of transformed images.

        """

    @abstractmethod
    def post_transform(
        self,
        predictions: FloatFeaturesBatch,
    ) -> list[int]:
        """Post-transform input image after inference.

        Args:
        ----
            predictions (Float(np.ndarray)): (B, ...) as output from a network.

        Returns:
        -------
            (list): Rescaled bboxes.

        """

    @abstractmethod
    def __call__(
        self,
        ims: list[UInt8HWC],
        return_embeddings: bool = False,
    ) -> list[int | FloatFeaturesBatch]:
        """Runs inference on the YOLOv8 model.

        Args:
        ----
            ims (List(np.ndarray)): [(H, W, C) x N] for list.

        Returns:
        -------
            (List[Tuple[np.ndarray]]): Tuple containing the bboxes, score, class_id

        """

    @NoProfile
    @abstractmethod
    def warmup(
        self,
        iters: int = 10,
    ) -> None:
        """Warm up the model by running one forward pass with a dummy input.

        Args:
        ----
            iters (int): Number of iters to warmup the backend

        """

    def plot(
        self,
        image: UInt8HWC,
        cls_idx: int,
        padding: int = 10,
        save_path: str | None = None,
    ) -> None:
        """Plot results from neural network.

        Args:
        ----
            image (np.ndarray): image to plot the bboxes.
            cls_idx (int): class id.
            padding (int): padding around the text.
            save_path (str): path to save the image.

        Returns:
        -------
            None

        """
        if self.names and cls_idx < len(self.names):
            class_name = self.names[cls_idx]
        else:
            LOGGER.warning(f"WARNING ⚠️ Class index {cls_idx} is out of range for class names {self.names}")
            class_name = str(cls_idx)

        # Get the size of the text
        (text_width, text_height), _ = cv2.getTextSize(
            text=class_name,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            thickness=2,
        )

        # Calculate the position of the text box (top-right corner)
        box_x = image.shape[1] - text_width - 2 * padding
        box_y = text_height + 2 * padding

        # Draw the background rectangle
        cv2.rectangle(
            img=image,
            pt1=(box_x, box_y - text_height - padding),
            pt2=(image.shape[1], box_y + padding),
            color=(0, 0, 0),
            thickness=-1,
        )

        # Put the text on the image
        cv2.putText(
            img=image,
            text=class_name,
            org=(box_x + padding, box_y - padding),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
        )

        if save_path:
            success = cv2.imwrite(save_path, image)
            if success:
                LOGGER.info(f"Image saved to {save_path}")
            else:
                LOGGER.warning(f"Failed to save image to {save_path}")
