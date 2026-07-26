"""Abstract interface every detector model (YOLOv8, SAHIDetector, ...) implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from neudc.utils import NoProfile

if TYPE_CHECKING:
    from neudc.nn.backends import BaseBackend
    from neudc.utils.types import FloatBBoxesWithCls, FloatFeaturesBatch, FloatImagesBatch, ImageShape, UInt8HWC

__all__ = ("BaseDetector",)


class BaseDetector(ABC):
    """Base class for all detectors."""

    metadata: dict[str, Any]
    imgsz: tuple[int, int]
    names: dict[int, str]
    conf: float
    iou: float
    backend: BaseBackend

    @abstractmethod
    def __init__(  # noqa: PLR0917 - detector config constructor, one flag per tunable
        self,
        path: str,
        backend: type[BaseBackend],
        device_id: int = 0,
        imgsz: ImageShape | None = None,
        names: list | dict | None = None,
        conf: float = 0.2,
        iou: float = 0.7,
    ) -> None:
        """Initialize the BaseDetector for inference.

        Args:
        ----
            path (str): Path to the model weights file.
            backend (BaseBackend): Backend for inference the model
            device_id (int): device id, -1 for cpu device
            imgsz: (int | tuple[int, int]): Image size for the inference height x width.
            names: (list): Class names for the visualisation.
            conf (float): Confidence threshold
            iou (float): Intersection over Union threshold

        """

    @property
    def get_metadata(self) -> dict[str, Any]:
        """Return backend metadata (e.g. input size, class names) collected at load time."""
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
        im_shapes: list[ImageShape],
    ) -> list[FloatBBoxesWithCls]:
        """Post-transform input image before inference.

        Args:
        ----
            predictions (Float(np.ndarray)): (B, ...) as output from a network.
            im_shapes: List(Tuple(int, int)): Image shapes for rescale bboxes to original shape.

        Returns:
        -------
            (list): Rescaled bboxes.

        """

    @abstractmethod
    def __call__(
        self,
        ims: list[UInt8HWC],
    ) -> list[FloatBBoxesWithCls]:
        """Run inference on the YOLOv8 model.

        Args:
        ----
            ims (List(np.ndarray)): [(H, W, C) x N] for list.

        Returns:
        -------
            (List[Tuple[np.ndarray]]): Tuple containing the bboxes, score, class_id

        """

    @NoProfile  # type: ignore[call-arg]  # NoProfile is a singleton instance mistyped as a class by mypy
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

    @abstractmethod
    def plot(
        self,
        image: UInt8HWC,
        bboxes: FloatBBoxesWithCls,
        save_path: str | None = None,
    ) -> None:
        """Plot results from neural network.

        Args:
        ----
            image (np.ndarray): image to plot the bboxes.
            bboxes (tuple(np.ndarray, ...)): bboxes which consists of (bboxs, scores, cls_id)
            save_path (str | None): Optional path to save the plotted image to; if None, not saved.

        Returns:
        -------
            None

        """

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the model."""
