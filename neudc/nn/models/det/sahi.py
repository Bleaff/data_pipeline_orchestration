from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neudc.nn.models.op import apply_nms, make_sahi_slices_batch, postprocess_yolo_outputs
from neudc.utils import LOGGER, PROFILE_FREQ, NoProfile, Profile
from neudc.utils.checks import to_tuple

from .base import BaseDetector

if TYPE_CHECKING:
    from neudc.utils.types import (
        FloatBBoxesWithCls,
        FloatFeaturesBatch,
        FloatImagesBatch,
        ImageShape,
        LetterboxParams,
        UInt8HWC,
    )

# from numba import jit, prange


__all__ = ("SAHIDetector",)


class SAHIDetector(BaseDetector):
    """Numba-optimized SAHI sliced strategy with multiple detector support."""

    def __init__(
        self,
        detector: BaseDetector,
        imgsz: ImageShape,
        crop_size: ImageShape = (640, 640),
        crop_overlap: ImageShape = (100, 100),
    ) -> SAHIDetector:
        """Initialize the SAHI detector.

        Args:
        ----
            detector (BaseDetector): Detector for inference the model
            imgsz: (int | tuple[int, int]): Full image size for the inference (height, width).
            crop_size: (int | tuple[int, int]): Image size for the inference (height, width).
            crop_overlap: (int | tuple[int, int]): Overlap for the inference (height, width).

        """
        self.detector = detector

        # convert to tuple
        imgsz, crop_size, crop_overlap = to_tuple(imgsz), to_tuple(crop_size), to_tuple(crop_overlap)

        # validate imgsz from detector
        if self.detector.imgsz[0] != crop_size[0] or self.detector.imgsz[1] != crop_size[1]:
            LOGGER.warning(f"WARNING ⚠️ Detector imgsz {self.detector.imgsz} is not equal to the crop size {crop_size}.")
            self.crop_size = self.detector.imgsz
        else:
            self.crop_size = crop_size

        # validate crop size
        if imgsz[0] < crop_size[0] or imgsz[1] < crop_size[1]:
            LOGGER.warning(f"WARNING ⚠️ Crop size {crop_size} is larger than the image size {self.imgsz}.")
            if imgsz[0] < crop_size[0]:
                self.imgsz = (crop_size[0], imgsz[1])
            if imgsz[1] < crop_size[1]:
                self.imgsz = (imgsz[0], crop_size[1])
        else:
            self.imgsz = imgsz

        # validate crop overlap
        if crop_overlap[0] >= crop_size[0] or crop_overlap[1] >= crop_size[1]:
            LOGGER.warning(f"WARNING ⚠️ Crop overlap {crop_overlap} is larger than the crop size {crop_size}.")
            self.crop_overlap = (0, 0)
        elif crop_overlap[0] < 0 or crop_overlap[1] < 0:
            LOGGER.warning(f"WARNING ⚠️ Crop overlap {crop_overlap} is negative.")
            self.crop_overlap = (0, 0)
        else:
            self.crop_overlap = crop_overlap

        # Increase the image size to be divisible by crop size and overlap
        adjusted_imgsz = self._adjust_imgsz_to_crop_overlap(self.imgsz, self.crop_size, self.crop_overlap)
        if adjusted_imgsz != self.imgsz:
            LOGGER.warning(
                f"WARNING ⚠️ Adjusted inference image size from {self.imgsz} to {adjusted_imgsz} "
                f"to ensure integer number of crops with crop size {self.crop_size} and overlap {self.crop_overlap}.",
            )
            self.imgsz = adjusted_imgsz

        # Calculate the number of crops
        self.ncrops = (
            (self.imgsz[0] - self.crop_overlap[0])
            // (self.crop_size[0] - self.crop_overlap[0])
            * (self.imgsz[1] - self.crop_overlap[1])
            // (self.crop_size[1] - self.crop_overlap[1])
        )

    @staticmethod
    def _adjust_imgsz_to_crop_overlap(
        imgsz: tuple[int, int],
        crop_size: tuple[int, int],
        overlap: tuple[int, int],
    ) -> tuple[int, int]:
        """Adjust the image size to be divisible by crop size and overlap.

        Args:
        ----
            imgsz (tuple[int, int]): Image size.
            crop_size (tuple[int, int]): Crop size.
            overlap (tuple[int, int]): Overlap.

        Returns:
        -------
            (tuple[int, int]): Adjusted image size.

        """
        adjusted_size = []
        for i in range(2):
            step = crop_size[i] - overlap[i]
            if (imgsz[i] - overlap[i]) % step != 0:
                adjusted = ((imgsz[i] - overlap[i]) // step + 1) * step + overlap[i]
            else:
                adjusted = imgsz[i]
            adjusted_size.append(adjusted)

        return tuple(adjusted_size)

    @Profile(use_cuda=False, logger=LOGGER, freq=PROFILE_FREQ)
    def pre_transform(
        self,
        ims: list[UInt8HWC],
    ) -> tuple[FloatImagesBatch, list[LetterboxParams]]:
        """Pre-transform input image in BGR format before inference.

        Args:
        ----
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
        -------
            (tuple): A list of transformed images, letterbox params, and origins.

        """
        letterbox_images, letterbox_params = self.detector._pre_transform(
            ims=ims,
            imgsz=self.imgsz,
            stride=self.detector.stride,
            fp16=self.detector.backend.fp16,
        )  # letterbox_images: (n, 3, h, w), letterbox_params: (n, 2)

        full_crops, full_origins = make_sahi_slices_batch(
            imgs=letterbox_images,
            crop_size=self.crop_size,
            crop_overlap=self.crop_overlap,
        )

        # merge letterbox_params and origins
        # Expand letterbox_params to match number of crops by repeating for each crop's image_index
        letterbox_params = [
            np.array([*letterbox_params[origin[0]], *origin], dtype=np.float32) for origin in full_origins
        ]  # (n, 5)

        return np.stack(full_crops), letterbox_params

    @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True)
    def _post_transform(
        predictions: list[FloatFeaturesBatch],
        letterbox_params: list[LetterboxParams],
        conf: float,
        iou: float,
        max_det: int,
        n_ims: int,
    ) -> list[FloatBBoxesWithCls]:
        """Numba-compiled post-transform input image before inference.

        Args:
        ----
            predictions (Float(np.ndarray)): (B, ...) as output from a network.
            letterbox_params (List(LetterboxParams)): Ratios, pads and origins of every image after letterbox, (B, 5)
            conf (float): Confidence threshold.
            iou (float): IoU threshold.
            max_det (int): Maximum number of detections to return.
            n_ims (int): Number of images.

        Returns:
        -------
            (list): Rescaled bboxes.

        """
        b = len(predictions)  # batch = number of crops * number of images
        ncrops = b // n_ims  # number of crops
        output = [np.zeros((max_det * ncrops, 6), dtype=predictions.dtype) for _ in range(n_ims)]
        counts = np.zeros(n_ims, dtype=np.int64)
        for i in range(b):
            rx, ry, px, py, image_id, ox, oy = letterbox_params[i]
            image_id = int(image_id)
            dets = postprocess_yolo_outputs(
                predictions=predictions[i],
                conf=conf,
                iou=iou,
                max_det=max_det,
                ratio=(rx, ry),
                pad=(px, py),
            )
            dets[:, 0] += ox
            dets[:, 1] += oy
            dets[:, 2] += ox
            dets[:, 3] += oy
            output[image_id][counts[image_id] : counts[image_id] + len(dets)] = dets
            counts[image_id] += len(dets)

        for i in range(n_ims):
            bboxes = output[i][: counts[i]]
            keep = apply_nms(
                boxes=bboxes[:, :4],
                scores=bboxes[:, 4],
                iou=iou,
                max_det=max_det,
            )
            output[i] = bboxes[keep]

        return output

    @Profile(use_cuda=False, logger=LOGGER, freq=PROFILE_FREQ)
    def post_transform(
        self,
        predictions: FloatFeaturesBatch,
        letterbox_params: list[LetterboxParams],
    ) -> list[FloatBBoxesWithCls]:
        """Post-transform input image before inference.

        Args:
        ----
            predictions (Float(np.ndarray)): (B, ...) as output from a network, B = K * N
            letterbox_params: List(LetterboxParams): Ratios and pads of every image after letterbox, (K, 2)
            origins: List(Tuple[int, int]): Origins of every image after letterbox, (N, 3)

        Returns:
        -------
            (list): Rescaled bboxes.

        """
        return self._post_transform(
            predictions=predictions,
            letterbox_params=letterbox_params,
            conf=self.detector.conf,
            iou=self.detector.iou,
            max_det=self.detector.max_det,
            n_ims=len(letterbox_params) // self.ncrops,
        )

    def __call__(
        self,
        ims: list[UInt8HWC],
    ) -> list[FloatBBoxesWithCls]:
        """Runs inference on the YOLOv8 model.

        Args:
        ----
            ims (List(np.ndarray)): [(H, W, C) x N] for list.

        Returns:
        -------
            (List[Tuple[np.ndarray]]): Tuple containing the bboxes, score, class_id

        """
        batch_ims, batch_params = self.pre_transform(ims)
        predictions = self.detector.backend(batch_ims)[0]  # yolov8 has only one output
        return self.post_transform(
            predictions=predictions,
            letterbox_params=batch_params,
        )

    @NoProfile
    def warmup(
        self,
        iters: int = 10,
    ) -> None:
        """Warm up the model by running one forward pass with a dummy input.

        Args:
        ----
            iters (int): Number of iterations to warm up the model.

        """
        # First, warm up the underlying detector.
        self.detector.warmup(iters=iters)

        # Then, warm up the SAHI detector.
        im = [
            np.empty(
                shape=(*self.imgsz, 3),
                dtype=np.uint8,
            ),
        ]  # input

        for _ in range(iters):
            self(im)  # warmup

    def plot(self, *args, **kwargs) -> None:
        self.detector.plot(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"SAHI("
            f"detector={self.detector.__repr__()}, "
            f"imgsz={self.imgsz}, "
            f"crop_size={self.crop_size}, "
            f"crop_overlap={self.crop_overlap}"
            f")"
        )
