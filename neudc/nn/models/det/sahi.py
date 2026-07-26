"""SAHI model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from neudc.nn.models.op import apply_nms, calculate_slices_coordinates, letterbox, postprocess_yolo_outputs
from neudc.utils import LOGGER, PROFILE_FREQ, USE_NUMBA, NoProfile, Profile, conditional_jit, get_profile
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

    from .yolo import YOLOv8

__all__ = ("SAHIDetector",)


class SAHIDetector(BaseDetector):
    """Numba-optimized SAHI sliced strategy with multiple detector support."""

    def __init__(
        self,
        detector: YOLOv8,
        crop_size: ImageShape = (640, 640),
        crop_overlap: ImageShape = (100, 100),
        batch: int = 16,
    ) -> None:
        """Initialize the SAHI detector.

        Args:
        ----
            detector (BaseDetector): Detector for inference the model
            crop_size: (int | tuple[int, int]): Image size for the inference (height, width).
            crop_overlap: (int | tuple[int, int]): Overlap for the inference (height, width).
            batch: (int): Batch data. Use -1 to put all.

        """
        self.detector = detector
        self.batch = batch

        # convert to tuple
        crop_size, crop_overlap = to_tuple(crop_size), to_tuple(crop_overlap)

        # validate imgsz from detector
        if self.detector.imgsz[0] != crop_size[0] or self.detector.imgsz[1] != crop_size[1]:
            LOGGER.warning(
                f"WARNING ⚠️ Detector imgsz {self.detector.imgsz} is not equal to the crop size {crop_size}.",
            )
            self.crop_size = self.detector.imgsz
        else:
            self.crop_size = crop_size

        # validate crop overlap
        if crop_overlap[0] >= crop_size[0] or crop_overlap[1] >= crop_size[1]:
            LOGGER.warning(f"WARNING ⚠️ Crop overlap {crop_overlap} is larger than the crop size {crop_size}.")
            self.crop_overlap = (0, 0)
        elif crop_overlap[0] < 0 or crop_overlap[1] < 0:
            LOGGER.warning(f"WARNING ⚠️ Crop overlap {crop_overlap} is negative.")
            self.crop_overlap = (0, 0)
        else:
            self.crop_overlap = crop_overlap

    @staticmethod
    @conditional_jit(nopython=True, fastmath=True, parallel=False, inline="always", turn_on=USE_NUMBA)
    def make_slices(
        im: UInt8HWC,
        crop_size: ImageShape,
        crop_overlap: ImageShape,
        im_id: int = 0,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Make slices from the image."""
        crop_images, crop_params = [], []
        slice_indexes = calculate_slices_coordinates(
            imgsz=im.shape[:2],
            crop_size=crop_size,
            crop_overlap=crop_overlap,
        )
        # numba nopython mode does not support zip(strict=...); these arrays are always equal length by construction.
        for y1, x1, y2, x2 in zip(*slice_indexes):  # noqa: B905
            crop = im[y1:y2, x1:x2, :]
            crop_images.append(crop)
            crop_params.append(np.array([1, 1, 0, 0, im_id, y1, x1], dtype=np.float32))

        return crop_images, crop_params

    @Profile(use_cuda=False, freq=PROFILE_FREQ)
    def pre_transform(
        self,
        ims: list[UInt8HWC],
    ) -> tuple[FloatImagesBatch, LetterboxParams]:
        """Pre-transform input image in BGR format before inference.

        Args:
        ----
            ims (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
        -------
            (tuple): A list of transformed images, letterbox params, and origins.

        """
        crop_params: LetterboxParams = []
        crop_images = []

        for im_id, im in enumerate(ims):
            # step 1. adjust img to crop & overlap (in-place BGR->RGB conversion)
            cv2.cvtColor(src=im, code=cv2.COLOR_BGR2RGB, dst=im)

            # step 2. letterbox img to crop
            letterbox_im, letterbox_param = letterbox(
                img=im,
                auto=False,
                stride=self.detector.stride,
                new_shape=self.crop_size,
            )

            crop_params.append(np.array([*letterbox_param, im_id, 0, 0], dtype=np.float32))
            crop_images.append(letterbox_im)

            if im.shape[0] <= self.crop_size[0] and im.shape[1] <= self.crop_size[1]:
                LOGGER.warning(f"WARNING ⚠️ image size {im.shape} is less than crop size {self.crop_size}, skip.")
            else:

                crops, params = self.make_slices(
                    im=im,
                    crop_overlap=self.crop_overlap,
                    crop_size=self.crop_size,
                    im_id=im_id,
                )

                crop_params.extend(params)
                crop_images.extend(crops)

        # SAHIDetector deliberately reuses its wrapped detector's private normalize step.
        crop_images_arr = self.detector._pre_transform_normalize(  # noqa: SLF001
            ims=crop_images,
            fp16=self.detector.backend.fp16,
        )

        return crop_images_arr, crop_params

    @staticmethod
    @conditional_jit(nopython=True, fastmath=True, parallel=False, inline="always", turn_on=USE_NUMBA)
    def _post_transform(  # noqa: PLR0917 - numba-jitted postprocess, one parameter per decode/NMS tunable
        predictions: list[FloatFeaturesBatch],
        letterbox_params: LetterboxParams,
        n_ims: int,
        conf: float = 0.1,
        iou: float = 0.7,
        max_det: int = 100,
    ) -> list[FloatBBoxesWithCls]:
        """Numba-compiled post-transform input image before inference.

        Args:
        ----
            predictions (Float(np.ndarray)): (B, ...) as output from a network.
            letterbox_params (List(LetterboxParams)): Ratios, pads and origins of every image after letterbox, (B, 5)
            n_ims (int): Number of original images.
            conf (float): Confidence threshold.
            iou (float): IoU threshold.
            max_det (int): Maximum number of detections to return.

        Returns:
        -------
            (list): Rescaled bboxes.

        """
        b = len(predictions)
        output = [np.zeros((max_det, 6), dtype=predictions[0].dtype) for _ in range(n_ims)]
        counts = np.zeros(n_ims, dtype=np.int64)
        for i in range(b):
            ry, rx, py, px, image_id, oy, ox = letterbox_params[i]
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

    @Profile(use_cuda=False, freq=PROFILE_FREQ)
    def post_transform(
        self,
        predictions: FloatFeaturesBatch,
        letterbox_params: LetterboxParams,
        n_ims: int,
    ) -> list[FloatBBoxesWithCls]:
        """Post-transform input image before inference.

        Args:
        ----
            predictions (Float(np.ndarray)): (B, ...) as output from a network, B = K * N
            letterbox_params: List(LetterboxParams): Ratios, pads and origins of every image after letterbox, (B, 5)
            n_ims (int): Number of original images.

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
            n_ims=n_ims,
        )

    def __call__(
        self,
        ims: list[UInt8HWC],
    ) -> list[FloatBBoxesWithCls]:
        """Run SAHI-tiled inference using the wrapped detector.

        Args:
        ----
            ims (List(np.ndarray)): [(H, W, C) x N] for list.

        Returns:
        -------
            (List[Tuple[np.ndarray]]): Tuple containing the bboxes, score, class_id

        """
        batch_ims, batch_params = self.pre_transform(ims)

        with get_profile(use_cuda=False, use_torch=True, freq=PROFILE_FREQ, name="sahi full backend"):
            tmp_batch = len(batch_ims)
            if self.batch == -1 or tmp_batch < self.batch:
                predictions = self.detector.backend(batch_ims)[0]
            else:
                # We break the list of crops into batches of size self.batch
                pred_chunks = []
                for i in range(0, tmp_batch, self.batch):
                    chunk = batch_ims[i : i + self.batch]
                    # Run the detector on the chunk
                    pred_chunk = self.detector.backend(chunk)[0]
                    pred_chunks.append(pred_chunk)
                # Concatenate along the batch dimension (axis=0)
                predictions = np.concatenate(pred_chunks, axis=0)

        return self.post_transform(
            predictions=predictions,
            letterbox_params=batch_params,
            n_ims=len(ims),
        )

    @NoProfile  # type: ignore[call-arg]  # NoProfile is a singleton instance mistyped as a class by mypy
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
        self.detector.warmup(iters=iters)  # type: ignore[call-arg]  # NoProfile is a singleton instance mistyped as a class by mypy

        # Then, warm up the SAHI detector.
        im = [
            np.empty(
                shape=(self.crop_size[0] * 2 - self.crop_overlap[0], self.crop_size[1] * 2 - self.crop_overlap[1], 3),
                dtype=np.uint8,
            ),
        ]  # input

        for _ in range(iters):
            self(im)  # warmup

    # Transparent passthrough to the wrapped detector's plot().
    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Delegate plotting to the wrapped detector; see `BaseDetector.plot`."""
        self.detector.plot(*args, **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return (
            f"SAHI("
            f"detector={self.detector.__repr__()}, "
            f"crop_size={self.crop_size}, "
            f"crop_overlap={self.crop_overlap}, "
            f"batch={self.batch}."
            f")"
        )
