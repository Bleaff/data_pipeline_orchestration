"""YOLOv8 model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from neudc.nn.models.op import letterbox, postprocess_yolo_outputs
from neudc.utils import LOGGER, PROFILE_FREQ, USE_NUMBA, NoProfile, Profile, conditional_jit
from neudc.utils.checks import check_class_names, default_class_names, to_tuple

from .base import BaseDetector

if TYPE_CHECKING:
    from neudc.nn.backends import BaseBackend
    from neudc.utils.types import (
        FloatBBoxesWithCls,
        FloatFeaturesBatch,
        FloatImagesBatch,
        ImageShape,
        LetterboxParams,
        UInt8HWC,
    )

__all__ = ("YOLOv8",)


class YOLOv8(BaseDetector):
    """YOLOv8 with multiple backends and numba postprocess."""

    def __init__(
        self,
        path: str,
        backend: BaseBackend,
        extract_embeddings: bool = False,
        device_id: int = 0,
        imgsz: ImageShape | None = None,
        names: list | dict | None = None,
        conf: float = 0.2,
        iou: float = 0.7,
        max_det: int = 100,
    ) -> YOLOv8:
        """Initialize the YOLOv8 for inference.

        Args:
        ----
            path (str): Path to the model weights file.
            backend (BaseBackend): Backend for inference the model
            device_id (int): device id, -1 for cpu device
            imgsz: (int | tuple[int, int]): Image size for the inference height x width.
            names: (list): Class names for the visualisation.
            conf (float): Confidence threshold
            iou (float): Intersection over Union threshold
            max_det (int): Maximum number of detections to return

        """
        super().__init__(path=path, backend=backend, device_id=device_id)
        backend = backend(
            path=path, device_id=device_id, 
        )
        # Check names
        if not names and "names" not in backend.metadata:  # names missing
            names = default_class_names()
        elif "names" in backend.metadata:
            LOGGER.warning(f"WARNING ⚠️ Overwrite names from {names} to {backend.metadata['names']}.")
            names = backend.metadata["names"]

        self.names = check_class_names(names)

        # Check imgsz
        if not imgsz and "imgsz" not in backend.metadata:  # imgsz missing
            msg = "imgsz is not initialized."
            raise NotImplementedError(msg)
        elif "imgsz" in backend.metadata:
            LOGGER.warning(f"WARNING ⚠️ Overwrite imgsz from {imgsz} to {backend.metadata['imgsz']}.")
            imgsz = backend.metadata["imgsz"]
        self.imgsz = to_tuple(imgsz)

        # Init conf and iou
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.stride = backend.metadata.get("stride", 32)

        # Set backend
        self.backend = backend
        self.path = path
        self.device_id = device_id
        self.extract_embeddings = extract_embeddings

    @staticmethod
    def _pre_transform_normalize(
        ims: list[UInt8HWC],
        fp16: bool = False,
    ) -> FloatImagesBatch:
        """YOLO-like normalize for images."""
        ims = np.stack(ims)
        ims = ims.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        ims = ims.astype(np.float16 if fp16 else np.float32)  # float
        np.divide(ims, 255, out=ims, dtype=ims.dtype)  # 0 - 255 to 0.0 - 1.0

        return ims

    @staticmethod
    def _pre_transform(
        ims: list[UInt8HWC],
        imgsz: list[ImageShape],
        stride: int = 32,
        fp16: bool = False,
    ) -> tuple[FloatImagesBatch, list[LetterboxParams]]:
        """Pre-transform input image in BGR format before inference.

        Args:
        ----
            ims (List(np.ndarray)): images in [(h, w, 3) x N] format.
            imgsz: (int | tuple[int, int]): Image size for the inference height x width.
            stride: (int): Stride of the model.
            fp16: (bool): Use float16 or float32 inference.

        Returns:
        -------
            (tuple): A list of transformed images in (n, 3, h, w) format and letterbox params.

        """
        letterbox_params, letterbox_images = [], []

        for im in ims:
            letterbox_im, letterbox_param = letterbox(img=im, auto=False, stride=stride, new_shape=imgsz)
            cv2.cvtColor(src=letterbox_im, code=cv2.COLOR_BGR2RGB, dst=letterbox_im)
            letterbox_params.append(letterbox_param)
            letterbox_images.append(letterbox_im)

        letterbox_images = YOLOv8._pre_transform_normalize(
            ims=letterbox_images,
            fp16=fp16,
        )

        return letterbox_images, letterbox_params

    @Profile(use_cuda=False, freq=PROFILE_FREQ)
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
            (tuple): A list of transformed images and letterbox params.

        """
        return self._pre_transform(
            ims=ims,
            imgsz=self.imgsz,
            stride=self.stride,
            fp16=self.backend.fp16,
        )

    @staticmethod
    @conditional_jit(nopython=True, fastmath=True, parallel=False, inline="always", turn_on=USE_NUMBA)
    def _post_transform(
        predictions: list[FloatFeaturesBatch],
        letterbox_params: list[LetterboxParams],
        conf: float,
        iou: float,
        max_det: int,
    ) -> list[FloatBBoxesWithCls]:
        """Numba-compiled post-transform input image before inference.

        Args:
        ----
            predictions (Float(np.ndarray)): (B, ...) as output from a network.
            letterbox_params: List(LetterboxParams): Ratio and pad of every image after letterbox
            conf: (float): Confidence threshold
            iou: (float): IoU threshold
            max_det: (int): Maximum number of detections to return

        Returns:
        -------
            (list): Rescaled bboxes.

        """
        output = []
        for i in range(len(predictions)):
            dets = postprocess_yolo_outputs(
                predictions=predictions[i],
                conf=conf,
                iou=iou,
                max_det=max_det,
                ratio=(letterbox_params[i][1], letterbox_params[i][0]),
                pad=(letterbox_params[i][3], letterbox_params[i][2]),
            )
            output.append(dets)

        return output

    @Profile(use_cuda=False, freq=PROFILE_FREQ)
    def post_transform(
        self,
        predictions: FloatFeaturesBatch,
        letterbox_params: list[LetterboxParams],
    ) -> list[FloatBBoxesWithCls]:
        """Post-transform input image before inference.

        Args:
        ----
            predictions (Float(np.ndarray)): (B, ...) as output from a network.
            letterbox_params: List(LetterboxParams): Ratio and pad of every image after letterbox

        Returns:
        -------
            (list): Rescaled bboxes.

        """
        return self._post_transform(
            predictions=predictions,
            letterbox_params=letterbox_params,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
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
        if self.extract_embeddings:
            predictions, embs = self.backend(batch_ims)

            return (
                self.post_transform(
                    predictions=predictions,
                    letterbox_params=batch_params,
                ),
                embs,
            )
        else:
            predictions = self.backend(batch_ims)
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
        im = [
            np.empty(
                shape=(*self.imgsz, 3),
                dtype=np.uint8,
            ),
        ]  # input

        for _ in range(iters):
            self(im)  # warmup

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

        Returns:
        -------
            None

        """
        img_height, img_width = image.shape[:2]

        for bbox in bboxes:
            xmin, ymin, xmax, ymax, score, cls_id = bbox
            xmin, ymin, xmax, ymax, score, cls_name = (
                int(xmin),
                int(ymin),
                int(xmax),
                int(ymax),
                float(score),
                self.names[int(cls_id)],
            )
            cv2.rectangle(
                img=image,
                pt1=(xmin, ymin),
                pt2=(xmax, ymax),
                color=(0, 255, 0),
                thickness=1,
            )

            # Determine text position
            text = f"{cls_name} | {score:.2f}"
            text_size, baseline = cv2.getTextSize(
                text=text,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=1,
            )
            text_width, text_height = text_size

            # Calculate text bottom position including baseline
            total_text_height = text_height + baseline

            # Determine text position with boundary checks
            # Check space above (considering full text height)
            if ymin - total_text_height >= 0:
                # Place above bbox
                text_y = ymin - baseline  # Baseline at top of bbox
                if xmin + text_width > img_width:  # Right-align if needed
                    text_org = (max(0, xmax - text_width), text_y)
                else:  # Left-align
                    text_org = (xmin, text_y)
            # Check space below
            elif ymax + total_text_height <= img_height:
                # Place below bbox
                text_y = ymax + text_height  # Baseline below bbox
                if xmin + text_width > img_width:  # Right-align if needed
                    text_org = (max(0, xmax - text_width), text_y)
                else:  # Left-align
                    text_org = (xmin, text_y)
            else:
                # Not enough space above or below - place inside top
                text_y = ymin + text_height  # Baseline inside bbox
                if xmin + text_width > img_width:  # Right-align if needed
                    text_org = (max(0, xmax - text_width), text_y)
                else:  # Left-align
                    text_org = (xmin, text_y)

            cv2.putText(
                img=image,
                text=text,
                org=text_org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=1,
            )

        if save_path:
            success = cv2.imwrite(save_path, image)
            if success:
                LOGGER.info(f"Image saved to {save_path}")
            else:
                LOGGER.warning(f"WARNING ⚠️ Failed to save image to {save_path}")

    def __repr__(self) -> str:
        return (
            f"YOLOv8("
            f"path={self.path}, "
            f"backend={self.backend}, "
            f"device_id={self.device_id}, "
            f"imgsz={self.imgsz}, "
            f"names={self.names}, "
            f"conf={self.conf}, "
            f"iou={self.iou}"
            f")"
        )
