from typing import Optional, Union

import cv2
import numpy as np

from neudc.nn.backends import BaseBackend
from neudc.nn.models.op import letterbox, postprocess_yolo_outputs
from neudc.utils import LOGGER, PROFILE_FREQ, NoProfile, Profile
from neudc.utils.checks import check_class_names, default_class_names, to_tuple
from neudc.utils.types import (
    FloatBBoxesWithCls,
    FloatFeaturesBatch,
    FloatImagesBatch,
    ImageShape,
    LetterboxParams,
    UInt8HWC,
)

from .base import BaseDetector

# from numba import jit, prange


__all__ = ("YOLOv8",)


class YOLOv8(BaseDetector):
    """YOLOv8 with multiple backends and numba postprocess."""

    def __init__(
        self,
        path: str,
        backend: BaseBackend,
        device_id: int = 0,
        imgsz: Optional[ImageShape] = None,
        names: Optional[Union[list, dict]] = None,
        conf: float = 0.2,
        iou: float = 0.7,
        max_det: int = 100,
        nms: bool = True,
    ) -> "YOLOv8":
        """
        Initialize the YOLOv8 for inference.

        Args:
            path (str): Path to the model weights file.
            backend (BaseBackend): Backend for inference the model
            device_id (int): device id, -1 for cpu device
            imgsz: (int | tuple[int, int]): Image size for the inference height x width.
            names: (list): Class names for the visualisation.
            conf (float): Confidence threshold
            iou (float): Intersection over Union threshold
            max_det (int): Maximum number of detections to return
            nms (bool): Whether to use non-maximum suppression
        """
        super().__init__(path=path, backend=backend, device_id=device_id)
        # Check names
        if not names and "names" not in backend.metadata:  # names missing
            names = default_class_names()
        elif "names" in backend.metadata:
            LOGGER.warning(f"WARNING ⚠️ Overwrite names from {names} to {backend.metadata['names']}.")
            names = backend.metadata["names"]

        self.names = check_class_names(names)

        # Check imgsz
        if not imgsz and "imgsz" not in backend.metadata:  # imgsz missing
            raise NotImplementedError("imgsz is not initialized.")
        elif "imgsz" in backend.metadata:
            LOGGER.warning(f"WARNING ⚠️ Overwrite imgsz from {imgsz} to {backend.metadata['imgsz']}.")
            imgsz = backend.metadata["imgsz"]
        self.imgsz = to_tuple(imgsz)

        # Init conf and iou
        self.conf = conf
        self.iou = iou
        self.nms = nms
        self.max_det = max_det
        self.stride = backend.metadata.get("stride", 32)

        # Set backend
        self.backend = backend

    @staticmethod
    def _pre_transform(
        ims: list[UInt8HWC],
        imgsz: ImageShape,
        stride: int,
        fp16: bool,
    ) -> tuple[FloatImagesBatch, list[LetterboxParams]]:
        """
        Pre-transform input image in BGR format before inference.

        Args:
            ims (List(np.ndarray)): images in [(h, w, 3) x N] format.
            imgsz: (int | tuple[int, int]): Image size for the inference height x width.
            stride: (int): Stride of the model.

        Returns:
            (tuple): A list of transformed images in (n, 3, h, w) format and letterbox params.
        """

        letterbox_params, letterbox_images = [], []

        for im in ims:
            letterbox_im, letterbox_param = letterbox(img=im, auto=False, stride=stride, new_shape=imgsz)
            cv2.cvtColor(letterbox_im, cv2.COLOR_BGR2RGB, dst=letterbox_im)
            letterbox_params.append(letterbox_param)
            letterbox_images.append(letterbox_im)

        letterbox_images = np.stack(letterbox_images)
        letterbox_images = letterbox_images.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        letterbox_images = letterbox_images.astype(np.float16 if fp16 else np.float32)  # float
        letterbox_images /= 255  # 0 - 255 to 0.0 - 1.0

        return letterbox_images, letterbox_params

    @Profile(use_cuda=False, logger=LOGGER, freq=PROFILE_FREQ)
    def pre_transform(
        self,
        ims: list[UInt8HWC],
    ) -> tuple[FloatImagesBatch, list[LetterboxParams]]:
        """
        Pre-transform input image in BGR format before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (tuple): A list of transformed images and letterbox params.
        """
        return self._pre_transform(
            ims=ims,
            imgsz=self.imgsz,
            stride=self.stride,
            fp16=self.backend.fp16,
        )

    @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True)
    def _post_transform(
        predictions: list[FloatFeaturesBatch],
        letterbox_params: list[LetterboxParams],
        conf: float,
        iou: float,
        max_det: int,
    ) -> list[FloatBBoxesWithCls]:
        """
        Numba-compiled post-transform input image before inference.

        Args:
            predictions (Float(np.ndarray)): (B, ...) as output from a network.
            letterbox_params: List(LetterboxParams): Ratio and pad of every image after letterbox
            conf: (float): Confidence threshold
            iou: (float): IoU threshold
            max_det: (int): Maximum number of detections to return

        Returns:
            (list): Rescaled bboxes.
        """

        output = []
        b = len(predictions)
        for i in range(b):
            dets = postprocess_yolo_outputs(
                predictions=predictions[i],
                conf=conf,
                iou=iou,
                max_det=max_det,
                ratio=(letterbox_params[i][0], letterbox_params[i][1]),
                pad=(letterbox_params[i][2], letterbox_params[i][3]),
            )
            output.append(dets)

        return output

    @Profile(use_cuda=False, logger=LOGGER, freq=PROFILE_FREQ)
    def post_transform(
        self,
        predictions: FloatFeaturesBatch,
        letterbox_params: list[LetterboxParams],
    ) -> list[FloatBBoxesWithCls]:
        """
        Post-transform input image before inference.

        Args:
            predictions (Float(np.ndarray)): (B, ...) as output from a network.
            letterbox_params: List(LetterboxParams): Ratio and pad of every image after letterbox

        Returns:
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
        """
        Runs inference on the YOLOv8 model.

        Args:
            ims (List(np.ndarray)): [(H, W, C) x N] for list.

        Returns:
            (List[Tuple[np.ndarray]]): Tuple containing the bboxes, score, class_id
        """

        batch_ims, batch_params = self.pre_transform(ims)
        predictions = self.backend(batch_ims)[0]  # yolov8 has only one output
        output = self.post_transform(
            predictions=predictions,
            letterbox_params=batch_params,
        )

        return output

    @NoProfile
    def warmup(
        self,
        iters: int = 10,
    ) -> None:
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
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
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot results from neural network.

        Args:
            image (np.ndarray): image to plot the bboxes.
            bboxes (tuple(np.ndarray, ...)): bboxes which consists of (bboxs, scores, cls_id)
        Returns:
            None
        """

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
                thickness=2,
            )
            cv2.putText(
                img=image,
                text=cls_name,
                org=(xmin, ymin),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
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
