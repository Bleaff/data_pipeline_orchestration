from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from neudc.nn.models.cls.feature_extractor import FeatureExtractor
from neudc.utils import LOGGER, PROFILE_FREQ, Profile

from .base import BaseClsModel

if TYPE_CHECKING:
    from neudc.nn.backends import BaseBackend
    from neudc.utils.types import (
        UInt8HWC,
    )


class BlurClassification(BaseClsModel):
    """A classification model for detecting blur in images using feature extraction and neural network inference.

    This class processes images by extracting features from regions of interest (ROI) and classifies
    them as blurry or sharp based on a configurable confidence threshold.

    Attributes
    ----------
        backend (BaseBackend): Neural network backend for inference
        conf (float): Confidence threshold for blur detection (0.0 to 1.0)
        path (str): Path to the model file
        device_id (int): Device ID for inference

    """

    def __init__(
        self,
        path: str,
        backend: type[BaseBackend],
        device_id: int = 0,
        conf: float = 0.5,
    ) -> None:
        """Initialize the BlurClassification model.

        Args:
        ----
            path (str): Path to the trained model file
            backend (BaseBackend): Neural network backend instance for model inference
            device_id (int, optional): GPU device ID for inference. Defaults to 0.
            conf (float, optional): Confidence threshold for blur detection.
                                  Values > conf are considered blurry. Defaults to 0.5.
            names (Optional[Union[List[str], Dict[int, str]]], optional):
                Class names mapping. Currently unused but kept for compatibility. Defaults to None.

        Returns:
        -------
            BlurClassification: Initialized blur classification instance

        """
        super().__init__(path=path, backend=backend, device_id=device_id, conf=conf)

        self.backend = backend(path=path, device_id=device_id)

        self.conf = conf
        self.path = path
        self.device_id = device_id

    @Profile(use_cuda=False, logger=LOGGER, freq=PROFILE_FREQ)
    def pre_transform(
        self,
        ims: list[UInt8HWC],
    ) -> list[np.ndarray | None]:
        """Preprocess input images by extracting blur-relevant features.

        This method converts images to grayscale, computes regions of interest (ROI),
        and extracts features using DCT transform coefficients for blur detection.

        Args:
        ----
            ims (List[UInt8HWC]): List of input images in HWC format with uint8 values

        Returns:
        -------
            List[Optional[np.ndarray]]: List of extracted features for each image.
                                      None is returned for images where features couldn't be extracted.
                                      Valid features are normalized float32 arrays in range [0, 1].

        Note:
        ----
            - Creates a new FeatureExtractor instance for each image to avoid state persistence
            - Images are automatically converted to grayscale if they have 3 channels
            - Features are normalized by dividing by 255.0

        """
        batch_feats: list[np.ndarray | None] = []
        for img in ims:
            fe = FeatureExtractor()
            processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            fe.resize_image(processed_img, *processed_img.shape[:2])
            fe.compute_roi()
            feats = fe.extract_feature()
            if not feats:
                batch_feats.append(None)
            else:
                feats = np.stack(feats).astype(np.float32) / 255.0
                batch_feats.append(feats)
        return batch_feats

    @Profile(use_cuda=False, logger=LOGGER, freq=PROFILE_FREQ)
    def post_transform(
        self,
        predictions: list[np.ndarray],
    ) -> list[bool]:
        """Post-process neural network predictions to determine blur classification.

        This method analyzes the model's output predictions and determines whether
        each image region is blurry based on the confidence threshold.

        Args:
        ----
            predictions (List[np.ndarray]): List of prediction arrays from the neural network.
                                          Each array contains class probabilities for image regions.

        Returns:
        -------
            List[bool]: List of boolean values indicating blur detection results.
                       True indicates the image is blurry, False indicates it's sharp.

        Note:
        ----
            - Uses argmax to get the predicted class for each region
            - Assumes class 0 represents "blurry" regions
            - An image is classified as blurry if the ratio of blurry regions exceeds self.conf

        """
        blur_result = []
        for preds in predictions:
            out = preds.argmax(axis=1)
            blur_ratio = (out == 0).mean()
            is_blurry = blur_ratio > self.conf
            blur_result.append(is_blurry)

        return blur_result

    def __call__(
        self,
        ims: list[UInt8HWC],
        return_embeddings: bool = False,
    ) -> list[Any]:
        """Perform blur detection on a batch of images.

        This is the main inference method that orchestrates the complete blur detection pipeline:
        preprocessing, neural network inference, and postprocessing.

        Args:
        ----
            ims (List[UInt8HWC]): List of input images in HWC format with uint8 values

        Returns:
        -------
            List[List[bool]]: Nested list where each inner list contains blur detection results
                             for the corresponding input image. True indicates blur detected,
                             False indicates the image is sharp.

        Note:
        ----
            - Images with no extractable features are automatically classified as blurry ([True])
            - Each image is processed independently to avoid state interference
            - The outer list corresponds to input images, inner lists to classification results

        Example:
        -------
            >>> classifier = BlurClassification("model.onnx", backend, conf=0.6)
            >>> results = classifier([image1, image2])  # [[True], [False]]

        """
        res = []

        batch_fetch = self.pre_transform(ims)
        for fetch in batch_fetch:
            logits = self.backend(fetch)
            final = self.post_transform(logits)
            res.append(final)
        return res

    def __repr__(self) -> str:
        """Return a string representation of the BlurClassification instance.

        Returns
        -------
            str: String representation containing model path, device ID, and confidence threshold

        """
        return f"BlurClassification(path={self.path}, device_id={self.device_id}, conf={self.conf})"
