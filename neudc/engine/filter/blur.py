import cv2
import numpy as np

from neudc.utils.types import BoolMask, UInt8HWC

from .base import BaseFilter

__all__ = ("BlurFilter",)


class BlurFilter(BaseFilter):
    def __init__(self, threshold: float = 0.01) -> "BlurFilter":
        """
        Args:
            threshold (float): Threshold for the blur filter.
        """
        self.threshold = threshold

    def __call__(self, ims: list[UInt8HWC], loader_ids: list[int]) -> BoolMask:
        """
        Args:
            ims (list[UInt8HWC]): List of images.
            loader_ids (list[int]): List of loader IDs.

        Returns:
            BoolMask: Boolean mask indicating if the image is blurred.
        """
        mask = np.zeros(len(ims), dtype=np.bool)
        for i, im in enumerate(ims):
            # Laplacian variance
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            mask[i] = laplacian_var < self.threshold
        return mask
