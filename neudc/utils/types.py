"""Shared jaxtyping/NumPy type aliases used across model and pre/post-processing code."""

import numpy as np
from jaxtyping import Bool, Float, UInt8

__all__ = (
    "BoolMask",
    "FloatBBoxesWithCls",
    "FloatFeaturesBatch",
    "FloatImagesBatch",
    "ImageShape",
    "LetterboxParams",
    "UInt8HWC",
)

# Boolean mask
BoolMask = Bool[np.ndarray, "..."]

# Letterbox Params Nx4 or Nx7 (ratio_x ratio_y, pad_x, pad_y, Optional[image_index, origin_x, origin_y])
LetterboxParams = list[Float[np.ndarray, "4"] | Float[np.ndarray, "7"]]

# Image shape
ImageShape = int | tuple[int, int]

# Basic Uint8 image as input to object detector
UInt8HWC = UInt8[np.ndarray, "H W C"]

# Batch of images, stacked after pre-transform
FloatImagesBatch = Float[np.ndarray, "B C H W"]

# Batch of features as output of object detector
FloatFeaturesBatch = Float[np.ndarray, "B ..."]

# Bounding boxes
FloatBBoxesWithCls = Float[np.ndarray, "N 6"]
