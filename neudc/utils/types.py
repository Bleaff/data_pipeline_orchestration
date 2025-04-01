from typing import Union

import numpy as np
from jaxtyping import Bool, Float, UInt8

__all__ = (
    "ImageShape",
    "UInt8HWC",
    "FloatImagesBatch",
    "FloatFeaturesBatch",
    "FloatBBoxesWithCls",
    "LetterboxParams",
    "BoolMask",
)

# Boolean mask
BoolMask = Bool[np.ndarray, "..."]

# Letterbox Params Nx4 or Nx7 (ratio_x ratio_y, pad_x, pad_y, Optional[image_index, origin_x, origin_y])
LetterboxParams = list[Union[Float[np.ndarray, "4"], Float[np.ndarray, "7"]]]

# Image shape
ImageShape = Union[int, tuple[int, int]]

# Basic Uint8 image as input to object detector
UInt8HWC = UInt8[np.ndarray, "H W C"]

# Batch of images, stacked after pre-transform
FloatImagesBatch = Float[np.ndarray, "B C H W"]

# Batch of features as output of object detector
FloatFeaturesBatch = Float[np.ndarray, "B ..."]

# Bounding boxes
FloatBBoxesWithCls = Float[np.ndarray, "N 6"]
