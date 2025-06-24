import numpy as np
from numba import njit


@njit(fastmath=True)
def xyxy2xysr(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Converts bounding box coordinates from (x1, y1, x2, y2) format to (x, y, s, r) format.

    Args:
    ----
        bbox (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
    -------
        z (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, s, r) format, where
                                          x, y is the center of the box,
                                          s is the scale (area), and
                                          r is the aspect ratio.

    """
    # Extract coordinates directly
    x1 = x[0]
    y1 = x[1]
    x2 = x[2]
    y2 = x[3]

    w = x2 - x1  # width
    h = y2 - y1  # height

    # Initialize output array (4x1)
    y = np.empty(shape=(4, 1), dtype=x.dtype)

    # Compute center x, center y, scale, and aspect ratio
    y[0, 0] = x1 + w / 2.0  # x center
    y[1, 0] = y1 + h / 2.0  # y center
    y[2, 0] = w * h  # scale (area)
    y[3, 0] = w / (h + eps)  # aspect ratio

    return y
