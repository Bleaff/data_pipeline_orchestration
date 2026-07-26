"""Numba-accelerated tensor/geometry ops shared by detector models: letterboxing, NMS, SAHI slicing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from numba import float32, int64, jit, types

from neudc.utils import USE_NUMBA, conditional_jit

if TYPE_CHECKING:
    from neudc.utils.types import FloatBBoxesWithCls, ImageShape

__all__ = ("calculate_slices_coordinates", "letterbox", "postprocess_yolo_outputs")


@conditional_jit(
    types.Tuple(
        (
            int64[:, :],  # x: 2D int64 array
            int64[:, :],  # y: 2D int64 array
        ),
    )(
        int64[:],  # x: 1D int64 array
        int64[:],  # y: 1D int64 array
    ),
    nopython=True,
    fastmath=True,
    parallel=False,
    inline="always",
    turn_on=USE_NUMBA,
)
def meshgrid2d_ij(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numba-compiled meshgrid2d."""
    shape = (x.size, y.size)  # Matrix dimensions: rows=x, columns=y
    xx = np.empty(shape, dtype=x.dtype)
    yy = np.empty(shape, dtype=y.dtype)

    # Broadcast x along columns (axis=1)
    xx[...] = x[:, np.newaxis]  # Reshape x to (x.size, 1)

    # Broadcast y along rows (axis=0)
    yy[...] = y[np.newaxis, :]  # Reshape y to (1, y.size)

    return xx, yy


@conditional_jit(
    nopython=True,
    fastmath=True,
    parallel=False,
    inline="always",
    turn_on=USE_NUMBA,
)
def calculate_slices_coordinates(
    imgsz: tuple[int, int],
    crop_size: tuple[int, int],
    crop_overlap: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adjust the image size to be divisible by crop size and overlap.

    Image size should be more than crop_overlap!.

    Args:
    ----
        imgsz (tuple[int, int]): Image size - (height, width).
        crop_size (tuple[int, int]): Crop size - (height, width).
        crop_overlap (tuple[int, int]): Overlap size - (height, width).

    Returns:
    -------
        (tuple[int, int]): Adjusted image size.

    """
    img_h, img_w = imgsz
    crop_h, crop_w = crop_size
    overlap_h, overlap_w = crop_overlap

    step_h = crop_h - overlap_h
    step_w = crop_w - overlap_w

    # Calculate number of steps using integer arithmetic
    n_steps_h = 1 if img_h <= crop_h else (img_h - crop_h + step_h - 1) // step_h + 1
    n_steps_w = 1 if img_w <= crop_w else (img_w - crop_w + step_w - 1) // step_w + 1

    # Generate starting positions
    y1_base = np.arange(0, n_steps_h) * step_h
    x1_base = np.arange(0, n_steps_w) * step_w

    # Adjust last row/column
    if img_h > crop_h and n_steps_h > 1:
        y1_base[-1] = img_h - crop_h
    if img_w > crop_w and n_steps_w > 1:
        x1_base[-1] = img_w - crop_w

    # Create meshgrid and flatten
    y1_grid, x1_grid = meshgrid2d_ij(y1_base, x1_base)
    y1_flat, x1_flat = y1_grid.flatten(), x1_grid.flatten()

    # Calculate end positions
    y2_flat = np.minimum(y1_flat + crop_h, img_h)
    x2_flat = np.minimum(x1_flat + crop_w, img_w)

    return y1_flat, x1_flat, y2_flat, x2_flat


@conditional_jit(
    types.Tuple(
        (
            types.UniTuple(float32, 2),  # ratio: (float32, float32)
            types.UniTuple(int64, 2),  # new_unpad: (int64, int64)
            float32,  # dwf: float32
            float32,  # dhf: float32
        ),
    )(
        types.UniTuple(int64, 2),  # shape: (int64, int64)
        types.UniTuple(int64, 2),  # new_shape: (int64, int64)
        types.boolean,  # auto: bool
        types.boolean,  # scale_fill: bool
        types.boolean,  # scaleup: bool
        int64,  # stride: int64
    ),
    nopython=True,
    fastmath=True,
    inline="always",
    turn_on=USE_NUMBA,
)
def compute_letterbox_params(  # noqa: PLR0917 - positional order is fixed by the numba signature above
    shape: tuple[int, int],
    new_shape: tuple[int, int],
    auto: bool,  # noqa: FBT001 - positional order/types must match the numba signature declared above
    scale_fill: bool,  # noqa: FBT001
    scaleup: bool,  # noqa: FBT001
    stride: int = 32,
) -> tuple[tuple[float, float], tuple[int, int], float, float]:
    """Compute letterbox parameters.

    Args:
    ----
        shape (tuple[int, int]): Original shape of the image.
        new_shape (tuple[int, int]): New shape of the image.
        auto (bool): If True, the image will be padded to the nearest multiple of the stride.
        scale_fill (bool): If True, the image will be padded to the new shape.
        scaleup (bool): If True, the image will be scaled up to the new shape.
        stride (int): Stride of the image.

    Returns:
    -------
        tuple[tuple[float, float], tuple[int, int], float, float]: Ratio, new unpadded shape, dw, dh.

    """
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (round(shape[0] * r), round(shape[1] * r))

    dw: float = new_shape[1] - new_unpad[1]
    dh: float = new_shape[0] - new_unpad[0]

    ratio = (r, r)
    if auto:
        dw = dw % stride
        dh = dh % stride
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = (new_shape[0] / shape[0], new_shape[1] / shape[1])

    dw /= 2
    dh /= 2

    return ratio, new_unpad, dh, dw


def letterbox(  # noqa: PLR0917 - mirrors compute_letterbox_params' parameter list
    img: np.ndarray,
    new_shape: ImageShape = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,  # noqa: FBT001, FBT002 - established ML-config convention
    scale_fill: bool = False,  # noqa: FBT001, FBT002 - established ML-config convention
    scaleup: bool = False,  # noqa: FBT001, FBT002 - established ML-config convention
    stride: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize + pad image (letterbox) for maintaining proportions.

    Args:
    ----
        img (np.ndarray): Image to resize.
        new_shape (ImageShape): New shape of the image.
        color (tuple[int, int, int]): Color of the border.
        auto (bool): If True, the image will be padded to the nearest multiple of the stride.
        scale_fill (bool): If True, the image will be padded to the new shape.
        scaleup (bool): If True, the image will be scaled up to the new shape.
        stride (int): Stride of the image.

    Returns:
    -------
        tuple[np.ndarray, np.ndarray]: Image, letterbox params.

    """
    shape = img.shape[:2]  # (height, width)

    ratio, new_unpad, dh, dw = compute_letterbox_params(
        shape=shape,
        new_shape=new_shape,
        auto=auto,
        scale_fill=scale_fill,
        scaleup=scaleup,
        stride=stride,
    )

    if shape != new_unpad:
        img = cv2.resize(
            src=img,
            dsize=(new_unpad[1], new_unpad[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    img = cv2.copyMakeBorder(
        src=img,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    return img, np.array([*ratio, dh, dw], dtype=np.float32)


@conditional_jit(nopython=True, fastmath=True, inline="always", turn_on=USE_NUMBA)
def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between a box and a list of boxes.

    Args:
    ----
        box (np.ndarray): Box.
        boxes (np.ndarray): List of boxes.

    Returns:
    -------
        np.ndarray: IoU scores.

    """
    box = box.astype(np.float32)
    boxes = boxes.astype(np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area
    return inter_area / union_area


@conditional_jit(nopython=True, fastmath=True, inline="always", turn_on=USE_NUMBA)
def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou: float = 0.5,
    max_det: int = 100,
) -> np.ndarray:
    """Apply NMS to a list of boxes.

    Args:
    ----
        boxes (np.ndarray): List of boxes.
        scores (np.ndarray): List of scores.
        iou (float): IoU threshold.
        max_det (int): Maximum number of detections to keep.

    Returns:
    -------
        np.ndarray: Indices of the boxes to keep.

    """
    indices = scores.argsort()[::-1]
    keep = []

    while indices.size:
        current = indices[0]
        keep.append(current)
        if indices.size == 1:
            break

        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        iou_bbox = compute_iou(current_box, other_boxes)

        indices = indices[1:][iou_bbox < iou]

        # Optional: break early if we've reached max_det
        if len(keep) >= max_det:
            break

    return np.array(keep, dtype=np.int64)


@conditional_jit(
    float32[:, :](  # all_boxes: 2D float32 array
        float32[:, :],  # predictions: 3D float32 array
        float32,  # conf: float32
        types.UniTuple(float32, 2),  # image_shape: Tuple[int, int]
        types.UniTuple(float32, 2),  # original_image_shape: Tuple[int, int]
    ),
    nopython=True,
    fastmath=True,
    inline="always",
    turn_on=USE_NUMBA,
)
def decode_output(
    predictions: np.ndarray,
    conf: float,
    ratio: tuple[float, float],
    pad: tuple[float, float],
) -> np.ndarray:
    """Decode output.

    Args:
    ----
        predictions (np.ndarray): Predictions.
        conf (float): Confidence threshold.
        ratio (tuple[float, float]): Ratio of the image.
        pad (tuple[float, float]): Padding of the image.

    Returns:
    -------
        np.ndarray: Decoded predictions.

    """
    dw, dh = pad
    rw, rh = ratio

    pos = np.where(predictions[4:, :] >= conf)

    all_dets = np.empty((len(pos[1]), 6), dtype=predictions.dtype)

    for j in range(len(pos[0])):
        px = pos[0][j]
        py = pos[1][j]
        bx = predictions[0, py]
        by = predictions[1, py]
        bw = predictions[2, py] / 2
        bh = predictions[3, py] / 2
        all_dets[j][0] = (bx - bw - dw) / rw
        all_dets[j][1] = (by - bh - dh) / rh
        all_dets[j][2] = (bx + bw - dw) / rw
        all_dets[j][3] = (by + bh - dh) / rh
        all_dets[j][4] = predictions[4 + px, py]
        all_dets[j][5] = px

    return all_dets


@jit(nopython=True, fastmath=True, inline="always")
def postprocess_yolo_outputs(  # noqa: PLR0917 - one parameter per decode/NMS tunable, matches decode_output + apply_nms
    predictions: np.ndarray,
    conf: float = 0.2,
    iou: float = 0.7,
    max_det: int = 100,
    ratio: tuple[float, float] = (1.0, 1.0),
    pad: tuple[float, float] = (0.0, 0.0),
) -> FloatBBoxesWithCls:
    """Decode raw YOLO outputs and apply NMS.

    Args:
    ----
        predictions (np.ndarray): Predictions.
        conf (float): Confidence threshold.
        iou (float): IoU threshold.
        max_det (int): Maximum number of detections to keep.
        ratio (tuple[float, float]): Ratio of the image.
        pad (tuple[float, float]): Padding of the image.

    Returns:
    -------
        np.ndarray: Decoded and NMSed predictions with xyxy format.

    """
    # Decode
    all_dets = decode_output(
        predictions=predictions,
        conf=conf,
        ratio=ratio,
        pad=pad,
    )

    # NMS
    keep_indices = apply_nms(
        boxes=all_dets[..., :4],
        scores=all_dets[..., 4],
        iou=iou,
        max_det=max_det,
    )

    return all_dets[keep_indices]
