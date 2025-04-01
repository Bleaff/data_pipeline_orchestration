import cv2
import numpy as np

from neudc.utils.types import FloatBBoxesWithCls, ImageShape

# from numba import float32, int64, jit, types


__all__ = "letterbox", "postprocess_yolo_outputs"


# @jit(
#     types.Tuple(
#         (
#             types.UniTuple(float32, 2),  # ratio: (float32, float32)
#             types.UniTuple(int64, 2),  # new_unpad: (int64, int64)
#             float32,  # dwf: float32
#             float32,  # dhf: float32
#         )
#     )(
#         types.UniTuple(int64, 2),  # shape: (int64, int64)
#         types.UniTuple(int64, 2),  # new_shape: (int64, int64)
#         types.boolean,  # auto: bool
#         types.boolean,  # scale_fill: bool
#         types.boolean,  # scaleup: bool
#         int64,  # stride: int64
#     ),
#     nopython=True,
#     fastmath=True,
#     inline="always",
# )
def compute_letterbox_params(
    shape: tuple[int, int],
    new_shape: tuple[int, int],
    auto: bool,
    scale_fill: bool,
    scaleup: bool,
    stride: int,
) -> tuple[tuple[float, float], tuple[int, int], float, float]:
    """
    Compute letterbox parameters.

    Args:
        shape (tuple[int, int]): Original shape of the image.
        new_shape (tuple[int, int]): New shape of the image.
        auto (bool): If True, the image will be padded to the nearest multiple of the stride.
        scale_fill (bool): If True, the image will be padded to the new shape.
        scaleup (bool): If True, the image will be scaled up to the new shape.
        stride (int): Stride of the image.

    Returns:
        tuple[tuple[float, float], tuple[int, int], float, float]: Ratio, new unpadded shape, dw, dh.
    """
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    ratio = (r, r)
    if auto:
        dw = dw % stride
        dh = dh % stride
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])
        ratio = r

    dw /= 2
    dh /= 2

    return ratio, new_unpad, dw, dh


def letterbox(
    img: np.ndarray,
    new_shape: ImageShape = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = False,
    stride: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize + pad image (letterbox) for maintaining proportions.

    Args:
        img (np.ndarray): Image to resize.
        new_shape (ImageShape): New shape of the image.
        color (tuple[int, int, int]): Color of the border.
        auto (bool): If True, the image will be padded to the nearest multiple of the stride.
        scale_fill (bool): If True, the image will be padded to the new shape.
        scaleup (bool): If True, the image will be scaled up to the new shape.
        stride (int): Stride of the image.

    Returns:
        tuple[np.ndarray, np.ndarray]: Image, letterbox params.
    """
    shape = img.shape[:2]  # (height, width)

    ratio, new_unpad, dw, dh = compute_letterbox_params(shape, new_shape, auto, scale_fill, scaleup, stride)

    if (shape[1], shape[0]) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, np.array([*ratio, dw, dh], dtype=np.float32)


# @jit(nopython=True, fastmath=True, inline="always")
def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between a box and a list of boxes.

    Args:
        box (np.ndarray): Box.
        boxes (np.ndarray): List of boxes.

    Returns:
        np.ndarray: IoU scores.
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area
    return inter_area / union_area


# @jit(nopython=True, fastmath=True, inline="always")
def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou: float = 0.5,
    max_det: int = 100,
) -> np.ndarray:
    """
    Apply NMS to a list of boxes.

    Args:
        boxes (np.ndarray): List of boxes.
        scores (np.ndarray): List of scores.
        iou (float): IoU threshold.

    Returns:
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


# @jit(
#     float32[:, :](  # all_boxes: 2D float32 array
#         float32[:, :],  # predictions: 3D float32 array
#         float32,  # conf: float32
#         types.UniTuple(float32, 2),  # image_shape: Tuple[int, int]
#         types.UniTuple(float32, 2),  # original_image_shape: Tuple[int, int]
#     ),
#     nopython=True,
#     fastmath=True,
#     inline="always",
# )
def decode_output(
    predictions: np.ndarray,
    conf: float,
    ratio: tuple[float, float],
    pad: tuple[float, float],
) -> np.ndarray:
    """
    Decode output.

    Args:
        predictions (np.ndarray): Predictions.
        conf (float): Confidence threshold.
        ratio (tuple[float, float]): Ratio of the image.
        pad (tuple[float, float]): Padding of the image.

    Returns:
        np.ndarray: Decoded predictions.
    """
    dw, dh = pad
    rw, rh = ratio

    pos = np.where(predictions[4:, :] >= conf)

    all_dets = np.empty((len(pos[1]), 6), dtype=np.float32)

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


# @jit(nopython=True, fastmath=True, inline="always")
def postprocess_yolo_outputs(
    predictions: np.ndarray,
    conf: float = 0.2,
    iou: float = 0.7,
    max_det: int = 100,
    ratio: tuple[float, float] = (1.0, 1.0),
    pad: tuple[float, float] = (0.0, 0.0),
) -> FloatBBoxesWithCls:
    """
    Performs decode + NMS

    Args:
        predictions (np.ndarray): Predictions.
        conf (float): Confidence threshold.
        iou (float): IoU threshold.
        ratio (tuple[float, float]): Ratio of the image.
        pad (tuple[float, float]): Padding of the image.

    Returns:
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


# @jit(
#     nopython=True,
#     fastmath=True,
#     inline="always",
# )
def make_sahi_slices_batch(
    imgs: np.ndarray,
    crop_size: tuple[int, int],
    crop_overlap: tuple[int, int],
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    """
    Slice a batch of images (imgs: (B, C, H, W)) into crops.
    Imgs size is guaranteed to be divisible by crop_size + crop_overlap.

    Args:
        imgs (np.ndarray): Images in (B, C, H, W) format.
        crop_size (tuple[int, int]): Size of the crops.
        crop_overlap (tuple[int, int]): Overlap of the crops.

    Returns:
        crops: list of (C, crop_h, crop_w) array.
        origins: list of (3) array with [image_index, x_origin, y_origin].
    """
    B, _, H, W = imgs.shape
    crop_h, crop_w = crop_size
    overlap_h, overlap_w = crop_overlap
    step_h = crop_h - overlap_h
    step_w = crop_w - overlap_w
    n_y = (H - overlap_h) // step_h
    n_x = (W - overlap_w) // step_w
    crops, origins = [], []
    for b in range(B):
        for i in range(n_y):
            y = i * step_h
            if y > H - crop_h:
                y = H - crop_h
            for j in range(n_x):
                x = j * step_w
                if x > W - crop_w:
                    x = W - crop_w
                crops.append(imgs[b, :, y : y + crop_h, x : x + crop_w])
                origins.append([b, x, y])

    return crops, origins
