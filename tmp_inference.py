import cv2
import numpy as np

from neudc.nn.backends.onnxruntime import ONNXRuntimeBackend
from neudc.utils.types import FloatImagesBatch


def preprocess_input(image_path) -> FloatImagesBatch:
    """Preprocess the input image for ONNX inference.

    Args:
    ----
        image (np.ndarray): Input image to preprocess.

    Returns:
    -------
        FloatImagesBatch: Preprocessed image batch.

    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)  # h w c

    # Example preprocessing: normalize and convert to float
    image = image.astype(np.float32) / 255.0
    np.tramspose(image, (2, 0, 1))
    np.expand_dims(image, axis=0)
    return image  # Wrap in a list to match FloatImagesBatch type


def main() -> None:
    # Example usage of ONNXRuntimeBackend
    backend = ONNXRuntimeBackend(path="models/yolov11.onnx", device_id=-1)

    ready_for_inf = preprocess_input("test_images/1.jpeg")

    # Perform inference
    backend(ready_for_inf)

