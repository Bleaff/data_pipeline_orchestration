import cv2
import numpy as np
import os

from neudc.nn.models.det.yolo import YOLOv8
from neudc.nn.backends import ONNXRuntimeBackend
from neudc.utils.types import FloatImagesBatch


def main() -> None:
    # Example usage of ONNXRuntimeBackend
    backend = ONNXRuntimeBackend("./yolov8n.onnx", -1)
    model = YOLOv8(
        path="./yolov8n.onnx",
        backend=backend,
        device_id=-1,
        imgsz=640,
    )

    ready_for_inf = cv2.imread("test_images/1.jpeg") # hwc image
    print(ready_for_inf.shape)
    
    print(model([ready_for_inf]))
    # backend(ready_for_inf)


if __name__ == "__main__":
    print(os.getcwd())
    print(os.listdir())
    main()