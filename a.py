import cv2

from neudc.nn.backends import ONNXRuntimeBackend
from neudc.nn.models.det import SAHIDetector, YOLOv8

model = YOLOv8(
    path="/workspace/Basharov/Basharov/neudc/yolov8n.onnx",
    backend=ONNXRuntimeBackend,
    device_id=0,
    conf=0.5,
)
sahi = SAHIDetector(model, 640, 0)
sahi.warmup(iters=100)

im = cv2.imread("/workspace/Basharov/Basharov/neudc/assets/images/IMG_8556.jpeg")
for _i in range(100):
    output = sahi(
        [
            im,
        ],
    )
sahi.plot(im, bboxes=output[0], save_path="/workspace/Basharov/Basharov/neudc/assets/images/im1_saved.jpg")
