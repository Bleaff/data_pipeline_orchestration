from neudc.core.node.model.mixins.infer_det import InferDetMixin
from neudc.nn import ModelFactory
from neudc.utils import NoProfile, Profile
from typing import Any, List, Union
import numpy as np
import cv2

class TestInferDetMixin(InferDetMixin):
    """Test class for InferDetMixin."""

    def __init__(self, model, *args: Any, **kwargs: Any) -> None:
        """Initialize the TestInferDetMixin."""
        self.model = model
        super().__init__(*args, **kwargs)
        self.profile = Profile()
    
    def __call__(self, *args, **kwds):
        return self.process(*args, **kwds)
    

def main():
    """Main function to test the InferDetMixin."""
    # Example model configuration
    config = {
        "type": "YOLOv8",
        "path": "models/yolov8n.onnx",
        "backend": "ONNXRuntimeBackend",
        "device_id": -1,
        "imgsz": (640, 640),
        "conf": 0.25,
        "iou": 0.45,
        "max_det": 100,
        "nms": True
    }
    
    # Create model instance
    model = ModelFactory.create(config)
    
    # Create an instance of TestInferDetMixin
    infer_mixin = TestInferDetMixin(model)
    
    # Example input images (list of numpy arrays)
    image = cv2.imread("test_images/bbhair.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    # image = np.transpose(image, (2, 0, 1))  # Convert to CHW format if required by the model
    # images = [image]
    # Process images
    expanded_images = np.expand_dims(image, axis=0)  # Add a batch dimension to the image
    results = infer_mixin.process(expanded_images)
    
    print("Detection results:", results)

    
if __name__ == "__main__":
    main()

