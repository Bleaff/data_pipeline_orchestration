from typing import Any
import logging

from neudc.nn.backends import get_backend
from neudc.nn.models.det.base import BaseDetector
from neudc.nn.models.cls.base import BaseClsModel
from neudc.nn.models.track.basetracker import BaseTracker

from neudc.nn.models.det.yolo import YOLOv8
from neudc.nn.models.cls.blur_model import BlurClassification
from neudc.nn.models.det.sahi import SAHIDetector
from neudc.nn.models.track.deepocsort import DeepOcSort

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[BaseClsModel | BaseDetector | BaseTracker]] = {
    "YOLOv8": YOLOv8,
    "SAHIDetector": SAHIDetector,
    "DeepOcSort": DeepOcSort,
    "BlurClassification": BlurClassification
}

class ModelFactory:
    """Factory to create model instances based on config."""

    @staticmethod
    def create(config: dict[str, Any], device_id: int = 0) -> BaseClsModel | BaseDetector | BaseTracker:
        """
        Create a model instance from its config.

        Args:
            config (dict): Model config with keys "type" and "backend".
            device_id (int): Device id, -1 for CPU.

        Returns:
            BaseClsModel | BaseDetector | BaseTracker: Instantiated model.
        """
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary.")

        config = config.copy()  # Prevent side effects
        model_type = config.pop("type", None)
        backend_name = config.pop("backend", "TorchBackend")

        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_class = MODEL_REGISTRY[model_type]
        backend = get_backend(backend_name)
        config['backend'] = backend
        logger.debug(f"Instantiating model: {model_type} with backend {backend_name}")

        return model_class(
            **config,
        )

__all__ = ["YOLOv8", "SAHIDetector", "DeepOcSort", "BlurClassification", "ModelFactory"]
