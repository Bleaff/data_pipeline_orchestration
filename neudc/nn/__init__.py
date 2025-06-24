from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neudc.nn.backends import get_backend
from neudc.nn.models.det.sahi import SAHIDetector
from neudc.nn.models.det.yolo import YOLOv8
from neudc.nn.models.track.deepocsort import DeepOcSort

if TYPE_CHECKING:
    from neudc.nn.models.cls.base import BaseClsModel
    from neudc.nn.models.det.base import BaseDetector
    from neudc.nn.models.track.basetracker import BaseTracker

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[BaseClsModel | BaseDetector | BaseTracker]] = {
    "YOLOv8": YOLOv8,
    "SAHIDetector": SAHIDetector,
    "DeepOcSort": DeepOcSort,
}


class ModelFactory:
    """Factory to create model instances based on config."""

    @staticmethod
    def create(config: dict[str, Any], device_id: int = 0) -> BaseClsModel | BaseDetector | BaseTracker:
        """Create a model instance from its config.

        Args:
        ----
            config (dict): Model config with keys "type" and "backend".
            device_id (int): Device id, -1 for CPU.

        Returns:
        -------
            BaseClsModel | BaseDetector | BaseTracker: Instantiated model.

        """
        if not isinstance(config, dict):
            msg = "Config must be a dictionary."
            raise TypeError(msg)

        config = config.copy()  # Prevent side effects
        model_type = config.pop("type", None)
        backend_name = config.pop("backend", "TorchBackend")

        if model_type not in MODEL_REGISTRY:
            msg = f"Unsupported model type: {model_type}"
            raise ValueError(msg)

        model_class = MODEL_REGISTRY[model_type]
        backend = get_backend(backend_name)
        config["backend"] = backend
        logger.debug(f"Instantiating model: {model_type} with backend {backend_name}")

        return model_class(
            **config,
        )


__all__ = ["YOLOv8", "SAHIDetector", "DeepOcSort", "ModelFactory"]
