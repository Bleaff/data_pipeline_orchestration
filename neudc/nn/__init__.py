"""Model factory: builds concrete detector/classifier model instances from YAML config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neudc.nn.backends import get_backend
from neudc.utils import LOGGER
from neudc.utils.extras import import_module_for

if TYPE_CHECKING:
    from neudc.nn.models.cls.base import BaseClsModel
    from neudc.nn.models.det.base import BaseDetector

# model type -> (module path, class name), imported lazily in `ModelFactory.create` —
# mirrors `NodeFactory` (neudc/core/node/node_factory.py). `neudc.nn` is on the import
# path of every ASR/LLM/TTS backend (a package's __init__.py always runs before its
# submodules), so eagerly importing cv2/torch-based model classes here would force
# those hard CV dependencies onto audio-only/API-only installs that never use them.
MODEL_IMPORTS: dict[str, tuple[str, str]] = {
    "YOLOv8": ("neudc.nn.models.det.yolo", "YOLOv8"),
    "SAHIDetector": ("neudc.nn.models.det.sahi", "SAHIDetector"),
    "BlurClassification": ("neudc.nn.models.cls.blur_model", "BlurClassification"),
    "EmbeddingFilter": ("neudc.nn.models.cls.embedding_model", "EmbeddingFilter"),
}


class ModelFactory:
    """Factory to create model instances based on config."""

    @staticmethod
    def create(
        config: dict[str, Any],
        device_id: int = 0,  # noqa: ARG004 - suspected pre-existing bug, never forwarded into config; flagged not fixed, see report
    ) -> BaseClsModel | BaseDetector:
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

        if model_type not in MODEL_IMPORTS:
            msg = f"Unsupported model type: {model_type}"
            raise ValueError(msg)

        module_path, class_name = MODEL_IMPORTS[model_type]
        model_class = getattr(import_module_for(module_path, f"Model type '{model_type}'"), class_name)
        backend = get_backend(backend_name)
        config["backend"] = backend
        LOGGER.debug(f"Instantiating model: {model_type} with backend {backend_name}")

        return model_class(
            **config,
        )


__all__ = ["MODEL_IMPORTS", "ModelFactory"]
