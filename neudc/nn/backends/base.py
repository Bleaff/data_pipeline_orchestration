from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from neudc.utils.types import FloatFeaturesBatch, FloatImagesBatch

__all__ = ("BaseBackend", "BackendType")


# Define the allowed backend types
class BackendType(str, Enum):
    TORCH = "TorchBackend"
    TENSORRT = "TensorRTBackend"
    ONNXRUNTIME = "ONNXRuntimeBackend"


class BaseBackend(ABC):
    def __init__(
        self,
        path: str,
        device_id: int = 0,
        *args,
        **kwargs,
    ) -> "BaseBackend":
        """
        Init of BaseBackend backend.

        Args:
            path (str): weights path of a model.
            device_id (int): device id for the inference. -1 is cpu device.
        """
        self.metadata = {}

    @abstractmethod
    def __call__(
        self,
        input: FloatImagesBatch,
    ) -> list[FloatFeaturesBatch]:
        """Call the backend engine."""
        ...

    @abstractmethod
    def __del__(self):
        """Frees the resources allocated in allocate_buffers."""
        ...

    @property
    def get_metadata(self) -> dict[str, Any]:
        return self.metadata
