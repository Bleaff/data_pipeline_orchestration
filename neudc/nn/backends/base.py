"""Backend abstraction: the common interface pluggable inference engines implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neudc.utils.types import FloatFeaturesBatch, FloatImagesBatch

__all__ = ("BackendType", "BaseBackend")


# Define the allowed backend types
class BackendType(StrEnum):
    """Identifiers for the inference backends supported by `get_backend`."""

    TORCH = "TorchBackend"
    TENSORRT = "TensorRTBackend"
    ONNXRUNTIME = "ONNXRuntimeBackend"


class BaseBackend(ABC):
    """Common interface every concrete inference backend (torch/onnxruntime/trt) must implement."""

    metadata: dict[str, Any]
    fp16: bool

    @abstractmethod
    def __init__(
        self,
        path: str,
        device_id: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Init of BaseBackend backend.

        Args:
        ----
            path (str): weights path of a model.
            device_id (int): device id for the inference. -1 is cpu device.
            *args (Any): backend-specific positional arguments.
            **kwargs (Any): backend-specific keyword arguments.

        """

    @abstractmethod
    def __call__(
        self,
        input: FloatImagesBatch,
    ) -> list[FloatFeaturesBatch]:
        """Call the backend engine."""

    @property
    def get_metadata(self) -> dict[str, Any]:
        """Return backend metadata (e.g. input size, class names) collected at load time."""
        return self.metadata
