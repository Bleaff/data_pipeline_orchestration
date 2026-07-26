"""Pluggable inference backends (torch, TensorRT, ONNX Runtime) and their factory."""

from neudc.utils import CUDA_PROFILE_ENABLE, ONNX_PROFILE_ENABLE, TENSORRT_ENABLE

from .base import BackendType, BaseBackend
from .torch import TorchBackend

__all__ = ["BackendType", "BaseBackend", "TorchBackend"]

if CUDA_PROFILE_ENABLE and TENSORRT_ENABLE:
    from .trt import TensorRTBackend

    __all__ += ["TensorRTBackend"]
if ONNX_PROFILE_ENABLE:
    from .onnxruntime import ONNXRuntimeBackend

    __all__ += ["ONNXRuntimeBackend"]


# Factory function to get the appropriate backend
def get_backend(backend_type: BackendType) -> type[BaseBackend]:
    if backend_type == BackendType.TORCH:
        return TorchBackend
    if backend_type == BackendType.TENSORRT:
        if CUDA_PROFILE_ENABLE and TENSORRT_ENABLE:
            return TensorRTBackend
        msg = "TensorRTBackend is not available. Ensure CUDA_PROFILE_ENABLE and TENSORRT_ENABLE are True."
        raise RuntimeError(
            msg,
        )
    if backend_type == BackendType.ONNXRUNTIME:
        if ONNX_PROFILE_ENABLE:
            return ONNXRuntimeBackend
        msg = "ONNXRuntimeBackend is not available. Ensure ONNX_PROFILE_ENABLE is True."
        raise RuntimeError(msg)
    msg = f"Unsupported backend type: {backend_type}"
    raise ValueError(msg)
