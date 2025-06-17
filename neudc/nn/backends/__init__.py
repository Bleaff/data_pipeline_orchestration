from neudc.utils import CUDA_PROFILE_ENABLE, ONNX_PROFILE_ENABLE, TENSORRT_ENABLE

from .base import BackendType, BaseBackend
from .torch import TorchBackend

__all__ = ["BaseBackend", "BackendType", "TorchBackend"]

if CUDA_PROFILE_ENABLE and TENSORRT_ENABLE:
    from .trt import TensorRTBackend

    __all__.append("TensorRTBackend")
if ONNX_PROFILE_ENABLE:
    from .onnxruntime import ONNXRuntimeBackend

    __all__.append("ONNXRuntimeBackend")


# Factory function to get the appropriate backend
def get_backend(backend_type: BackendType) -> BaseBackend:
    if backend_type == BackendType.TORCH:
        return TorchBackend
    elif backend_type == BackendType.TENSORRT:
        if CUDA_PROFILE_ENABLE and TENSORRT_ENABLE:
            return TensorRTBackend
        else:
            raise RuntimeError(
                "TensorRTBackend is not available. Ensure CUDA_PROFILE_ENABLE and TENSORRT_ENABLE are True."
            )
    elif backend_type == BackendType.ONNXRUNTIME:
        if ONNX_PROFILE_ENABLE:
            return ONNXRuntimeBackend
        else:
            raise RuntimeError("ONNXRuntimeBackend is not available. Ensure ONNX_PROFILE_ENABLE is True.")
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
