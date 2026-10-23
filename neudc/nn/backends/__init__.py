"""Pluggable inference backends (torch, TensorRT, ONNX Runtime) and their factory."""

from neudc.utils import CUDA_PROFILE_ENABLE, ONNX_PROFILE_ENABLE, TENSORRT_ENABLE

from .base import BackendType, BaseBackend

__all__ = ["BackendType", "BaseBackend"]

# torch is part of the optional `cv` extra (see pyproject.toml), not a hard dependency —
# this package is imported transitively by every ASR/LLM/TTS backend module (Python
# always runs a package's __init__.py before its submodules), including the
# audio-only ones that never need torch at all.
try:
    from .torch import TorchBackend

    TORCH_AVAILABLE = True
    __all__ += ["TorchBackend"]
except ImportError:
    TORCH_AVAILABLE = False

if CUDA_PROFILE_ENABLE and TENSORRT_ENABLE:
    from .trt import TensorRTBackend

    __all__ += ["TensorRTBackend"]
if ONNX_PROFILE_ENABLE:
    from .onnxruntime import ONNXRuntimeBackend

    __all__ += ["ONNXRuntimeBackend"]


# Factory function to get the appropriate backend
def get_backend(backend_type: BackendType) -> type[BaseBackend]:
    if backend_type == BackendType.TORCH:
        if TORCH_AVAILABLE:
            return TorchBackend
        msg = "TorchBackend is not available. Install the `cv` extra (torch) to use it."
        raise RuntimeError(msg)
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
