"""PyTorch inference backend (TorchScript and regular `nn.Module` checkpoints)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import torch

from neudc.nn.backends import BaseBackend
from neudc.utils import LOGGER, PROFILE_FREQ, Profile
from neudc.utils.checks import torch_compile

if TYPE_CHECKING:
    from neudc.utils.types import FloatFeaturesBatch, FloatImagesBatch

__all__ = ("TorchBackend",)


class TorchBackend(BaseBackend):
    """Runs inference through a loaded PyTorch (or TorchScript) `nn.Module`."""

    cuda = False
    cpu = False
    vendor = "any"

    def __init__(  # noqa: PLR0917 - backend config constructor, one flag per tunable
        self,
        path: str,
        device_id: int = 0,
        extract_embeddings: bool = False,  # noqa: FBT001, FBT002 - established ML-config convention
        embed_layer_idx: int = -2,
        compile: bool = False,  # noqa: FBT001, FBT002 - established ML-config convention
        fp16: bool = False,  # noqa: FBT001, FBT002 - established ML-config convention
    ) -> None:
        """Initialize the TorchBackend class.

        Args:
        ----
            path (str): The path to the model file.
            device_id (int): The device ID to use for the inference.
            extract_embeddings (bool): Whether to also return intermediate-layer embeddings.
            embed_layer_idx (int): Index of the layer to extract embeddings from.
            compile (bool): Whether to compile the model.
            fp16 (bool): Whether to use half-precision floating point for the inference.

        """
        if device_id >= 0 and not torch.cuda.is_available():
            LOGGER.warning("WARNING ⚠️ CUDA is not available, switching to cpu.")
            device_id = -1

        TorchBackend.cuda = device_id >= 0
        TorchBackend.cpu = not TorchBackend.cuda

        self.device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")
        self.extract_embeddings = extract_embeddings
        self.embed_layer_idx = embed_layer_idx
        model, metadata = self._load_model(path, self.device)

        self.metadata = metadata
        model = model.eval().to(self.device)
        fp16 = fp16 or (next(model.parameters()).dtype == torch.float16)

        if fp16:
            model = model.half()

        if compile:
            try:
                model = torch_compile(model, dynamic=True)
                LOGGER.info("Model compiled successfully")
            except Exception as e:  # noqa: BLE001 - torch.compile may raise varied errors; must not abort load
                LOGGER.warning(f"WARNING ⚠️ torch.compile failed: {e}, running without compilation")
                compile = False  # noqa: A001 - reassigns the `compile` param; renaming would break the config kwarg
        self.model: torch.nn.Module | None = model
        self.fp16 = fp16
        self.compile = compile
        # Reused pinned host buffer for CUDA input staging (see _stage_input).
        self._pinned_staging: torch.Tensor | None = None

    # Tries TorchScript then plain-PyTorch/YOLO fallback loading; splitting hurts readability more than it helps.
    @staticmethod
    def _load_model(  # noqa: C901, PLR0912
        path: str,
        device: torch.device,
    ) -> tuple[torch.nn.Module, dict]:
        """Load model from path, handling both torchscript and regular PyTorch models.

        Args:
        ----
            path (str): Path to the model file.
            device (torch.device): Device to map the loaded model to.

        Returns:
        -------
            tuple[torch.nn.Module, dict]: Loaded model and metadata.

        """
        metadata = {}

        try:
            extra_files = {"config.txt": ""}
            model = torch.jit.load(path, map_location=device, _extra_files=extra_files)
        except Exception as e:  # noqa: BLE001 - torch.jit.load raises varied types; falls back to plain PyTorch below
            LOGGER.info(f"Failed to load as TorchScript: {e}")
        else:
            if extra_files["config.txt"]:
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))
            LOGGER.info(f"Loaded TorchScript model from {path}")
            return model, metadata

        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    model = checkpoint["model"]
                    for key in ["metadata", "config", "meta"]:
                        if key in checkpoint:
                            metadata = checkpoint[key]
                            break
                    if "train_args" in checkpoint:
                        metadata = checkpoint["train_args"]
                elif "state_dict" in checkpoint:
                    msg = "Model architecture required for state_dict loading"
                    raise TypeError(msg)  # noqa: TRY301 - raise-then-catch routes through the shared log/wrap below
                else:
                    model = checkpoint
            else:
                model = checkpoint

            if not isinstance(model, torch.nn.Module):
                msg = f"Loaded object is not a PyTorch model: {type(model)}"
                raise TypeError(msg)  # noqa: TRY301 - raise-then-catch routes through the shared log/wrap below
        except Exception as e2:
            LOGGER.error("Failed to load as PyTorch model", exc_info=e2)
            msg = f"Failed to load model from {path}. Tried TorchScript, PyTorch, and YOLO formats."
            raise RuntimeError(msg) from e2
        else:
            LOGGER.info(f"Loaded PyTorch model from {path}")
            return model, metadata

    def _needs_new_staging(self, tensor: torch.Tensor) -> bool:
        """Whether the cached pinned buffer must be (re)allocated for this input."""
        buf = self._pinned_staging
        return buf is None or tuple(buf.shape) != tuple(tensor.shape) or buf.dtype != tensor.dtype

    def _stage_input(self, input_np: FloatImagesBatch) -> torch.Tensor:
        """Move a numpy batch to the model device.

        On CUDA, copy through a pinned host buffer that is reused across calls of the
        same shape/dtype (steady-state batching), avoiding a fresh pinned allocation
        every call, then transfer asynchronously. On CPU this is a plain transfer.
        """
        tensor = torch.from_numpy(input_np)
        if self.device.type != "cuda":
            return tensor.to(self.device, non_blocking=True)

        if self._needs_new_staging(tensor):
            self._pinned_staging = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, pin_memory=True)
        assert self._pinned_staging is not None
        self._pinned_staging.copy_(tensor)
        return self._pinned_staging.to(self.device, non_blocking=True)

    @Profile(use_cuda=True, use_torch=True, freq=PROFILE_FREQ, name="torch")
    @torch.inference_mode()
    def __call__(
        self,
        input: FloatImagesBatch,
    ) -> list[FloatFeaturesBatch]:
        """Call the model with the given input.

        Args:
        ----
            input (FloatImagesBatch): The input to the model.

        Returns:
        -------
            list[FloatFeaturesBatch]: The output of the model.

        """

        def postprocess_output(obj: Any) -> Any:
            """Recursively put tensors to cpu and fix formats."""

            def process_output(obj: Any) -> Any:
                """Recursively put tensors to cpu."""
                if torch.is_tensor(obj):
                    return obj.to("cpu", non_blocking=False).numpy()
                if isinstance(obj, (list, tuple)):
                    return [process_output(item) for item in obj]
                return obj

            obj = process_output(obj)

            if not isinstance(obj, list):
                obj = [obj]

            return obj

        torch_input = self._stage_input(input)

        torch_input = torch_input.half() if self.fp16 else torch_input.float()

        assert self.model is not None
        torch_output = self.model(torch_input)

        return postprocess_output(torch_output)

    def __del__(self) -> None:
        """Release the reference to the underlying model so CUDA memory can be freed promptly."""
        self.model = None
