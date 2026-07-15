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

    cuda = False
    cpu = False
    vendor = "any"

    def __init__(
        self,
        path: str,
        device_id: int = 0,
        extract_embeddings: bool = False,
        embed_layer_idx: int = -2,
        compile: bool = False,
        fp16: int = False,
    ) -> TorchBackend:
        """Initialize the TorchBackend class.

        Args:
        ----
            path (str): The path to the model file.
            device_id (int): The device ID to use for the inference.
            compile (bool): Whether to compile the model.
            fp16 (int): Whether to use half-precision floating point for the inference.

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
            except Exception as e:
                LOGGER.warning(f"WARNING ⚠️ torch.compile failed: {e}, running without compilation")
                compile = False
        self.model = model
        self.fp16 = fp16
        self.compile = compile

    @staticmethod
    def _load_model(path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
        """
        Load model from path, handling both torchscript and regular PyTorch models.

        Args:
            path (str): Path to the model file

        Returns:
            tuple[torch.nn.Module, dict]: Loaded model and metadata
        """
        metadata = {}

        try:
            extra_files = {"config.txt": ""}
            model = torch.jit.load(path, map_location=device, _extra_files=extra_files)

            if extra_files["config.txt"]:
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

            LOGGER.info(f"Loaded TorchScript model from {path}")
            return model, metadata

        except Exception as e:
            LOGGER.info(f"Failed to load as TorchScript: {e}")

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
                        raise ValueError("Model architecture required for state_dict loading")
                    else:
                        model = checkpoint
                else:
                    model = checkpoint

                if not isinstance(model, torch.nn.Module):
                    raise ValueError(f"Loaded object is not a PyTorch model: {type(model)}")

                LOGGER.info(f"Loaded PyTorch model from {path}")
                return model, metadata

            except Exception as e2:
                LOGGER.error(f"Failed to load as PyTorch model", exc_info=e2)

            raise RuntimeError(f"Failed to load model from {path}. Tried TorchScript, PyTorch, and YOLO formats.")

    @Profile(use_cuda=True, use_torch=True, freq=PROFILE_FREQ, name="torch")
    @torch.inference_mode()
    def __call__(
        self,
        input: FloatImagesBatch,
    ) -> list[FloatFeaturesBatch]:
        """
        Call the model with the given input.
        Args:
            input (FloatImagesBatch): The input to the model.
        Returns:
            list[FloatFeaturesBatch]: The output of the model.
        """

        def postprocess_output(obj: Any) -> Any:
            """Recursively put tensors to cpu and fix formats."""

            def process_output(obj):
                """Recursively put tensors to cpu."""
                if torch.is_tensor(obj):
                    return obj.to("cpu", non_blocking=False).numpy()
                elif isinstance(obj, (list, tuple)):
                    return [process_output(item) for item in obj]
                else:
                    return obj

            obj = process_output(obj)

            if not isinstance(obj, list):
                obj = [obj]

            return obj

        if torch.cuda.is_available():
            torch_input = torch.from_numpy(input).pin_memory().to(self.device, non_blocking=True)
        else:
            torch_input = torch.from_numpy(input).to(self.device, non_blocking=True)

        if self.fp16:
            torch_input = torch_input.half()
        else:
            torch_input = torch_input.float()

        torch_output = self.model(torch_input)

        torch_output_post = postprocess_output(torch_output)

        return torch_output_post

    def __del__(self) -> None:
        self.model = None
