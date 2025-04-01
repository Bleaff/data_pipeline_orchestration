import json

import torch

from neudc.nn.backends import BaseBackend
from neudc.utils import LOGGER, PROFILE_FREQ, Profile
from neudc.utils.checks import torch_compile
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
        compile: bool = True,
        fp16: int = False,
    ) -> "TorchBackend":
        """
        Initialize the TorchBackend class.

        Args:
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
        extra_files = {"config.txt": ""}  # model metadata
        model = torch.jit.load(path, map_location=self.device, _extra_files=extra_files)
        if extra_files["config.txt"]:  # load metadata dict
            self.metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))
        else:
            self.metadata = {}
        model = model.eval().to(self.device)

        if fp16:
            model = model.half()
        if compile:
            model = torch_compile(model)

        self.model = model
        self.fp16 = fp16

    @Profile(use_cuda=cuda, use_torch=True, logger=LOGGER, freq=PROFILE_FREQ, name="torch")
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
        torch_input = torch.from_numpy(input).pin_memory().to(self.device, non_blocking=True)
        # Run the model on the GPU.
        torch_output = self.model(torch_input)

        # Transfer the output back to CPU using non_blocking transfer and convert to numpy.
        numpy_output = torch_output.to("cpu", non_blocking=False).numpy()

        if not isinstance(numpy_output, list):
            numpy_output = [numpy_output]

        return numpy_output

    def __del__(self) -> None:
        self.model = None
