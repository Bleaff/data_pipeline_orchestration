"""ONNX Runtime inference backend."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import onnxruntime as ort

from neudc.nn.backends import BaseBackend
from neudc.utils import LOGGER, PROFILE_FREQ, Profile

if TYPE_CHECKING:
    from neudc.utils.types import FloatFeaturesBatch, FloatImagesBatch

__all__ = ("ONNXRuntimeBackend",)


class ONNXRuntimeBackend(BaseBackend):
    """Runs inference through an ONNX Runtime `InferenceSession`, CPU or CUDA."""

    cuda = False
    cpu = False
    vendor = "any"

    def __init__(
        self,
        path: str,
        device_id: int = 0,
    ) -> None:
        """Init of ONNX Runtime backend.

        Args:
        ----
            path (str): Path to ONNX model.
            device_id (int): Device id for the inference. -1 is cpu device.

        """
        providers: list[str | tuple[str, dict]] = ["CPUExecutionProvider"]
        if device_id >= 0:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                LOGGER.warning("WARNING ⚠️ CUDA is not available, switching to cpu.")
                device_id = -1
            else:
                providers = [
                    ("CUDAExecutionProvider", {"device_id": device_id}),
                ]

        ONNXRuntimeBackend.cuda = device_id >= 0
        ONNXRuntimeBackend.cpu = not ONNXRuntimeBackend.cuda

        # Setting optimization options
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")

        # Initializing inference session
        self.model = ort.InferenceSession(path, sess_options=sess_options, providers=providers)

        # Getting input and output names
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [output.name for output in self.model.get_outputs()]

        # Parse metadata values into Python objects
        parsed_metadata = {}
        for key, value_str in self.model.get_modelmeta().custom_metadata_map.items():
            try:
                # Attempt to convert string to Python object
                parsed_value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                # Keep as string if conversion fails
                parsed_value = value_str
            parsed_metadata[key] = parsed_value

        self.metadata = parsed_metadata

        self.dynamic = isinstance(self.model.get_outputs()[0].shape[0], str)
        self.fp16 = "float16" in self.model.get_inputs()[0].type

    @Profile(use_cuda=cuda, use_torch=False, freq=PROFILE_FREQ, name="onnx")
    def __call__(
        self,
        input_data: FloatImagesBatch,
    ) -> list[FloatFeaturesBatch]:
        """Call the backend engine.

        Args:
        ----
            input_data (FloatImagesBatch): The input to the model.

        Returns:
        -------
            list[FloatFeaturesBatch]: The output of the model.

        """
        return self.model.run(
            self.output_names,
            {self.input_name: input_data},
        )
