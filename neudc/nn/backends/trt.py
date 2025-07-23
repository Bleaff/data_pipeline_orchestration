from __future__ import annotations

import ctypes
import json
from typing import TYPE_CHECKING, Callable

import numpy as np
import tensorrt as trt
from cuda import cuda, cudart

from neudc.nn import BaseBackend
from neudc.utils import LOGGER, PROFILE_FREQ, Profile, TensorRTLogger

if TYPE_CHECKING:
    from neudc.utils.types import FloatFeaturesBatch, FloatImagesBatch

__all__ = ("TensorRTBackend",)


def check_cuda_err(err) -> None:
    if isinstance(err, cuda.CUresult) and err != cuda.CUresult.CUDA_SUCCESS:
        msg = f"Cuda Error: {err}"
        raise RuntimeError(msg)
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            msg = f"Cuda Runtime Error: {err}"
            raise RuntimeError(msg)
    else:
        msg = f"Unknown error type: {err}"
        raise RuntimeError(msg)


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array."""

    def __init__(
        self,
        size: int,
        shape: tuple[int, ...],
        dtype: np.dtype = np.dtype(np.uint8),
    ) -> HostDeviceMem:
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes
        self.shape = shape

    @property
    def host(self) -> np.ndarray:
        return self._host.reshape(self.shape)

    @host.setter
    def host(self, data: np.ndarray | bytes) -> None:
        if isinstance(data, np.ndarray):
            if data.size > self._host.size:
                msg = f"Tried to fit an array of size {data.size} into host memory of size {self._host.size}"
                raise ValueError(
                    msg,
                )
            np.copyto(
                dst=self._host[: data.size],
                src=data.flat,
                casting="safe",
            )
        else:
            assert self._host.dtype == np.uint8
            self._host[: self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self) -> str:
        return f"Host: {self._host}\nDevice: {self.device}\nSize: {self.nbytes}\n"

    def __repr__(self) -> str:
        return self.__str__()

    def free(self) -> None:
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


def do_inference_base(
    inputs: list[HostDeviceMem],
    outputs: list[HostDeviceMem],
    stream: ctypes.c_void_p,
    execute_async_func: Callable,
) -> list[np.ndarray]:
    """Transfer input data to the GPU."""
    # Transfer input data to the GPU.
    [
        cuda_call(
            cudart.cudaMemcpyAsync(
                dst=inp.device,
                src=inp.host,
                count=inp.nbytes,
                kind=cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream=stream,
            ),
        )
        for inp in inputs
    ]

    # Run inference
    execute_async_func()

    # Transfer predictions back from the GPU
    [
        cuda_call(
            cudart.cudaMemcpyAsync(
                dst=out.host,
                src=out.device,
                count=out.nbytes,
                kind=cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream=stream,
            ),
        )
        for out in outputs
    ]

    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))

    # Return only the host outputs
    return [out.host for out in outputs]


def do_inference_old(
    context: trt.IExecutionContext,
    engine: trt.ICudaEngine,
    bindings: list[int],
    inputs: list[HostDeviceMem],
    outputs: list[HostDeviceMem],
    stream: ctypes.c_void_p,
) -> list[np.ndarray]:
    """This function is generalized for multiple inputs/outputs.
    inputs and outputs are expected to be lists of HostDeviceMem objects.
    Warning: only for TensorRT < 10.0.
    """

    def execute_async_func() -> None:
        context.execute_async_v2(
            stream_handle=stream,
            bindings=bindings,
        )

    return do_inference_base(
        inputs=inputs,
        outputs=outputs,
        stream=stream,
        execute_async_func=execute_async_func,
    )


def do_inference_new(
    context: trt.IExecutionContext,
    engine: trt.ICudaEngine,
    bindings: list[int],
    inputs: list[HostDeviceMem],
    outputs: list[HostDeviceMem],
    stream: ctypes.c_void_p,
) -> list[np.ndarray]:
    """This function is generalized for multiple inputs/outputs.
    inputs and outputs are expected to be lists of HostDeviceMem objects.
    Warning: only for TensorRT >= 10.0.
    """

    def execute_async_func() -> None:
        context.execute_async_v3(stream_handle=stream)

    # Setup context tensor address.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

    return do_inference_base(
        inputs=inputs,
        outputs=outputs,
        stream=stream,
        execute_async_func=execute_async_func,
    )


class TensorRTBackend(BaseBackend):

    cuda = True
    cpu = False
    vendor = "nvidia"

    def __init__(
        self,
        path: str,
        device_id: int = 0,
    ) -> TensorRTBackend:
        """Init of TensorRT backend.

        Args:
        ----
            path (str): path of a model.
            device_id (int): device id for the inference. -1 is cpu device.

        """
        logger = TensorRTLogger()
        (status,) = cudart.cudaSetDevice(device_id)
        assert status == cudart.cudaError_t.cudaSuccess, f"CUDA set device to {device_id} errored: {status}."

        trt.init_libnvinfer_plugins(logger, "")

        engine, metadata = None, None
        LOGGER.info(f"Loading {path} for TensorRT inference for device_id {device_id}...")
        with open(path, "rb") as f, trt.Runtime(logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
                metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
            except UnicodeDecodeError:
                f.seek(0)  # engine file may lack embedded metadata
            engine = runtime.deserialize_cuda_engine(f.read())  # read engine
            is_trt10 = not hasattr(engine, "num_bindings")

        # Eval metadata
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names"} and isinstance(v, str):
                    metadata[k] = eval(v)
        else:
            LOGGER.warning(f"WARNING ⚠️ Metadata not found for {path}")

        self.metadata = metadata

        # Model context
        try:
            self.context = engine.create_execution_context()
        except Exception:  # model is None
            LOGGER.error(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")
            raise

        self.inputs, self.outputs, self.bindings, self.stream, self.fp16 = self._allocate_buffers(engine)
        self.engine = engine

        self.do_inference = do_inference_new if is_trt10 else do_inference_old

    @Profile(use_cuda=True, logger=LOGGER, freq=PROFILE_FREQ, name="engine")
    def __call__(
        self,
        input: FloatImagesBatch,
    ) -> list[FloatFeaturesBatch]:
        """Inference on the TensorRT backend.

        Args:
        ----
            input (FloatImagesBatch): Input images.

        Returns:
        -------
            list[FloatFeaturesBatch]: Output features.

        """
        # TODO check input shapes
        self.inputs[0].host = input

        return self.do_inference(
            context=self.context,
            engine=self.engine,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
        )

    @staticmethod
    def _allocate_buffers(
        engine: trt.ICudaEngine,
        profile_idx: int | None = None,
    ) -> tuple[
        list[HostDeviceMem],
        list[HostDeviceMem],
        list[int],
        ctypes.c_void_p,
        bool,
    ]:
        """Allocates buffers for inference."""
        inputs: list[HostDeviceMem] = []
        outputs: list[HostDeviceMem] = []
        bindings: list[int] = []
        stream = cuda_call(cudart.cudaStreamCreate())
        fp16 = False  # default updated below
        is_trt10 = not hasattr(engine, "num_bindings")
        num = range(engine.num_io_tensors) if is_trt10 else range(engine.num_bindings)
        for i in num:
            name = engine.get_tensor_name(i) if is_trt10 else engine.get_binding_name(i)
            trt_type = trt.nptype(engine.get_tensor_dtype(name) if is_trt10 else engine.get_binding_dtype(i))

            # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
            # Pick out the max shape to allocate enough memory for the binding.
            if profile_idx is None:
                shape = engine.get_tensor_shape(name) if is_trt10 else engine.get_binding_shape(i)
            elif is_trt10:
                shape = engine.get_tensor_profile_shape(name, profile_idx)[-1]
            else:
                shape = engine.get_profile_shape(profile_idx, i)[-1]

            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                msg = f"Binding {name} has dynamic shape, but no profile was specified."
                raise ValueError(msg)

            size = trt.volume(shape)
            # Allocate host and device buffers
            try:
                dtype = np.dtype(trt_type)
                bindingMemory = HostDeviceMem(size, shape, dtype)
            except TypeError:  # no numpy support: create a byte array instead (BF16, FP8, INT4)
                size = int(size * trt_type.itemsize)
                bindingMemory = HostDeviceMem(size, shape)

            # Append the device buffer to device bindings.
            bindings.append(int(bindingMemory.device))

            if is_trt10:
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            else:
                is_input = engine.binding_is_input(i)

            # Append to the appropriate list.
            if is_input:
                inputs.append(bindingMemory)
                if trt_type == np.float16:
                    fp16 = True
            else:
                outputs.append(bindingMemory)

        return inputs, outputs, bindings, stream, fp16

    def __del__(self) -> None:
        """Frees the resources allocated in allocate_buffers."""
        buffers = []
        if hasattr(self, "inputs"):
            buffers += self.inputs

        if hasattr(self, "outputs"):
            buffers += self.outputs

        for mem in buffers:
            mem.free()

        if hasattr(self, "stream"):
            cuda_call(cudart.cudaStreamDestroy(self.stream))
