"""
Module contains a simple decorator for profiling of functions and methods.

Decorator is generating logging messages with information about function name,
arguments and result of the function call.

Example of usage:

.. code-block:: python

    @DetectorProfiler("my_detector")
    def my_detector(image):
        # do the detection
        pass

    my_detector(image)

"""

from __future__ import annotations

from functools import wraps
from typing import Self

import torch
from prometheus_client import Counter, Gauge

from neudc.profilers.profiler_metrics import (
    EXEC_TIME_GAUGE,
    FILTERED_FRAMES_COUNTER,
    GO_THROUGH_COUNTER,
    TOTAL_FRAMES_COUNTER,
)
from neudc.utils import LOGGER

try:
    import cuda

    CUDA_PROFILE_ENABLE = True
except ImportError:
    CUDA_PROFILE_ENABLE = False
    LOGGER.info("Cuda library is not installed. Check your installation carefully.")


if CUDA_PROFILE_ENABLE:
    from cuda import cuda, cudart

import logging
from functools import wraps
from typing import Self

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class FilterProcessProfiler:
    """
    Profiler for functions and methods.

    This profiler generates logging messages with information about function name,
    arguments and result of the function call.

    It can be used as a decorator:

    .. code-block:: python

        @FilterProcessProfiler("my_detector")
        def my_detector(image):
            # do the detection
            pass

        my_detector(image)

    or as a context manager:


        with FilterProcessProfiler("my_detector") as p:
            my_detector(image)

    """

    def __init__(
        self,
        *,
        node_name=None,
        t: float = 0.0,
        use_cuda: bool = True,
        use_torch: bool = False,
        freq: int | None = None,
        max_calls: int = 100_000,
        enable_metrics: bool = True,
    ):
        """
        Initialize the FilterProcessProfiler.

        Args:
            node_name (str, optional): Node name which will be used in log messages.
                Defaults to None.
            t (float, optional): Initial accumulated time. Defaults to 0.0.
            use_cuda (bool, optional): Whether to synchronize with CUDA device.
                Defaults to True.
            use_torch (bool, optional): Whether to use torch.cuda for sync if CUDA_PROFILE_ENABLE is False.
                Defaults to False.
            freq (int, optional): Log frequency (calls). None means no logging.
                Defaults to None.
            max_calls (int, optional): Maximum calls before reset. Defaults to 100_000.
        """
        self
        self.node_name = node_name
        self.start = None
        self.cache = {"filtered": 0, "go_through": 0, "total": 0, "total_time": 0}
        self.t = t
        self.use_cuda = use_cuda
        self.use_torch = use_torch
        self.freq = freq
        self.max_calls = max_calls
        self.call_count = 0
        self.start = 0.0
        self.dt = 0.0
        self.enable_metrics = True  # new flag
        if self.enable_metrics and self.node_name:
            self._init_metrics()

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        metric_prefix = f"{self.node_name}".replace(" ", "_")
        self.exec_time_gauge = Gauge(f"{metric_prefix}_execution_time_seconds", "Execution time of the function")
        self.total_frames_counter = Counter(f"{metric_prefix}_frames_total", "Total number of frames")
        self.filtered_frames_counter = Counter(f"{metric_prefix}_frames_filtered", "Number of filtered frames")
        self.go_through_counter = Counter(f"{metric_prefix}_frames_passed", "Number of passed frames")

    def __call__(self, func):
        @wraps(func)
        def wrapper(method_self, *args, **kwargs):
            """
            Wraps a function to log its arguments and result.

            Args:
                *args: Function arguments.
                **kwargs: Function keyword arguments.

            Returns:
                The result of the function call.
            """
            if self.node_name is None:
                self.node_name = method_self.name
            inside_argument = args[0] if len(args) else next(iter(kwargs.values()))
            func_result = None
            with self:
                func_result = func(method_self, inside_argument)
            LOGGER.debug(
                f"[{self.node_name}] Function: {func.__name__}, result: {func_result}, argument: {inside_argument}"
            )
            return func_result

        return wrapper


def _process_frame(self, frame_in, frame_out) -> None:
    if frame_out is not None:
        self.cache["go_through"] += 1
        GO_THROUGH_COUNTER.labels(node=self.node_name).inc()
    else:
        self.cache["filtered"] += 1
        FILTERED_FRAMES_COUNTER.labels(node=self.node_name).inc()

    self.cache["total"] += 1
    TOTAL_FRAMES_COUNTER.labels(node=self.node_name).inc()

    def __enter__(self) -> Self:
        """Start timing."""
        self.start = self.time()
        return self


def __exit__(self, *_):
    if self.start:
        self.dt = self.time() - self.start
        self.t += self.dt
        self.call_count += 1
        EXEC_TIME_GAUGE.labels(node=self.node_name).set(self.dt)

    def __str__(self) -> str:
        """Return a human-readable string of the accumulated elapsed time."""
        return f"Elapsed time for '{self.node_name}' is {self.t} s"

    def time(self) -> float:
        """Get the current time, synchronizing with CUDA if needed."""
        if self.use_cuda:
            sync_success = False
            if CUDA_PROFILE_ENABLE:
                (err,) = cudart.cudaDeviceSynchronize()
                sync_success = err == cuda.CUresult.CUDA_SUCCESS
            elif self.use_torch:
                try:
                    torch.cuda.synchronize()
                    sync_success = True
                except (ImportError, AttributeError, RuntimeError) as e:
                    LOGGER.exception("Torch CUDA sync failed.", exc_info=e)

            if not sync_success:
                LOGGER.error("CUDA device synchronization failed.")

        return time.time()
