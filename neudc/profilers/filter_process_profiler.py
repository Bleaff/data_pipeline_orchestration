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

import time
from functools import wraps
from typing import Self

from prometheus_client import Counter, Gauge

from neudc.core.communication.messaging.types import Batch, Frame
from neudc.profilers.base import BaseProfiler
from neudc.profilers.profiler_metrics import (
    EXEC_TIME_GAUGE,
    FILTERED_FRAMES_COUNTER,
    GO_THROUGH_COUNTER,
    TOTAL_FRAMES_COUNTER,
)
from neudc.utils import LOGGER

try:
    pass

    CUDA_PROFILE_ENABLE = True
except ImportError:
    CUDA_PROFILE_ENABLE = False
    LOGGER.info("Cuda library is not installed. Check your installation carefully.")


if CUDA_PROFILE_ENABLE:
    pass

import logging
from functools import wraps
from typing import Self

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class FilterProcessProfiler(BaseProfiler):
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
        self.enable_metrics = enable_metrics

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
            if not hasattr(self, "node_name"):
                self.node_name = method_self.id
                LOGGER.info(f"Node name is {self.node_name}")
                self._init_metrics()
            inside_argument = args[0] if len(args) else next(iter(kwargs.values()))
            func_result = None
            with self:
                func_result = func(method_self, inside_argument)
            self._process_frame(inside_argument, func_result)
            return func_result

        return wrapper

    def _process_frame(self, frame_in, frame_out) -> None:
        """Update statistics for processed frames or batches."""
        if isinstance(frame_out, Frame):
            self.cache["total"] += 1
            TOTAL_FRAMES_COUNTER.labels(node=self.node_name).inc()

            if frame_out.drop:
                self.cache["filtered"] += 1
                GO_THROUGH_COUNTER.labels(node=self.node_name).inc()
            else:
                self.cache["go_through"] += 1
                FILTERED_FRAMES_COUNTER.labels(node=self.node_name).inc()

        elif isinstance(frame_out, Batch):
            total_in = len(frame_in.frames)
            self.cache["total"] += total_in
            TOTAL_FRAMES_COUNTER.labels(node=self.node_name).inc(total_in)
            for frame in frame_out.frames:
                if frame.drop:
                    self.cache["filtered"] += 1
                    FILTERED_FRAMES_COUNTER.labels(node=self.node_name).inc()
                else:
                    self.cache["go_through"] += 1
                    GO_THROUGH_COUNTER.labels(node=self.node_name).inc()
        else:
            LOGGER.warning(f"[{self.node_name}] Unknown frame_out type: {type(frame_out)}")
            return

        LOGGER.debug(f"[{self.node_name}] cache: {self.cache}")

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
            self.cache["total_time"] = self.t
        LOGGER.debug(f"Elapsed time for '{self.node_name}' is {self.t} s")

    def __str__(self) -> str:
        """Return a human-readable string of the accumulated elapsed time."""
        return f"Elapsed time for '{self.node_name}' is {self.t} s"

    def time(self) -> float:
        """Get the current time, synchronizing with CUDA if needed."""
        return time.time()
