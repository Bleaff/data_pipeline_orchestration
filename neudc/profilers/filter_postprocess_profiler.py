"""Module contains a simple decorator for profiling of functions and methods.

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
from typing import TYPE_CHECKING, Any, Self

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

if TYPE_CHECKING:
    from collections.abc import Callable


class FilterPostprocessProfiler(BaseProfiler):
    """Profiler for functions and methods.

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
    ) -> None:
        """Initialize the FilterPostprocessProfiler.

        Args:
        ----
            t (float, optional): Initial accumulated time. Defaults to 0.0.
            use_cuda (bool, optional): Whether to synchronize with CUDA device.
                Defaults to True.
            use_torch (bool, optional): Whether to use torch.cuda for sync if CUDA_PROFILE_ENABLE is False.
                Defaults to False.
            freq (int, optional): Log frequency (calls). None means no logging.
                Defaults to None.
            max_calls (int, optional): Maximum calls before reset. Defaults to 100_000.
            enable_metrics (bool, optional): Whether to register and update Prometheus metrics.
                Defaults to True.

        """
        self.node_name: str | None = None
        self.cache: dict[str, float] = {
            "filtered": 0,
            "go_through": 0,
            "total": 0,
            "total_time": 0,
            "avarage_time": 0,
        }
        self.t = t
        self.use_cuda = use_cuda
        self.use_torch = use_torch
        self.freq = freq
        self.max_calls = max_calls
        self.call_count = 0
        self.start = 0.0
        self.dt = 0.0
        self.enable_metrics = enable_metrics

    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        metric_prefix = f"{self.node_name}".replace(" ", "_")
        self.exec_time_gauge = Gauge(f"{metric_prefix}_execution_time_seconds", "Execution time of the function")
        self.total_frames_counter = Counter(f"{metric_prefix}_frames_total", "Total number of frames")
        self.filtered_frames_counter = Counter(f"{metric_prefix}_frames_filtered", "Number of filtered frames")
        self.go_through_counter = Counter(f"{metric_prefix}_frames_passed", "Number of passed frames")

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``func`` so each call is timed and its result recorded as metrics."""

        @wraps(func)
        def wrapper(method_self: Any, result: Any, item: Batch | Frame) -> Any:
            if self.node_name is None:
                self.node_name = method_self.id
                LOGGER.info(f"Node name is {self.node_name}")
                self._init_metrics()

            with self:
                func_result = func(method_self, result, item)
            # Be careful - in filter and postprocess we have different result - frame output.
            if func_result is None:
                return None
            if isinstance(item, Batch):
                self._process_batch_result(func_result, item)
            elif isinstance(item, Frame):
                self._process_frame_result(func_result, item)
            return func_result

        return wrapper

    def _process_batch_result(self, result: Batch, batch: Batch) -> None:
        """Update cached stats and Prometheus counters for a batch filter/postprocess result."""
        total_in = len(batch.frames)
        self.cache["total"] += total_in
        TOTAL_FRAMES_COUNTER.labels(node=self.node_name).inc(total_in)
        for frame in result.frames:
            if frame.drop:
                self.cache["filtered"] += 1
                FILTERED_FRAMES_COUNTER.labels(node=self.node_name).inc()
            else:
                self.cache["go_through"] += 1
                GO_THROUGH_COUNTER.labels(node=self.node_name).inc()
        self.cache["avarage_time"] = self.cache["total_time"] / self.cache["total"]
        LOGGER.debug(
            f"[{self.node_name}] {self.cache['total']=} frames,  {self.cache['filtered']=} frames, {self.cache['go_through']=}. Average time of execution: {self.cache['avarage_time']:.6f}s"
        )

    def _process_frame_result(self, result: Frame, _frame: Frame) -> None:
        """Update cached stats and Prometheus counters for a single-frame filter/postprocess result."""
        self.cache["total"] += 1
        TOTAL_FRAMES_COUNTER.labels(node=self.node_name).inc()
        if result.drop:
            self.cache["filtered"] += 1
            GO_THROUGH_COUNTER.labels(node=self.node_name).inc()
        else:
            self.cache["go_through"] += 1
            FILTERED_FRAMES_COUNTER.labels(node=self.node_name).inc()

        self.cache["avarage_time"] = self.cache["total_time"] / self.cache["total"]
        LOGGER.debug(
            f"[{self.node_name}] {self.cache['total']=} frames,  {self.cache['filtered']=} frames, {self.cache['go_through']=}. Average time of execution: {self.cache['avarage_time']:.6f}s"
        )

    def __enter__(self) -> Self:
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop timing and, if metrics are enabled, record the elapsed time."""
        if self.start:
            self.dt = self.time() - self.start
            self.t += self.dt
            self.call_count += 1
            self.cache["total_time"] = self.t
            EXEC_TIME_GAUGE.labels(node=self.node_name).set(self.dt)
        LOGGER.debug(f"Elapsed time for '{self.node_name}' is {self.t} s")
        self.cache["total_time"] = self.t

    def __str__(self) -> str:
        """Return a human-readable string of the accumulated elapsed time."""
        return f"Elapsed time for '{self.node_name}' is {self.t} s"

    def time(self) -> float:
        """Get the current time, synchronizing with CUDA if needed."""
        return time.time()
