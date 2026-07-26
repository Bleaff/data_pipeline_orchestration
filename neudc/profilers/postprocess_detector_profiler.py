"""Prometheus-backed profiler for postprocessing steps of detector nodes.

Wraps a node's postprocess method as a decorator, timing each call and recording
per-node frame/box counters and execution-time gauges.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Self

from prometheus_client import Counter, Gauge

from neudc.core.communication.messaging.types import Batch, Frame
from neudc.profilers.base import BaseProfiler
from neudc.utils import LOGGER

if TYPE_CHECKING:
    from collections.abc import Callable


class PostprocessProfiler(BaseProfiler):
    """Times a node's postprocess call and reports frame/box counts to Prometheus."""

    def __init__(self, *, use_cuda: bool = True, use_torch: bool = False, enable_metrics: bool = True) -> None:
        """Initialize the profiler.

        Args:
        ----
            use_cuda (bool): Whether to synchronize with the CUDA device before timing.
            use_torch (bool): Whether to use ``torch.cuda`` for sync instead of the raw CUDA API.
            enable_metrics (bool): Whether to register and update Prometheus metrics.

        """
        self.node_name: str | None = None
        self.use_cuda = use_cuda
        self.use_torch = use_torch
        self.enable_metrics = enable_metrics
        self.start = 0.0
        self.dt = 0.0
        self.cache: dict[str, float] = {"total_frames": 0, "total_boxes": 0, "total_time": 0, "avarage_time": 0.0}

    def _init_metrics(self) -> None:
        """Create the Prometheus Gauge/Counter metrics scoped to the current node name."""
        metric_prefix = f"{self.node_name}".replace(" ", "_")

        # Metric creation scoped with the node label.
        self.exec_time_gauge = Gauge(f"{metric_prefix}_postprocess_exec_time", "Postprocess execution time", ["node"])

        self.total_frames_counter = Counter(
            f"{metric_prefix}_postprocess_frames", "Total postprocessed frames", ["node"]
        )

        self.total_boxes_counter = Counter(
            f"{metric_prefix}_postprocess_boxes", "Total boxes after postprocess", ["node"]
        )

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``func`` so each call is timed and its result recorded as metrics."""

        @wraps(func)
        def wrapper(method_self: Any, result: Any, item: Batch | Frame) -> Any:
            if not self.node_name:
                self.node_name = method_self.id
                self._init_metrics()

            with self:
                func_result = func(method_self, result, item)

            if isinstance(item, Batch):
                self._process_batch_result(result, item)
            elif isinstance(item, Frame):
                self._process_frame_result(result, item)
            return func_result

        return wrapper

    def _process_batch_result(self, result: list[Any], batch: Batch) -> None:
        """Update cached stats and Prometheus counters for a batch postprocess result."""
        total_frames = len(batch.frames)
        total_boxes = sum(len(res) for res in result)

        self.total_frames_counter.labels(node=self.node_name).inc(total_frames)
        self.total_boxes_counter.labels(node=self.node_name).inc(total_boxes)

        self.cache["total_frames"] += total_frames
        self.cache["total_boxes"] += total_boxes
        self.cache["avarage_time"] = self.cache["total_time"] / self.cache["total_frames"]
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                f"[{self.node_name}] Postprocessed {self.cache['total_frames']} frames with "
                f"{self.cache['total_boxes']} boxes. Average execution time: {self.cache['avarage_time']}s"
            )

    def _process_frame_result(self, result: list[Any], _frame: Frame) -> None:
        """Update cached stats and Prometheus counters for a single-frame postprocess result."""
        total_boxes = len(result)
        self.total_boxes_counter.labels(node=self.node_name).inc(total_boxes)
        self.total_frames_counter.labels(node=self.node_name).inc()

        self.cache["total_boxes"] += total_boxes
        self.cache["total_frames"] += 1
        self.cache["avarage_time"] = self.cache["total_time"] / self.cache["total_frames"]
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                f"[{self.node_name}] Postprocessed frame with {self.cache['total_boxes']} boxes. "
                f"Average execution time: {self.cache['avarage_time']}s"
            )

    def __enter__(self) -> Self:
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop timing and record the elapsed time as the exec-time gauge."""
        self.dt = self.time() - self.start
        self.exec_time_gauge.labels(node=self.node_name).set(self.dt)
        self.cache["total_time"] += self.dt

    def time(self) -> float:
        """Return the current wall-clock time.

        CUDA/torch synchronization was intentionally never wired up here (unlike the
        other profilers in this package); this simply mirrors ``time.time()``.
        """
        return time.time()
