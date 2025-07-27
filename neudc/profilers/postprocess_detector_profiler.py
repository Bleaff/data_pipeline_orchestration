from __future__ import annotations

import time
from functools import wraps

from prometheus_client import Counter, Gauge

from neudc.core.communication.messaging.types import Batch, Frame
from neudc.profilers.base import BaseProfiler
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

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class PostprocessProfiler(BaseProfiler):
    def __init__(self, *, use_cuda=True, use_torch=False, enable_metrics=True):
        self.node_name = None
        self.use_cuda = use_cuda
        self.use_torch = use_torch
        self.enable_metrics = enable_metrics
        self.start = 0.0
        self.dt = 0.0
        self.cache = {"total_frames": 0, "total_boxes": 0, "total_time": 0, "avarage_time": 0.0}

    def _init_metrics(self):
        metric_prefix = f"{self.node_name}".replace(" ", "_")

        # Правильное создание с label'ом node
        self.exec_time_gauge = Gauge(f"{metric_prefix}_postprocess_exec_time", "Postprocess execution time", ["node"])

        self.total_frames_counter = Counter(
            f"{metric_prefix}_postprocess_frames", "Total postprocessed frames", ["node"]
        )

        self.total_boxes_counter = Counter(
            f"{metric_prefix}_postprocess_boxes", "Total boxes after postprocess", ["node"]
        )

    def __call__(self, func):
        @wraps(func)
        def wrapper(method_self, result, item):
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

    def _process_batch_result(self, result, batch: Batch):
        total_frames = len(batch.frames)
        total_boxes = sum(len(res) for res in result)

        self.total_frames_counter.labels(node=self.node_name).inc(total_frames)
        self.total_boxes_counter.labels(node=self.node_name).inc(total_boxes)

        self.cache["total_frames"] += total_frames
        self.cache["total_boxes"] += total_boxes
        self.cache["avarage_time"] = self.cache["total_time"] / self.cache["total_frames"]
        LOGGER.debug(
            f"[{self.node_name}] Postprocessed {self.cache['total_frames']} frames with {self.cache['total_boxes']} boxes. Average time of execution: {self.cache['avarage_time']}s"
        )

    def _process_frame_result(self, result, frame: Frame):
        total_boxes = len(result)
        self.total_boxes_counter.labels(node=self.node_name).inc(total_boxes)
        self.total_frames_counter.labels(node=self.node_name).inc()

        self.cache["total_boxes"] += total_boxes
        self.cache["total_frames"] += 1
        self.cache["avarage_time"] = self.cache["total_time"] / self.cache["total_frames"]
        LOGGER.debug(
            f"[{self.node_name}] Postprocessed frame with {self.cache['total_boxes']} boxes. Average time of execution: {self.cache['avarage_time']}s"
        )

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, *_):
        self.dt = self.time() - self.start
        self.exec_time_gauge.labels(node=self.node_name).set(self.dt)
        self.cache["total_time"] += self.dt

    def time(self) -> float:
        # if self.use_cuda:
        #     sync_success = False
        #     if CUDA_PROFILE_ENABLE:
        #         (err,) = cudart.cudaDeviceSynchronize()
        #         sync_success = err == cuda.CUresult.CUDA_SUCCESS
        #     elif self.use_torch:
        #         try:
        #             torch.cuda.synchronize()
        #             sync_success = True
        #         except Exception as e:
        #             LOGGER.exception("Torch CUDA sync failed", exc_info=e)
        #     if not sync_success:
        #         LOGGER.error("CUDA device synchronization failed.")
        return time.time()
