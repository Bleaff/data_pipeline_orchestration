"""Mixin implementing mailbox-to-batch-queue collection, decoupled from any specific node base."""

from __future__ import annotations

import multiprocessing
import queue
import time
from queue import Full, Queue
from typing import Any

from neudc.core.communication.messaging.types import Batch, Frame


class CollectBatchMixin:
    """Mixin class for collecting data from mailbox."""

    mailbox: Any  # provided by a sibling class in the MRO (e.g. a node base class)

    def __init__(
        self,
        batch_size: int,
        batch_queue_size: int = 20,
        batch_collect_timeout: float = 0.1,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        """Initialize the batch-collection state.

        Args:
        ----
            batch_size (int): Target number of frames per batch.
            batch_queue_size (int): Maximum number of pending batches queued for processing.
            batch_collect_timeout (float): Max seconds to wait while filling a single batch.
            *_args (Any): Unused; accepted for cooperative-constructor call compatibility.
            **_kwargs (Any): Unused; accepted for cooperative-constructor call compatibility.

        """
        self.batch_size = batch_size
        self.batch_collect_timeout = batch_collect_timeout  # max wait time in seconds
        self.batch_queue: Queue[Batch] = Queue(maxsize=batch_queue_size)
        self.stop_event = multiprocessing.Event()

    def _collect_batch(self) -> None:
        """Collect frames from the mailbox and push assembled batches onto the batch queue."""
        while not self.stop_event.is_set():
            batch: Batch[Frame] = Batch(frames=[])
            start_time = time.time()

            while len(batch.frames) < self.batch_size:
                remaining_time = self.batch_collect_timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break

                message = self.mailbox.receive(timeout=remaining_time)
                if message is not None:
                    batch.frames.append(message)
                else:
                    break  # timeout hit

            if batch.frames:
                try:
                    self.batch_queue.put(batch.model_copy(), timeout=0.1)
                except Full:
                    while self.batch_queue.full():
                        time.sleep(0.1)
                        if self.stop_event.is_set():
                            return
                    self.batch_queue.put(batch.model_copy(), timeout=0.1)

    def _collect_data(self) -> Batch | None:
        """Retrieve the next collected batch from the queue, or None if none is ready yet."""
        try:
            return self.batch_queue.get(timeout=0.1)
        except queue.Empty:
            return None
