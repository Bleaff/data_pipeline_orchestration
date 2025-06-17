from __future__ import annotations

import multiprocessing
import queue
import time
from queue import Full, Queue
from typing import Any

from neudc.core.communication.messaging.types import Batch


class CollectBatchMixin:
    """Mixin class for collecting data from mailbox."""

    def __init__(self, batch_size: int, batch_queue_size: int = 20, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.batch_collect_timeout = 0.1  # max wait time in seconds
        self.batch_queue: Queue[Batch] = Queue(maxsize=batch_queue_size)
        self.stop_event = multiprocessing.Event()

    def _collect_batch(self) -> None:
        """Collects data from mailbox and puts batches into a queue."""
        while not self.stop_event.is_set():
            batch = Batch()
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
                    self.batch_queue.put(batch.copy(), timeout=0.1)
                except Full:
                    self.logger.warning("Batch queue full, dropping batch.")

    def _collect_data(self) -> Batch | None:
        """Retrieves collected batch from the queue."""
        try:
            return self.batch_queue.get(timeout=0.1)
        except queue.Empty:
            return None
