"""Base class for inference nodes that batch incoming frames before running the model."""

from __future__ import annotations

import queue
import time
from abc import abstractmethod
from copy import copy
from queue import Full, Queue
from threading import Thread
from typing import Any

from neudc.core.communication.messaging.types import Batch, Frame
from neudc.core.node.model.base_process_inference import BaseProcessInference
from neudc.nn import ModelFactory
from neudc.utils import LOGGER


class BaseBatchProcessInference(BaseProcessInference):
    """Process-inference node that batches incoming frames before running the model.

    A background thread (`_collect_batch`) accumulates frames from the mailbox into
    fixed-size (or timeout-bounded) batches and hands them to `process()` via a queue.
    """

    def __init__(  # noqa: PLR0917 -- forwarded verbatim via *args/**kwargs by subclasses; reordering risks breakage.
        self,
        batch_size: int,
        model_config: dict[str, Any],
        mailbox: Any,
        batch_queue_size: int = 20,
        batch_collect_timeout: float = 0.1,
        id: str = "BaseBatchProcessInference",
    ) -> None:
        """Initialize the batching inference node.

        Args:
        ----
            batch_size (int): Target number of frames per batch.
            model_config (dict[str, Any]): Configuration used to build the model via `ModelFactory`.
            mailbox (Any): Mailbox to receive incoming frames from.
            batch_queue_size (int): Maximum number of pending batches queued for processing.
            batch_collect_timeout (float): Max seconds to wait while filling a single batch.
            id (str): Identifier for this node instance.

        """
        super().__init__(model_config, mailbox, id=id)
        self.batch_size = batch_size
        self.batch_queue_size = batch_queue_size
        self.batch_collect_timeout = batch_collect_timeout

    def _collect_batch(self) -> None:
        """Collect frames from the mailbox and push assembled batches onto the batch queue."""
        while not self.stop_event.is_set():
            batch: Batch[Frame] = Batch(frames=[])
            start_time = time.time()

            while len(batch.frames) < self.batch_size:
                remaining_time = self.batch_collect_timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                if self.mailbox is None:
                    continue
                message = self.mailbox.receive(timeout=remaining_time)
                if message is not None:
                    batch.frames.append(message)
                else:
                    break  # timeout hit

            if batch.frames:
                try:
                    self.batch_queue.put(batch.model_copy(), timeout=0.1)
                    LOGGER.debug(f"-------------------->[COLLECTED BATCH SIZE OF {len(batch.frames)}]")
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

    # Intentionally diverges from BaseProcessInference: this family processes Batch, not Frame.
    @abstractmethod
    def postprocess_result(self, result: Any, item: Batch) -> Batch | Frame | None:  # type: ignore[override]
        """Postprocess the inference result.

        Args:
        ----
            result:Any Inference result for batch.
            item:Batch Input data.

        Returns:
        -------
            Any: Processed data.
            If it is multiple output, return a Batch with saved frames. Could return Batch with 1 Frame. Could return Frame. Could return None

        """
        msg = "postprocess_result not implemented"
        raise NotImplementedError(msg)

    def init_process_runtime(self) -> None:
        """Initialize the model.

        This method should be implemented by subclasses to initialize the model.

        Returns
        -------
            Any: Initialized model.

        """
        if self._model_initialized.is_set():
            return
        self._model_initialized.set()
        LOGGER.info(f"Created model at {id(self)}")
        LOGGER.info(f"Initializing model...🙈\nModel config is{self.model_config}")
        self.model = ModelFactory.create(copy(self.model_config))
        self.batch_collect_thread = Thread(target=self._collect_batch, daemon=True)
        self.batch_queue: Queue[Batch] = Queue(maxsize=self.batch_queue_size)
        self.batch_collect_thread.start()

    # Intentionally diverges from BaseProcessInference: this family processes Batch, not Frame.
    def process(self, item: Batch) -> Any:  # type: ignore[override]
        """Process the input data.

        This method is called for each input data. It initializes the model on the
        first call and then runs the model on the input data.

        Args:
        ----
            item: Input data to be processed.

        Returns:
        -------
            Any: Processed data.

        """
        # Run the model on the input data
        while self.model is None:
            time.sleep(0.1)
        batch = [frame.image for frame in item.frames]
        result = self.model(batch)
        return self.postprocess_result(result, item)

    @classmethod
    def from_config(cls: type[BaseBatchProcessInference], config: dict[str, Any]) -> BaseBatchProcessInference:
        """From config-based constructor for building node with specified config.

        Args:
        ----
            config: Configuration dictionary for the node.

        Returns:
        -------
            BaseProcessNode: Node instance created from the configuration.

        """
        return cls(**config)
