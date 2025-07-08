from neudc.core.node.model.base_process_inference import BaseProcessInference
from neudc.core.node.model.mixins.collect_batch import CollectBatchMixin
from neudc.core.communication.messaging.types import Batch, Frame
from typing import Any, List, Optional
from abc import ABC, abstractmethod
from neudc.nn import ModelFactory
from queue import Full, Queue
from threading import Thread
from copy import copy
import queue
import time

class BaseBatchProcessInference(BaseProcessInference):
    def __init__(self, batch_size:int, model_config, mailbox: Any, logger: Any, batch_queue_size: int = 20, batch_collect_timeout: float = 0.1) -> None:
        super().__init__(model_config, mailbox, logger)
        self.batch_size = batch_size
        self.batch_queue_size = batch_queue_size
        self.batch_collect_timeout = batch_collect_timeout

    def _collect_batch(self) -> None:
        """Collects data from mailbox and puts batches into a queue."""
        while not self.stop_event.is_set():
            batch = Batch(frames=list())
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
                    self.logger.debug(f"-------------------->[COLLECTED BATCH SIZE OF {len(batch.frames)}]")
                except Full:
                    while self.batch_queue.full():
                        time.sleep(0.1)
                        if self.stop_event.is_set():
                            return
                    self.batch_queue.put(batch.model_copy(), timeout=0.1)

    def _collect_data(self) -> Batch | None:
        """Retrieves collected batch from the queue."""
        try:
            return self.batch_queue.get(timeout=0.1)
        except queue.Empty:
            return None


    @abstractmethod
    def postprocess_result(self, result: Any, item: Batch)-> Optional[Batch] | Frame | None:
        """
        Postprocess the inference result.

        Args:
        ----
            result:Any Inference result for batch.
            item:Batch Input data.

        Returns:
        -------
            Any: Processed data.
            If it is multiple output, return a Batch with saved frames. Could return Batch with 1 Frame. Could return Frame. Coudl return None
        """
        raise NotImplementedError("postprocess_result not implemented")

    def init_process_runtime(self) -> None:
        """
        Initialize the model.

        This method should be implemented by subclasses to initialize the model.

        Returns:
        -------
            Any: Initialized model.
        """
        if self._model_initialized.is_set():
            return
        self._model_initialized.set()
        self.logger.info(f"Created model at {id(self)}")
        self.logger.info(f"Initializing model...🙈\nModel config is{self.model_config}")
        self.model = ModelFactory.create(copy(self.model_config))
        self.batch_collect_thread = Thread(target=self._collect_batch, daemon=True)
        self.batch_queue: Queue[Batch] = Queue(maxsize=self.batch_queue_size)
        self.batch_collect_thread.start()

    def process(self, item: Batch) -> Any:
        """
        Process the input data.

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
        pp_item = self.postprocess_result(result, item)
        return pp_item
    
    @classmethod
    def from_config(cls: type["BaseBatchProcessInference"], config: dict[str, Any]) -> "BaseBatchProcessInference":
        """
        From config-based constructor for building node with specified config.

        Args:
        ----
            config: Configuration dictionary for the node.

        Returns:
        -------
            BaseProcessNode: Node instance created from the configuration.
        """
        return cls(**config)