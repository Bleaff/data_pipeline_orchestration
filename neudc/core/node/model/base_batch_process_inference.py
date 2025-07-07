from neudc.core.node.model.base_process_inference import BaseProcessInference
from neudc.core.node.model.mixins.collect_batch import CollectBatchMixin
from neudc.core.communication.messaging.types import Batch, Frame
from abc import ABC, abstractmethod
from neudc.nn import ModelFactory
from typing import Any, List, Optional
from threading import Thread
from copy import copy
import time

class BaseBatchProcessInference(CollectBatchMixin, BaseProcessInference):
    def __init__(self, batch_size:int, model_config, mailbox: Any, logger: Any, *args: Any, **kwargs: Any) -> None:
        print(f'{args=}, {kwargs=}')
        print(f'{model_config=}, {mailbox=}, {logger=}')
        CollectBatchMixin.__init__(self, batch_size, *args,  **kwargs)
        BaseProcessInference.__init__(self, model_config, mailbox, logger)


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
        self.batch_collect_thread = Thread(target=self._collect_batch, daemon=True)
        self.batch_collect_thread.start()
        if self._model_initialized.is_set():
            return
        self._model_initialized.set()
        self.logger.info(f"Created model at {id(self)}")
        self.logger.info(f"Initializing model...🙈\nModel config is{self.model_config}")
        self.model = ModelFactory.create(copy.copy(self.model_config))

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

        result = self.model(item.frames)
        pp_item = self.postprocess_result(result, item)
        return pp_item