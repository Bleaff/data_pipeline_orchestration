
import abc
from typing import Any, Optional
import time
from neudc.core.base.base_process import BaseProcessNode
from neudc.core.communication.messaging.types import Frame
from multiprocessing import Event
from neudc.nn import ModelFactory
import copy

class BaseProcessInference(BaseProcessNode, metaclass=abc.ABCMeta):
    """
    Base class for inference nodes.

    This class provides a basic implementation for inference nodes. It includes
    model initialization and the process method.

    The process method is implemented to initialize the model on the first call
    and then run the model on the input data.

    """

    def __init__(self, model_config, mailbox: Any, logger: Any) -> None:
        """
        Initialize the base inference node.

        Args:
        ----
            mailbox: Mailbox for inter-process communication.
            logger: Logger instance for logging messages.
        """
        super().__init__(mailbox, logger)
        self.model = None
        self.model_config = model_config
        self._model_initialized = Event()
        self.model_initialization_status = Event()

    def process(self, item: Frame) -> Any:
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

        result = self.model([item.image,])
        pp_item = self.postprocess_result(result, item)
        return pp_item

    @abc.abstractmethod
    def postprocess_result(self, result: Any, item: Frame)-> Optional[Frame]:
        """
        Postprocess the inference result.

        Args:
        ----
            result:Any Inference result.
            item:Frame Input data.

        Returns:
        -------
            Any: Processed data.
        """
        raise NotImplementedError("postprocess_result not implemented")

    @classmethod
    def from_config(cls: type[BaseProcessNode], config: dict[str, Any]) -> BaseProcessNode:
        """
        From config-based constructor for building node with specified config.

        Args:
        ----
            config: Configuration dictionary for the node.

        Returns:
        -------
            BaseProcessNode: Node instance created from the configuration.
        """
        return cls(config["model_config"], config["mailbox"], config["logger"])

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
        self.model = ModelFactory.create(copy.copy(self.model_config))

