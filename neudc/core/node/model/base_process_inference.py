"""Base class for process nodes that perform model inference.

This class is the base class for all process nodes that execute a model. The
class is designed to be used as a node in a dataflow graph. The class is
thread-safe and can be used with asyncio.
"""

from __future__ import annotations

import abc
import copy
from multiprocessing import Event
from typing import TYPE_CHECKING, Any

from neudc.core.base.base_process import BaseProcessNode
from neudc.nn import ModelFactory

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import Frame


class BaseProcessInference(BaseProcessNode, metaclass=abc.ABCMeta):
    """Base class for inference nodes.

    This class provides a basic implementation for inference nodes. It includes
    model initialization and the process method.

    The process method is implemented to initialize the model on the first call
    and then run the model on the input data.

    """

    def __init__(self, model_config: dict[str, Any], mailbox: Any, logger: Any) -> None:
        """Initialize the base inference node.

        Args:
        ----
            mailbox: Mailbox for inter-process communication.
            logger: Logger instance for logging messages.

        """
        super().__init__(mailbox, logger)
        self.model = None  # type: ignore[misc]
        self.model_config = model_config
        self._model_initialized = Event()

    def process(self, item: Frame) -> Any:
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
        if not self._model_initialized.is_set():
            # Initialize the model on the first call
            self.initialize_model()
            self.logger.info(f"Model initialized!-> Warmuping...{self.model}")
            self.model.warmup(iters=10)  # type: ignore[attr-defined]
        # Run the model on the input data
        result = self.model(  # type: ignore[misc]
            [
                item.image,
            ],
        )
        # Print the result
        return self.postprocess_result(result, item)

    @abc.abstractmethod
    def postprocess_result(self, result: Any, item: Frame) -> Frame:
        """Postprocess the inference result.

        Args:
        ----
            result:Any Inference result.
            item:Frame Input data.

        Returns:
        -------
            Any: Processed data.

        """
        msg = "postprocess_result not implemented"
        raise NotImplementedError(msg)

    @classmethod
    def from_config(cls: type[BaseProcessNode], config: dict[str, Any]) -> BaseProcessNode:
        """From config-based constructor for building node with specified config.

        Args:
        ----
            config: Configuration dictionary for the node.

        Returns:
        -------
            BaseProcessNode: Node instance created from the configuration.

        """
        return cls(config["model_config"], config["mailbox"], config["logger"])

    def initialize_model(self) -> None:
        """Initialize the model.

        This method should be implemented by subclasses to initialize the model.

        Returns
        -------
            Any: Initialized model.

        """
        if self._model_initialized.is_set():
            return
        self.logger.info(f"Created model at {id(self)}")
        self.logger.info(f"Initializing model...🙈\nModel config is{self.model_config}")
        self.model = ModelFactory.create(copy.copy(self.model_config))
        self._model_initialized.set()
