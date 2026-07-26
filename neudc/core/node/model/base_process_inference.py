"""Base class for process-inference nodes: a `BaseProcessNode` that lazily loads a model."""

from __future__ import annotations

import copy
import time
from abc import ABCMeta, abstractmethod
from multiprocessing import Event
from typing import TYPE_CHECKING, Any

from neudc.core.base.base_process import BaseProcessNode
from neudc.nn import ModelFactory
from neudc.utils import LOGGER

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import Frame
    from neudc.nn.models.cls.base import BaseClsModel
    from neudc.nn.models.det.base import BaseDetector


class BaseProcessInference(BaseProcessNode, metaclass=ABCMeta):
    """Base class for inference nodes.

    This class provides a basic implementation for inference nodes. It includes
    model initialization and the process method.

    The process method is implemented to initialize the model on the first call
    and then run the model on the input data.

    """

    def __init__(self, model_config: dict, mailbox: Any, id: str = "BaseProcessInference") -> None:
        """Initialize the base inference node.

        Args:
        ----
            model_config: Configuration used to build the model via `ModelFactory`.
            mailbox: Mailbox for inter-process communication.
            id: Identifier for this node instance.

        """
        super().__init__(mailbox, id=id)
        self.model: BaseClsModel | BaseDetector | None = None
        self.model_config = model_config
        self._model_initialized = Event()
        self.model_initialization_status = Event()

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
        # Run the model on the input data
        while self.model is None:
            time.sleep(0.1)

        result = self.model(
            [
                item.image,
            ],
        )
        return self.postprocess_result(result, item)

    @abstractmethod
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

    @classmethod
    def from_config(cls: type[BaseProcessInference], config: dict[str, Any]) -> BaseProcessInference:
        """From config-based constructor for building node with specified config.

        Args:
        ----
            config: Configuration dictionary for the node.

        Returns:
        -------
            BaseProcessInference: Node instance created from the configuration.

        """
        return cls(config["model_config"], config["mailbox"], id=config["id"])

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
        LOGGER.info(f"Initializing model...\nModel config is{self.model_config}")
        self.model = ModelFactory.create(copy.copy(self.model_config))
