"""BaseNode class serves as an abstract base class for node implementations in a distributed system.

Nodes are the fundamental building blocks that process data and communicate with
each other through mailboxes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseNode(ABC):
    """Abstract base class for node implementations in a distributed system.

    Attributes
    ----------
        config (dict[str, Any]): Configuration details for the node.
        mailbox (Any): Communication mailbox for the node.

    """

    @abstractmethod
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the BaseNode with the given configuration.

        Args:
        ----
            config (dict[str, Any]): Configuration details for the node.

        """
        self.config = config
        self.mailbox = None

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the node's main functionality.

        Args:
        ----
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
        -------
            Any: Result of the node's execution.

        Raises:
        ------
            NotImplementedError: If the method is not implemented by a subclass.

        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Get the name of the node.

        Returns
        -------
            str: The class name of the node.

        """
        return self.__class__.__name__

    @property
    def config(self) -> dict[str, Any]:
        """Get the configuration of the node.

        Returns
        -------
            dict[str, Any]: The node's configuration details.

        """
        return self._config

    @config.setter
    def config(self, config: dict[str, Any]) -> None:
        """Set the configuration of the node.

        Args:
        ----
            config (dict[str, Any]): The new configuration details for the node.

        """
        self._config = config
