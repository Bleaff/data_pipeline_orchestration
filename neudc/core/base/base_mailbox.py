"""File is part of the core module of a distributed system.

It defines the BaseMailbox class, which serves as an abstract base class for mailbox implementations.
The mailbox is used for communication between nodes in a distributed system.
The class provides an interface for sending and receiving messages, as well as managing the mailbox's state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Self, TypeVar

T = TypeVar("T")


class BaseMailbox(ABC, Generic[T]):
    """Base class for mailbox implementations.

    This class defines the interface for a mailbox, which is used for communication between nodes in a distributed system.
    """

    @abstractmethod
    def stop(self) -> None:
        """Stop the mailbox.

        This method should be overridden by subclasses to implement the specific shutdown logic for the mailbox.
        """

    @abstractmethod
    def send(self, message: T) -> None:
        """Send a message to the mailbox.

        This method should be overridden by subclasses to implement the specific sending logic for the mailbox.
        """

    @abstractmethod
    def receive(self, timeout: float | None = None) -> T | None:
        """Receive a message from the mailbox, or None if nothing arrived in time.

        This method should be overridden by subclasses to implement the specific receiving logic for the mailbox.
        """

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict[str, Any]) -> Self:
        """Create a mailbox from a state dictionary.

        This method should be overridden by subclasses to implement the specific creation logic for the mailbox.
        """
