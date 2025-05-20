"""File is part of the core module of a distributed system.

It defines the BaseMailbox class, which serves as an abstract base class for mailbox implementations.
The mailbox is used for communication between nodes in a distributed system.
The class provides an interface for sending and receiving messages, as well as managing the mailbox's state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseMailbox(ABC, Generic[T]):
    """Base class for mailbox implementations.

    This class defines the interface for a mailbox, which is used for communication between nodes in a distributed system.
    """

    @abstractmethod
    def start(self) -> None:
        """Start the mailbox.

        This method should be overridden by subclasses to implement the specific startup logic for the mailbox.
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
    def receive(self) -> T:
        """Receive a message from the mailbox.

        This method should be overridden by subclasses to implement the specific receiving logic for the mailbox.
        """

    @abstractmethod
    def has_messages(self) -> bool:
        """Check if there are messages in the mailbox.

        This method should be overridden by subclasses to implement the specific logic for checking messages in the mailbox.
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear the mailbox.

        This method should be overridden by subclasses to implement the specific logic for clearing the mailbox.
        """
