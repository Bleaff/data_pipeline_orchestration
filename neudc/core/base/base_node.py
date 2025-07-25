"""BaseNode class serves as an abstract base class for node implementations in a distributed system.

Nodes are the fundamental building blocks that process data and communicate with
each other through mailboxes.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any

from neudc.utils import LOGGER


class BaseNode(ABC):
    """Abstract base class for node implementations in a distributed system.

    ----
    This class defines the standard methods a node must implement to be used in
    the system. It provides a common interface for all nodes, regardless of the
    specific functionality they provide.
    ----
    Nodes are the fundamental building blocks that process data and communicate
    with each other through mailboxes.
    ----
    There are 2 main types of nodes:
    1. Process Nodes: These nodes run in separate processes and can handle
       parallel execution of tasks.
    2. Threaded Nodes: These nodes run in separate threads and are suitable for
       tasks that require concurrency without the overhead of process management.
    ----
    Each node must implement the `process` method, which contains the main
    functionality of the node. The `run` method is responsible for continuously
    processing data from the mailbox and sending results back to the mailbox.
    It is also possible to change data collection logic by overriding the `_collect_data` method.
    ----
    The `start` method initializes the node and starts its processing thread.
    By default you can use just `init_runtime` method to make your node work.
    """

    def __init__(self, mailbox: Any, id: str = "BaseNode", _join_timeout: float = 0.1) -> BaseNode:
        """Initialize the node with a mailbox and a logger.

        Args:
        ----
            mailbox (Any): The mailbox to use for communication with other nodes.
            _id (str): The ID of the node.
            _join_timeout (float): The timeout for joining the thread.

        """
        super().__init__()
        self.mailbox = mailbox
        self.thread: threading.Thread | None = None
        self._join_timeout = _join_timeout
        self.is_running = False
        self.id = id
        self.is_ready = False

    def _collect_data(self) -> Any:
        """Grabs data from mailbox."""
        return self.mailbox.receive()

    @classmethod
    @abstractmethod
    def from_config(cls: BaseNode, config: dict[str, Any]) -> BaseNode:
        """Create a node instance from the given configuration.

        Args:
        ----
            config (dict[str, Any]): Configuration details for the node.

        Returns:
        -------
            BaseNode: A node instance.

        Raises:
        ------
            NotImplementedError: If the method is not implemented by a subclass.

        """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Start the processing loop by wrapping the `run` method."""
        return self._run(*args, **kwargs)

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the node's main functionality.

        Raises
        ------
            NotImplementedError: If the method is not implemented by a subclass.

        """

    def _run(self) -> None:
        """Run the node processing loop."""
        self.is_ready = True
        while self.is_running:
            data = self._collect_data()
            if data is not None:
                try:
                    result = self.process(data)
                    if result:
                        self.mailbox.send(result)
                except Exception as e:
                    self.is_ready = False
                    LOGGER.exception("Error while processing", exc_info=e)

    @abstractmethod
    def start(self) -> None:
        """Start func should implement behavior to initialize the node."""

    def init_runtime(self) -> None:
        """Initialize runtime resources for the node."""
        # This method can be overridden by subclasses to initialize specific resources
        LOGGER.info("Starting node...")
        self.is_running = True
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        LOGGER.info("Node started.")

    def stop(self) -> None:
        """Signal the thread to stop and wait for it."""
        LOGGER.info("Stopping node...")
        self.is_running = False
        self.is_ready = False
        self._stop_event.set()
        self.mailbox.stop()
        # prevent joining current thread
        if threading.current_thread() != self.thread and self.thread.is_alive():
            self.thread.join(timeout=self._join_timeout)
        LOGGER.info("Node stopped.")

    def status(self) -> bool:
        """Check the status of the node."""
        return self.is_ready
