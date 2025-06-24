"""Base node class that runs in a separate process.

This module contains the base class for all process nodes. Process nodes are a
special type of node that can be used to run other nodes in separate processes,
allowing for parallel execution of node graphs.
"""

from __future__ import annotations
import multiprocessing as mp
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from neudc.core.base.base_node import BaseNode

class BaseProcessNode(BaseNode, mp.Process, ABC):
    """Base node class that runs in a separate process.

    Inherits from BaseNode and multiprocessing.Process.
    Includes health monitoring and lifecycle control.
    """

    HEALTH_INITIAL = False
    HEALTH_NORMAL = True
    HEALTH_CHECK_INTERVAL = 5  # seconds
    HEALTH_TIMEOUT = 15  # seconds

    def __init__(self, mailbox: Any, logger: Any) -> None:
        """Initialize the base process node.

        Args:
        ----
            mailbox: Mailbox for inter-process communication.
            logger: Logger instance for logging messages.
            config: Configuration dictionary for the node.

        """
        # Initialize BaseNode and multiprocessing.Process
        BaseNode.__init__(self, None, logger)
        self.mailbox_config = mailbox.__getstate__()
        mp.Process.__init__(self)
        self._healthy = mp.Value("b", self.HEALTH_INITIAL)

    def start(self) -> None:
        """Start the process node."""
        self.logger.info(f"Starting process node {self.id}...")
        mp.Process.start(self)

    def _start_afterwords(self) -> None:
        """Start the base node's thread & initialize health monitoring."""
        from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox 

        
        self._health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        self._health_thread.start()
        # Call the start method of BaseNode to initialize its thread

        self.mailbox = ZMQMailbox.from_state(ZMQMailbox,self.mailbox_config)
        BaseNode.mailbox = self.mailbox
        BaseNode.init_runtime(self)

    def init_process_runtime(self) -> None:
        """Initialize runtime resources for the process node."""
        pass
    
    def run(self) -> None:
        """Process entrypoint: start health monitor and processing loop."""
        self.logger.info(f"Starting process node {self.id}...(PID: {os.getpid()})")
        self.init_process_runtime()
        self._healthy = mp.Value("b", self.HEALTH_NORMAL)
        self.stop_event = mp.Event()
        self._last_success_time = mp.Value("d", time.time())

        self._start_afterwords()  # Start the node's thread
        self.logger.info(f"Process node started, entering processing loop...Mailbox status:{self.mailbox.consume_port}")

        while not self.stop_event.is_set():
            try:
                data = self.mailbox.receive()
                if data is None:
                    continue
                result = self.process(data)
                self.mailbox.send(result)
                with self._last_success_time.get_lock():
                    self._last_success_time.value = time.time()
            except Exception:
                self.logger.exception("[BaseProcessNode] Exception")

    def stop(self) -> None:
        """Signal the process to stop and wait for the health thread."""
        self.logger.info("Stopping process node...")
        self.stop_event.set()
        if self._health_thread:
            self._health_thread.join(timeout=2)

    def is_healthy(self) -> bool:
        """Return current health status."""
        with self._healthy.get_lock():
            return self._healthy.value
    
    def status(self) -> bool:
        return self.is_healthy()

    def _health_monitor(self) -> None:
        """Monitor if node continues to process over time."""
        while not self.stop_event.is_set():
            time.sleep(self.HEALTH_CHECK_INTERVAL)
            with self._last_success_time.get_lock():
                delta = time.time() - self._last_success_time.value
            if delta > self.HEALTH_TIMEOUT:
                self.logger.warning("Health timeout exceeded!")
                with self._healthy.get_lock():
                    self._healthy.value = False

    @abstractmethod
    def process(self, item: Any) -> Any:
        """Core logic of the node. Must be implemented."""
        raise NotImplementedError

    @classmethod
    def from_config(cls: type[BaseProcessNode], config: dict[str, Any]) -> BaseProcessNode:
        """From config-based constructor for building node with specified config."""
        raise NotImplementedError

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the process node."""
        # Update the instance's __dict__ with the state dictionary
        # This is necessary for unpickling the process node
        self.__dict__.update(state)
        import logging

        self.logger = logging.getLogger(f"ProcessNode.{self.id}")
