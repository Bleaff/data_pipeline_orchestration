"""ZMQMailbox module for inter-node communication in a distributed system.

This module defines the ZMQMailbox class, a concrete implementation of the
BaseMailbox interface. It uses ZeroMQ (PAIR socket) to enable inter-process or
inter-thread communication between nodes. The mailbox supports message sending,
receiving, checking for pending messages, and clearing its queue.

Typical usage:
    mailbox = ZMQMailbox("tcp://127.0.0.1:5555", bind=True)
    mailbox.start()
    mailbox.send({"event": "hello"})
    if mailbox.has_messages():
        msg = mailbox.receive()
    mailbox.clear()
    mailbox.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import Any

from neudc.core.base.base_mailbox import BaseMailbox
from neudc.core.communication.zero_queue import ZeroQueuePub, ZeroQueueSub
from neudc.core.communication.zero_queue.zmq_state import ZeroQueueConnectionType

logging.basicConfig(level=logging.DEBUG)


class ZMQMailbox(BaseMailbox[dict]):
    """ZeroMQ-based mailbox implementation."""

    def __init__(
        self,
        *,
        message_queue_size: int = 20,
        logger: logging.Logger | None = None,
        name: str = "ZMQMailbox",
    ) -> None:
        """Initialize the ZeroMQ mailbox."""
        if not logger:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.pub_sockets: dict[int, ZeroQueuePub] = {}
        self.sub_queue: ZeroQueueSub = ZeroQueueSub()
        self.consume_port: int = self.sub_queue.port
        self._message_queue: Queue = Queue(message_queue_size)
        self._running = False
        self._thread: threading.Thread | None = None
        self._join_timeout = 0.1
        self.name = name

        if not logger:
            self.logger = logging.getLogger(__name__)
        self.start()

    def _receiver_loop(self) -> None:
        """Thread loop for receiving messages from the ZeroMQ subscriber."""
        _unsent_message: Any = None
        while self._running:
            message = self.sub_queue.get(timeout=self._join_timeout)
            if message:
                try:
                    self._message_queue.put(message, timeout=1)
                    if self.logger:
                        self.logger.debug(f"[RECV] ← message of type {type(message)}")

                except Full:
                    self.logger.warning("Message queue is full")
                    while self._message_queue.full():
                        time.sleep(0.01)  # Sleep briefly to avoid busy waiting
                    self._message_queue.put(_unsent_message)
                    self.logger.debug("[SENDING] → unsent message")
                except Exception:
                    self.logger.exception("Error in receiver loop")

    def start(self) -> None:
        """Start the receiving thread."""
        self._running = True
        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the mailbox."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._join_timeout)

        self.sub_queue.stop()

        for pub_socket in self.pub_sockets.values():
            pub_socket.stop()

    def send(self, message: Any) -> None:
        """Send a message to the mailbox."""
        for pub_socket in self.pub_sockets.values():
            if self.logger:
                self.logger.debug(f"[{self.name}][START SENDING] → {time.time()}")
            pub_socket.put(message)
            if self.logger:
                self.logger.debug(f"[{self.name}][END SENDING] → {time.time()}")

    def receive(self) -> dict:
        """Receive a message from the mailbox."""
        try:
            message = self._message_queue.get(timeout=0.1)
            if self.logger:
                self.logger.debug(f"[{self.name}][RECV][{time.time()}] ← {type(message)}")
        except Empty:
            message = None
        return message

    def add_publisher(self, port: int) -> None:
        """Connect a publisher to the mailbox. This method is not thread-safe."""
        self.pub_sockets[port] = ZeroQueuePub(port=port)
        if self.logger:
            self.logger.debug(f"[{self.name}][Added publisher] → port:{port}")

    def remove_publisher(self, port: int) -> None:
        """Remove a publisher from the mailbox."""
        rm_pub = self.pub_sockets.pop(port)
        if self.logger:
            self.logger.debug(f"[{self.name}][Removed publisher] → {rm_pub}")

    def __getstate__(self) -> dict[str, Any]:
        """Get the current state of the mailbox."""
        self.logger.debug(f"[{self.name}][GET STATE] → {self.name}")
        state = self.__dict__.copy()
        # Remove the thread and logger from the state to avoid issues with pickling
        state["_thread"] = None
        state["_running"] = False
        state["logger"] = None  # Avoid pickling the logger
        state["message_queue_size"] = self._message_queue.qsize()  # Store size instead of the queue itself
        state["pub_sockets"] = list(self.pub_sockets.keys())
        state["consume_port"] = self.consume_port
        state["sub_queue"] = None
        state["_message_queue"] = None  # Avoid pickling the queue itself

        self.stop()
        # Stop the thread to ensure a consistent state
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the mailbox from a saved state."""
        self.name = f'PICKLED|{state.get("name", "ZMQMailbox")}'
        self.logger = logging.getLogger(self.name)
        self.logger.debug(f"[{self.name}][SET STATE] → {self.name}")
        self._message_queue = Queue(state["message_queue_size"])

        self.pub_sockets = {port: ZeroQueuePub(port=port) for port in state["pub_sockets"]}
        self.sub_queue = ZeroQueueSub(port=state["consume_port"], contype=ZeroQueueConnectionType.BIND)
        self.consume_port = state["consume_port"]
        self._message_queue = Queue(state["message_queue_size"])

        self._running = True
        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._join_timeout = 0.5
        self._thread.start()
    
    @staticmethod
    def from_state(cls, state: dict[str, Any]) -> ZMQMailbox:
        mailbox = cls()
        mailbox.stop()
        mailbox.__setstate__(state)
        return mailbox
