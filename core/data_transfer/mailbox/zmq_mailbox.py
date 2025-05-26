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
from queue import Queue, Empty, Full
import time
import logging
from typing import Any

from core.base.base_mailbox import BaseMailbox
from core.data_transfer.zero_queue import ZeroQueuePub, ZeroQueueSub
import threading


logging.basicConfig(level=logging.DEBUG)


class ZMQMailbox(BaseMailbox[dict]):
    """ZeroMQ-based mailbox implementation."""

    def __init__(
        self,
        *,
        message_queue_size: int = 20,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the ZeroMQ mailbox."""
        self.logger = logger
        self.pub_sockets: dict[int, ZeroQueuePub] = {}
        self.sub_queue: ZeroQueueSub = ZeroQueueSub()
        self.consume_port: int = self.sub_queue.port
        self._message_queue: Queue = Queue(message_queue_size)
        self._running = False
        self._thread = None

        if not logger:
            self.logger = logging.getLogger(__name__)
    def _receiver_loop(self) -> None:
        """Thread loop for receiving messages from the ZeroMQ subscriber."""
        _unsent_message: Any = None
        while self._running:
            try:
                message = self.sub_queue.get(timeout=1)
                self._message_queue.put(message, timeout=1)
                if self.logger:
                    self.logger.debug(f"[ZMQMailbox] [RECV] ← message")
            except Empty:
                continue
            except Full:
                self.logger.warning("Message queue is full")
                while self._message_queue.full():
                    time.sleep(0.01)  # Sleep briefly to avoid busy waiting
                self._message_queue.put(_unsent_message)
                self.logger.debug(f"[_MESSAGE_QUEUE] [SENDING] -> unsent message")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in receiver loop: {e}")
            
    def start(self) -> None:
        """Start the receiving thread."""
        self._running = True
        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._thread.start()


    def stop(self) -> None:
        """Stop the mailbox."""
        self.sub_queue.stop()
        for pub_socket in self.pub_sockets.values():
            pub_socket.stop()

    def send(self, message: Any) -> None:
        """Send a message to the mailbox."""
        for pub_socket in self.pub_sockets.values():
            pub_socket.put(message)
        if self.logger:
            self.logger.debug(f"[SEND] → {message}")

    def receive(self) -> dict:
        """Receive a message from the mailbox."""
        message = self.sub_queue.get_nowait()
        if self.logger:
            self.logger.debug(f"[RECV] ← {message}")
        return message

    def add_publisher(self, port: int) -> None:
        self.pub_sockets[port] = ZeroQueuePub(port=port)

    def remove_publisher(self, port: int) -> None:
        self.pub_sockets.pop(port)
