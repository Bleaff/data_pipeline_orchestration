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

from typing import TYPE_CHECKING

import zmq

from core.base.base_mailbox import BaseMailbox

if TYPE_CHECKING:
    import logging


class ZMQMailbox(BaseMailbox[dict]):
    """ZeroMQ-based mailbox implementation."""

    def __init__(
        self,
        address: str = "tcp://127.0.0.1:5555",
        *,
        bind: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the ZeroMQ mailbox."""
        self.address = address
        self.bind = bind
        self.context = zmq.Context()
        self._started = False
        self.logger = logger

    def start(self) -> None:
        """Start the mailbox."""
        if self._started:
            msg = "ZMQMailbox already started"
            raise RuntimeError(msg)

        self.socket: zmq.Socket = self.context.socket(zmq.PAIR)
        self.socket.linger = 0
        (self.socket.bind if self.bind else self.socket.connect)(self.address)
        self._started = True
        if self.logger:
            self.logger.debug(f"ZMQMailbox started at {self.address} (bind={self.bind})")

    def stop(self) -> None:
        """Stop the mailbox."""
        self.socket.close(0)
        self.context.term()
        self._started = False
        if self.logger:
            self.logger.debug(f"ZMQMailbox stopped at {self.address}")

    def send(self, message: dict) -> None:
        """Send a message to the mailbox."""
        if not self._started or self.socket is None:
            msg = "Mailbox not started"
            raise RuntimeError(msg)
        self.socket.send_json(message)
        if self.logger:
            self.logger.debug(f"[SEND] → {self.address}: {message}")

    def receive(self) -> dict:
        """Receive a message from the mailbox."""
        message = self.socket.recv_json()
        if self.logger:
            self.logger.debug(f"[RECEIVE] ← {self.address}: {message}")
        return message

    def has_messages(self) -> bool:
        """Check if there are messages available in the socket."""
        return self.socket.poll(timeout=0) == zmq.POLLIN

    def clear(self) -> None:
        """Clear all pending messages in the mailbox."""
        while self.socket.poll(timeout=0) == zmq.POLLIN:
            dropped = self.socket.recv_json(zmq.NOBLOCK)
            if self.logger:
                self.logger.debug(f"[CLEAR] Dropped message: {dropped}")
