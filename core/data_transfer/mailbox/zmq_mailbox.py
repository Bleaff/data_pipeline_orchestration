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

import zmq

from core.base.base_mailbox import BaseMailbox


class ZMQMailbox(BaseMailbox[dict]):
    """ZeroMQ-based mailbox implementation."""

    def __init__(self, address: str = "tcp://127.0.0.1:5555", *, bind: bool = True) -> None:
        """Initialize the ZeroMQ mailbox.

        Args:
        ----
            address: The ZeroMQ address to bind or connect to.
            bind: Whether to bind or connect the socket.

        """
        self.address = address
        self.bind = bind
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self._started = False

    def start(self) -> None:
        """Start the mailbox.

        This method sets up the ZeroMQ socket connection.
        """
        if self.bind:
            self.socket.bind(self.address)
        else:
            self.socket.connect(self.address)

        self._started = True

    def stop(self) -> None:
        """Stop the mailbox.

        Closes the ZeroMQ socket and context.
        """
        self.socket.close(0)
        self.context.term()
        self._started = False

    def send(self, message: dict) -> None:
        """Send a message to the mailbox.

        Args:
        ----
            message: The message to send (as a dict).

        """
        self.socket.send_json(message)

    def receive(self) -> dict:
        """Receive a message from the mailbox.

        Returns
        -------
            The received message as a dict.

        """
        return self.socket.recv_json()

    def has_messages(self) -> bool:
        """Check if there are messages available in the socket.

        Returns
        -------
            True if a message is ready, False otherwise.

        """
        return self.socket.poll(timeout=0) == zmq.POLLIN

    def clear(self) -> None:
        """Clear all pending messages in the mailbox."""
        while self.socket.poll(timeout=0) == zmq.POLLIN:
            _ = self.socket.recv_json(zmq.NOBLOCK)
