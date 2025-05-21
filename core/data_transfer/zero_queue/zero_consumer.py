"""ZeroMQ-based message consumer for inter-process communication.

This module provides the ZeroQueueConsumer class, which implements the
zmq.REQ/zmq.REP pattern for message passing between processes on a single
machine.

"""

from __future__ import annotations

import logging
import queue
from typing import Any

import zmq

logger = logging.getLogger(__name__)


class ZeroQueueConsumer:
    """Message consumer using REQ/REP pattern."""

    TIMEOUT_MS = 100

    def __init__(self, port: int) -> None:
        """Initialize the ZeroQueueConsumer.

        Args:
        ----
            port: Port to bind for incoming messages.

        """
        self.port = port
        self._reconnect()

    def __str__(self) -> str:
        """Representation of the ZeroQueueConsumer with occupied port."""
        return f"{self.__class__.__name__}(port={self.port})"

    def _reconnect(self) -> None:
        """(Re)initialize context and REP socket."""
        if self.socket_rep:
            self.stop()

        logger.debug("Reconnecting ZeroQueueConsumer...")

        self.context: zmq.Context = zmq.Context()
        self.socket_rep: zmq.Context.socket = self.context.socket(zmq.REP)
        self.socket_rep.setsockopt(zmq.LINGER, 0)
        self.socket_rep.bind(f"tcp://*:{self.port}")
        logger.info(f"ZeroQueueConsumer bound to port {self.port}")

        self.poller: zmq.Poller = zmq.Poller()
        self.poller.register(self.socket_rep, zmq.POLLIN)

    def stop(self) -> None:
        """Close sockets and terminate context."""
        logger.debug("Stopping ZeroQueueConsumer...")
        try:
            if self.socket_rep:
                self.socket_rep.close()
        except zmq.ZMQError as e:

            logger.warning(f"Error while closing socket: {e}")
        try:
            if self.context:
                self.context.term()
        except zmq.ZMQError as e:
            logger.warning(f"Error while terminating context: {e}")
        self.socket_rep = None
        self.context = None

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.stop()

    def get(self, timeout: float | None = None) -> Any:
        """Receive a message (blocking).

        Returns
        -------
            Deserialized object.

        Raises
        ------
            queue.Empty: If no message received within timeout.

        """
        timeout_ms = int(timeout * 1000) if timeout else self.TIMEOUT_MS
        try:
            if self.socket_rep in dict(self.poller.poll(timeout_ms)):
                msg = self.socket_rep.recv_pyobj()
                self.socket_rep.send(b"ACK")
                logger.debug("Received message and sent ACK.")
                return msg
        except zmq.ZMQError:
            logger.exception("Failed to process message")

        msg = "No message received."
        raise queue.Empty(msg)

    def get_nowait(self) -> Any:
        """Non-blocking receive."""
        return self.get(timeout=self.TIMEOUT_MS / 1000)

    def put(self, *args: Any, **kwargs: Any) -> None:
        """Send a message (not implemented for consumer)."""
        raise NotImplementedError

    def put_nowait(self, *args: Any, **kwargs: Any) -> None:
        """Send a message without waiting (not implemented for consumer)."""
        raise NotImplementedError
