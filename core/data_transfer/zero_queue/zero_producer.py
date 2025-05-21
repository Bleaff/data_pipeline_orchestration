"""ZeroQueueProducer module for inter-process communication.

This module provides the ZeroQueueProducer class, which implements the
zmq.REQ/zmq.REP pattern for message passing between processes on a single
machine.

The producer class is responsible for sending messages to the consumer class.
The messages are sent in a loop until the stop() method is called.

The class also provides a method to check if the queue is empty.
"""

from __future__ import annotations

import logging
import queue
from collections import deque
from typing import Any

import zmq

logger = logging.getLogger(__name__)


class ZeroQueueProducer:
    """Message producer using REQ/REP pattern with buffering and retry logic."""

    DEQUE_LEN = 100
    TIMEOUT_MS = 100

    def __init__(self, port: int | None = None, deque_len: int | None = None) -> None:
        """Initialize the ZeroQueueProducer.

        Args:
        ----
            port: Port to connect to the consumer.
            deque_len: Max size of the message buffer.

        """
        self.port = port
        self.deque: deque[Any] = deque(maxlen=deque_len or self.DEQUE_LEN)
        self._reconnect()

    def __str__(self) -> str:
        """Representation of the ZeroQueueProducer with occupied port."""
        return f"{self.__class__.__name__}(port={self.port})"

    def _reconnect(self) -> None:
        """(Re)initialize context and REQ socket."""
        if self.socket_pub:
            self.stop()

        logger.debug("Reconnecting ZeroQueueProducer...")

        self.context: zmq.Context = zmq.Context()
        self.socket_pub: zmq.Context.socket = self.context.socket(zmq.REQ)
        self.socket_pub.setsockopt(zmq.LINGER, 0)

        if not self.port:
            self.port = self.socket_pub.bind_to_random_port("tcp://*")
            logger.info(f"ZeroQueueProducer bound to random port {self.port}")
        else:
            self.socket_pub.connect(f"tcp://localhost:{self.port}")
            logger.info(f"ZeroQueueProducer connected to port {self.port}")

        self.poller: zmq.Poller = zmq.Poller()
        self.poller.register(self.socket_pub, zmq.POLLIN)

    def stop(self) -> None:
        """Close sockets and terminate context."""
        logger.debug("Stopping ZeroQueueProducer...")
        try:
            if self.socket_pub:
                self.socket_pub.close()
        except zmq.ZMQError as e:
            logger.warning(f"Error while closing socket: {e}")
        try:
            if self.context:
                self.context.term()
        except zmq.ZMQError as e:
            logger.warning(f"Error while terminating context: {e}")
        self.socket_pub = None
        self.context = None

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.stop()

    def _put_from_deque(self, timeout: int = TIMEOUT_MS) -> bool:
        """Try to send the first item from the buffer.

        Returns True if the message was sent and ACK received.
        """
        item = self.deque[0]
        logger.debug("Attempting to send item from deque...")

        try:
            self.socket_pub.send_pyobj(item)
            if self.socket_pub in dict(self.poller.poll(timeout=timeout)):
                self.socket_pub.recv()
                self.deque.popleft()
                logger.debug("Item sent and acknowledged.")
                return True
        except queue.Empty:
            logger.debug("Queue is empty, cannot send item.")
        except zmq.ZMQError:
            logger.exception("ZMQ send/recv failed")

        logger.debug("Failed to send item from deque.")
        return False

    def put(self, item: Any, timeout: float | None = None) -> None:
        """Put an item in the buffer and attempt to send it.

        Retries once on failure by reconnecting.
        """
        self.deque.append(item)
        timeout_millis = int(timeout * 1000) if timeout else None

        while self.deque:
            success = self._put_from_deque(timeout=timeout_millis or self.TIMEOUT_MS)
            if not success:
                logger.warning("Send failed. Reconnecting and retrying...")
                self._reconnect()
                continue
            break

    def put_nowait(self, item: Any) -> None:
        """Send a message immediately without blocking.

        Raises queue.Full if ACK is not received.
        """
        try:
            self.socket_pub.send_pyobj(item)
            if self.socket_pub in dict(self.poller.poll(timeout=self.TIMEOUT_MS // 5 + 1)):
                self.socket_pub.recv()
                logger.debug("Message sent immediately.")
                return
        except zmq.ZMQError:
            logger.exception("Send failed in put_nowait")

        logger.warning("Immediate send failed, buffering item and reconnecting.")
        self.deque.appendleft(item)
        self._reconnect()
        msg = "Message not acknowledged in put_nowait."
        raise queue.Full(msg)

    def get(self, timeout: float | None = None) -> Any:
        """Gots a message (not implemented for producer)."""
        raise NotImplementedError

    def get_nowait(self) -> Any:
        """Gots a message without waiting (not implemented for producer)."""
        raise NotImplementedError
