"""ZeroQueue abstraction over producer/consumer ZeroMQ messaging.

Provides a simple interface for sending and receiving messages using the
ZeroMQ library. The class can be used as either a producer or consumer,
depending on the value of the 'is_producer' argument.

The producer creates a ZeroQueueProducer instance which connects to a port
and sends messages to the consumer. The consumer creates a ZeroQueueConsumer
instance which binds to a port and receives messages from the producer. The
consumer can also connect to an existing port if the producer is running on
another machine.

The class provides a simple interface for sending and receiving messages,
hiding the underlying details of the ZeroMQ library. The interface is
thread-safe, meaning that multiple threads can call the same methods
concurrently without worrying about race conditions.

"""

from __future__ import annotations

import logging
from typing import Any

import zmq

from core.base.base_queue import QueueLike

logger = logging.getLogger(__name__)


class ZeroQueue(QueueLike):
    """PUB/SUB-based queue implementation using ZeroMQ.

    Suitable for inter-process message passing on a single machine.
    """

    def __init__(self, port: int = -1) -> None:
        """Initialize the ZeroQueue.

        Args:
        ----
            port (Optional[int]): Port for PUB/SUB communication. If None, a random free port is chosen.

        """
        self._port: int = port  # type: ignore[assignment]
        self.context = zmq.Context()
        self.socket_pub = self.context.socket(zmq.PUB)
        if self.port == -1:
            self._port = self.socket_pub.bind_to_random_port("tcp://*")
        else:
            try:
                self.socket_pub.bind(f"tcp://*:{self.port}")
            except zmq.ZMQError as e:
                if e.errno == zmq.EADDRINUSE:
                    logger.exception(f"Port {self.port} is already in use.")
                    self._port = self.socket_pub.bind_to_random_port("tcp://*")
                else:
                    logger.exception(f"Failed to bind to port {self.port}")

        self.socket_sub = self.context.socket(zmq.SUB)
        self.socket_sub.connect(f"tcp://localhost:{self.port}")
        self.socket_sub.subscribe("")

        self.poller = zmq.Poller()
        self.poller.register(self.socket_sub, zmq.POLLIN)

    def __str__(self) -> str:
        """Magic methods for string representation of queue."""
        return f"{self.__class__.__name__}(port={self.port})"

    @property
    def port(self) -> int:
        """Get the port number."""
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        """Set the port number."""
        if self._port != value:
            self._port = value
            self.socket_pub.bind(f"tcp://*:{self.port}")
            self.socket_sub.connect(f"tcp://localhost:{self.port}")
            self.socket_sub.subscribe("")
            self.poller = zmq.Poller()
            self.poller.register(self.socket_sub, zmq.POLLIN)

    def get(self, timeout: float | None = None) -> Any | None:
        """Receive an item from the queue with timeout.

        Args:
        ----
            timeout (Optional[float]): Timeout in seconds.

        Returns:
        -------
            Optional[Any]: Received message, or None if timeout expired.

        """
        timeout_millis = int(timeout * 1000) if timeout else None
        if self.socket_sub in dict(self.poller.poll(timeout=timeout_millis)):
            return self.get_nowait()
        return None

    def get_nowait(self) -> Any:
        """Receive a message without waiting."""
        return self.socket_sub.recv_pyobj(zmq.NOBLOCK)

    def put(self, item: Any) -> None:
        """Send a message.

        Args:
        ----
            item (Any): Object to send.

        """
        self.socket_pub.send_pyobj(item)

    def put_nowait(self, item: Any) -> None:
        """Send a message without blocking.

        Args:
        ----
            item (Any): Object to send.

        """
        self.socket_pub.send_pyobj(item, zmq.NOBLOCK)

    def _after_fork(self) -> None:
        """Reset sockets after fork (Unix only)."""
        self._reset()

    def _reset(self) -> None:
        """Recreate PUB/SUB sockets."""
        self.socket_pub = self.context.socket(zmq.PUB)
        self.socket_pub.connect(f"tcp://*:{self.port}")

        self.socket_sub = self.context.socket(zmq.SUB)
        self.socket_sub.connect(f"tcp://localhost:{self.port}")
        self.socket_sub.subscribe("")

        self.poller = zmq.Poller()
