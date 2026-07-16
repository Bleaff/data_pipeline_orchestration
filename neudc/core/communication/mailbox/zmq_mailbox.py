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
from queue import Empty, Full, Queue
from typing import Any

from zmq.error import ZMQError

from neudc.core.base.base_mailbox import BaseMailbox
from neudc.core.communication.messaging import codec
from neudc.core.communication.messaging.types import Batch, Frame
from neudc.core.communication.zero_queue import ZeroQueuePub, ZeroQueueSub
from neudc.core.communication.zero_queue.zmq_state import ZeroQueueConnectionType
from neudc.utils import LOGGER


class ZMQMailbox(BaseMailbox[dict]):
    """ZeroMQ-based mailbox implementation."""

    def __init__(
        self,
        *,
        message_queue_size: int = 20,
        name: str = "ZMQMailbox",
        _join_timeout: float = 0.1,
    ) -> ZMQMailbox:
        """Initialize the ZeroMQ mailbox."""
        self.pub_sockets: dict[int, ZeroQueuePub] = {}
        self.sub_queue: ZeroQueueSub = ZeroQueueSub()
        self.consume_port: int = self.sub_queue.port
        self._message_queue: Queue = Queue(maxsize=message_queue_size)
        self._running = False
        self._thread: threading.Thread | None = None
        self._join_timeout = _join_timeout
        self.name = name

        self.start()

    def _receiver_loop(self) -> None:
        """Thread loop for receiving messages from the ZeroMQ subscriber."""
        while self._running:
            message = self.sub_queue.get(timeout=self._join_timeout)
            if message is None:
                continue
            try:
                # Block with periodic wake-ups so a full queue applies backpressure
                # instead of silently dropping the message (or injecting a bogus one).
                while self._running:
                    try:
                        self._message_queue.put(message, timeout=self._join_timeout)
                        break
                    except Full:
                        continue
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"[RECV] ← message of type {type(message)}")
            except Exception as e:
                LOGGER.exception("Error in receiver loop", exc_info=e)

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

    def send(self, message: Batch | Frame) -> None:
        """Send a message to every downstream edge.

        Each frame is serialized once and the same bytes are reused for all
        publishers, so fan-out to N nodes does not pay N serializations.
        """
        if isinstance(message, Batch):
            for frame in message:
                raw = codec.dumps(frame)
                for pub_socket in self.pub_sockets.values():
                    pub_socket.put_bytes(raw)
        else:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(f"[{self.name}][SEND] → message with type {type(message)}")
            raw = codec.dumps(message)
            for pub_socket in self.pub_sockets.values():
                pub_socket.put_bytes(raw)

    def receive(self, timeout: float | None = None) -> dict:
        """Receive a message from the mailbox."""
        try:
            message = self._message_queue.get(timeout=timeout if timeout else 0.1)
            if LOGGER.isEnabledFor(logging.DEBUG):
                frame_id = getattr(message, "frame_id", None)
                LOGGER.debug(f"[{self.name}][RECV] ← message {type(message).__name__} frame_id={frame_id}")
        except Empty:
            message = None
        except ZMQError:
            message = None
        return message

    def add_publisher(self, port: int) -> None:
        """Connect a publisher to the mailbox. This method is not thread-safe.

        ``ZeroQueuePub`` performs a one-time "slow joiner" settle on connect, so
        the subscription has propagated by the time this returns.
        """
        self.pub_sockets[port] = ZeroQueuePub(port=port)
        LOGGER.debug(f"[{self.name}][Added publisher] → port:{port}")

    def remove_publisher(self, port: int) -> None:
        """Remove a publisher from the mailbox."""
        rm_pub = self.pub_sockets.pop(port)
        LOGGER.debug(f"[{self.name}][Removed publisher] → {rm_pub}")

    def __getstate__(self) -> dict[str, Any]:
        """Return a picklable snapshot of the mailbox.

        This is a pure serializer: it must not mutate ``self`` (pickling an object
        should never stop it). Releasing the bound consume port for a hand-off to a
        child process is done explicitly by the caller (see BaseProcessNode.__init__).
        """
        LOGGER.debug(f"[{self.name}][GET STATE] → {self.name}")
        state = self.__dict__.copy()
        # Drop the non-picklable runtime pieces; sizes/ports are enough to rebuild.
        state["_thread"] = None
        state["_running"] = False
        state["logger"] = None  # Avoid pickling the logger
        state["message_queue_size"] = self._message_queue.qsize()  # Store size instead of the queue itself
        state["pub_sockets"] = list(self.pub_sockets.keys())
        state["consume_port"] = self.consume_port
        state["sub_queue"] = None
        state["_message_queue"] = None  # Avoid pickling the queue itself
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the mailbox from a saved state."""
        self.name = f'PICKLED|{state.get("name", "ZMQMailbox")}'
        LOGGER.debug(f"[{self.name}][SET STATE] → {self.name}")
        self._message_queue = Queue(state["message_queue_size"])

        self.pub_sockets = {port: ZeroQueuePub(port=port) for port in state["pub_sockets"]}
        self.sub_queue = ZeroQueueSub(port=state["consume_port"], contype=ZeroQueueConnectionType.BIND)
        self.consume_port = state["consume_port"]
        self._message_queue = Queue(state["message_queue_size"])

        self._running = True
        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._join_timeout = state["_join_timeout"]
        self._thread.start()

    @staticmethod
    def from_state(cls: type[ZMQMailbox], state: dict[str, Any]) -> ZMQMailbox:
        mailbox = cls()
        mailbox.stop()
        mailbox.__setstate__(state)
        return mailbox
