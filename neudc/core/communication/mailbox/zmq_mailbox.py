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

import zmq
from zmq.error import ZMQError

from neudc.core.base.base_mailbox import BaseMailbox
from neudc.core.communication.messaging import codec
from neudc.core.communication.messaging.types import BaseMessage, Batch
from neudc.core.communication.zero_queue import ZeroQueuePub, ZeroQueueSub
from neudc.core.communication.zero_queue.zmq_state import ZeroQueueConnectionType
from neudc.core.observability.metrics import QUEUE_DEPTH, QUEUE_HWM
from neudc.utils import LOGGER


class ZMQMailbox(BaseMailbox[BaseMessage]):
    """ZeroMQ-based mailbox implementation.

    Typed by :class:`BaseMessage`, not by :class:`~neudc.core.communication.messaging.types.Frame`:
    the transport carries any message subclass, so non-CV payloads need no change here.
    """

    def __init__(
        self,
        *,
        message_queue_size: int = 20,
        name: str = "ZMQMailbox",
        _join_timeout: float = 0.1,
    ) -> None:
        """Initialize the ZeroMQ mailbox."""
        # Keyed by the *logical* downstream target id, not by physical port: a replica
        # group (#15) is one entry with one ZeroQueuePub connected to every replica's
        # port, so fan-out to N node types still means len(pub_sockets) == N, which is
        # what the SHM single-consumer invariant below relies on.
        self.pub_sockets: dict[str, ZeroQueuePub] = {}
        self.sub_queue: ZeroQueueSub = ZeroQueueSub()
        self.consume_port: int = self.sub_queue.port
        self._message_queue: Queue = Queue(maxsize=message_queue_size)
        self._running = False
        self._thread: threading.Thread | None = None
        self._join_timeout = _join_timeout
        self.name = name

        # Informational: the ZMQ default (1000) unless a future config path overrides it.
        # Read back from the socket rather than hardcoded, so it stays truthful either way.
        assert self.sub_queue.socket_sub is not None
        QUEUE_HWM.labels(node=self.name).set(int(self.sub_queue.socket_sub.getsockopt(zmq.RCVHWM)))

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
                QUEUE_DEPTH.labels(node=self.name).set(self._message_queue.qsize())
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"[RECV] ← message of type {type(message)}")
            except Exception as e:  # noqa: BLE001 -- top-level receiver thread loop must never crash.
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

    def send(self, message: BaseMessage | Batch) -> None:
        """Send a message to every downstream edge.

        A :class:`Batch` is unrolled and its items are sent one by one; any other
        :class:`BaseMessage` goes out as-is. Each message is serialized once and the
        same bytes are reused for all publishers, so fan-out to N nodes does not pay
        N serializations.

        The shared-memory buffer transport (opt-in via ``NEUDC_SHM_IMAGES``) is used
        only on **single-consumer** edges: a segment is owned and unlinked by exactly
        one consumer, so fan-out (or no downstream) falls back to the in-band pickle to
        avoid a race on unlink. See :mod:`neudc.core.communication.messaging.codec`.
        """
        use_shm = codec.SHM_ENABLED and len(self.pub_sockets) == 1
        if isinstance(message, Batch):
            for item in message:
                raw = codec.dumps(item, use_shm=use_shm)
                for pub_socket in self.pub_sockets.values():
                    pub_socket.put_bytes(raw)
        else:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(f"[{self.name}][SEND] → message with type {type(message)}")
            raw = codec.dumps(message, use_shm=use_shm)
            for pub_socket in self.pub_sockets.values():
                pub_socket.put_bytes(raw)

    def receive(self, timeout: float | None = None) -> BaseMessage | None:
        """Receive a message from the mailbox, or None if nothing arrived in time."""
        try:
            message = self._message_queue.get(timeout=timeout or 0.1)
            QUEUE_DEPTH.labels(node=self.name).set(self._message_queue.qsize())
            if LOGGER.isEnabledFor(logging.DEBUG):
                # frame_id is CV-specific; absent on other payloads.
                frame_id = getattr(message, "frame_id", None)
                LOGGER.debug(f"[{self.name}][RECV] ← message {type(message).__name__} frame_id={frame_id}")
        except Empty:
            message = None
        except ZMQError:
            message = None
        return message

    def add_publisher(self, target_id: str, ports: int | list[int]) -> None:
        """Connect a publisher to one logical downstream target. Not thread-safe.

        ``target_id`` identifies the *node* being published to (e.g. its config id),
        not a single port: when that target has several replicas (#15), pass all of
        their ports and a single ``ZeroQueuePub`` is connected to every one of them,
        so ZMQ round-robins messages across the replica group. Calling this again
        with a ``target_id`` that is already registered grows the existing
        ``ZeroQueuePub``'s connections instead of replacing it (used by autoscale
        scale-up to add a freshly started replica's port).

        ``ZeroQueuePub`` performs a one-time "slow joiner" settle on connect, so
        the subscription has propagated by the time this returns.
        """
        port_list = [ports] if isinstance(ports, int) else list(ports)
        if target_id in self.pub_sockets:
            for port in port_list:
                self.pub_sockets[target_id].connect_additional(port)
        else:
            self.pub_sockets[target_id] = ZeroQueuePub(ports=port_list)
        LOGGER.debug(f"[{self.name}][Added publisher] → target:{target_id} ports:{port_list}")

    def remove_publisher(self, target_id: str) -> None:
        """Remove a publisher (an entire replica group) from the mailbox."""
        rm_pub = self.pub_sockets.pop(target_id)
        rm_pub.stop()
        LOGGER.debug(f"[{self.name}][Removed publisher] → {rm_pub}")

    @property
    def queue_depth(self) -> int:
        """Current depth of the internal inbound message queue.

        Public accessor so the autoscaler (#15) can read live queue depth without
        reaching into the private ``_message_queue`` from outside this module.
        """
        return self._message_queue.qsize()

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
        state["pub_sockets"] = {target_id: pub.ports for target_id, pub in self.pub_sockets.items()}
        state["consume_port"] = self.consume_port
        state["sub_queue"] = None
        state["_message_queue"] = None  # Avoid pickling the queue itself
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the mailbox from a saved state."""
        self.name = f'PICKLED|{state.get("name", "ZMQMailbox")}'
        LOGGER.debug(f"[{self.name}][SET STATE] → {self.name}")
        self._message_queue = Queue(state["message_queue_size"])

        self.pub_sockets = {target_id: ZeroQueuePub(ports=ports) for target_id, ports in state["pub_sockets"].items()}
        self.sub_queue = ZeroQueueSub(port=state["consume_port"], contype=ZeroQueueConnectionType.BIND)
        self.consume_port = state["consume_port"]
        self._message_queue = Queue(state["message_queue_size"])

        self._running = True
        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._join_timeout = state["_join_timeout"]
        self._thread.start()

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ZMQMailbox:
        """Rebuild a mailbox from a state dict produced by :meth:`__getstate__`."""
        mailbox = cls()
        mailbox.stop()
        mailbox.__setstate__(state)
        return mailbox
