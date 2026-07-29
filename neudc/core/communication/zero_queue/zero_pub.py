"""ZeroQueuePub class for publishing data to another ZeroQueueSubscriber.

This class extends the ZeroQueue class to publish data to a ZeroQueueSubscriber.
"""

from __future__ import annotations

from typing import NoReturn

from neudc.core.communication.zero_queue.zero_queue import ZeroQueue
from neudc.core.communication.zero_queue.zmq_state import ZeroQueueConnectionType, ZeroQueueMode
from neudc.utils import LOGGER


class ZeroQueuePub(ZeroQueue):
    """Publish data to another ZeroQueueSubscriber.

    Extends the ZeroQueue class to publish data to a ZeroQueueSubscriber.

    A single PUSH socket may be connected to more than one PULL peer (#15, worker
    replicas): ZeroMQ round-robins fairly among all currently-connected peers, so
    connecting one ``ZeroQueuePub`` to N replica ports is exactly "one PUSH fans out
    to N PULL workers" with no extra code on the sending side.
    """

    def __init__(
        self,
        port: int = -1,
        ports: list[int] | None = None,
        contype: ZeroQueueConnectionType = ZeroQueueConnectionType.CONNECT,
    ) -> None:
        """Initialize the ZeroQueuePub.

        Args:
        ----
            port (int): Port for PUB/SUB communication. If -1, a random free port is chosen.
                Ignored when ``ports`` is given. Kept for backward compatibility with the
                single-port constructor used throughout the codebase.
            ports (list[int] | None): All ports to connect to, for fan-out to several PULL
                peers (e.g. a replica group). When given, the socket connects to every port
                in the list, round-robin. Mutually exclusive with ``port`` in intent, but
                ``port`` is simply ignored if ``ports`` is provided.
            contype (ZeroQueueConnectionType): Connection type (bind or connect). Default is CONNECT.

        """
        all_ports = list(ports) if ports else ([port] if port != -1 else [])
        if not all_ports:
            msg = "ZeroQueuePub requires at least one port to connect to."
            raise ValueError(msg)
        super().__init__(all_ports[0], mode=ZeroQueueMode.PUB, contype=contype)
        self._connected_ports: list[int] = [all_ports[0]]
        for extra_port in all_ports[1:]:
            self.connect_additional(extra_port)

    @property
    def ports(self) -> list[int]:
        """All ports this PUSH socket is currently connected to."""
        return list(self._connected_ports)

    def connect_additional(self, port: int) -> None:
        """Connect the existing PUSH socket to one more PULL port.

        Used both to build an initial replica group and to grow one at runtime
        (autoscale scale-up, #15) without tearing down the socket or losing
        in-flight messages to already-connected peers.
        """
        if port in self._connected_ports:
            return
        assert self.socket_pub is not None
        self.socket_pub.connect(f"tcp://localhost:{port}")
        self._connected_ports.append(port)
        LOGGER.debug(f"ZeroQueuePub connected additional port {port}")

    def disconnect_additional(self, port: int) -> None:
        """Disconnect the PUSH socket from one previously connected port (scale-down, #15).

        Raises if ``port`` is the last remaining connection: stop() the queue instead
        of disconnecting the only peer.
        """
        if port not in self._connected_ports:
            msg = f"Port {port} is not connected."
            raise ValueError(msg)
        if len(self._connected_ports) <= 1:
            msg = "Cannot disconnect the only remaining port; stop() the queue instead."
            raise ValueError(msg)
        assert self.socket_pub is not None
        self.socket_pub.disconnect(f"tcp://localhost:{port}")
        self._connected_ports.remove(port)
        LOGGER.debug(f"ZeroQueuePub disconnected port {port}")

    def get(self, timeout: float | None = None) -> NoReturn:
        """Protect get method not supported for ZeroQueuePub."""
        msg = "ZeroQueuePub does not support get() method."
        raise NotImplementedError(msg)

    def get_nowait(self) -> NoReturn:
        """Protect get_nowait method not supported for ZeroQueuePub."""
        msg = "ZeroQueuePub does not support get_nowait() method."
        raise NotImplementedError(msg)
