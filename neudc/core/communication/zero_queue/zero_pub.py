"""ZeroQueuePub class for publishing data to another ZeroQueueSubscriber.

This class extends the ZeroQueue class to publish data to a ZeroQueueSubscriber.
"""

from __future__ import annotations

from typing import NoReturn

from neudc.core.communication.zero_queue.zero_queue import ZeroQueue
from neudc.core.communication.zero_queue.zmq_state import ZeroQueueConnectionType, ZeroQueueMode


class ZeroQueuePub(ZeroQueue):
    """Publish data to another ZeroQueueSubscriber.

    Extends the ZeroQueue class to publish data to a ZeroQueueSubscriber.
    """

    def __init__(self, port: int = -1, contype: ZeroQueueConnectionType = ZeroQueueConnectionType.CONNECT) -> None:
        """Initialize the ZeroQueuePub.

        Args:
        ----
            port (int): Port for PUB/SUB communication. If -1, a random free port is chosen.
            contype (ZeroQueueConnectionType): Connection type (bind or connect). Default is CONNECT.

        """
        super().__init__(port, mode=ZeroQueueMode.PUB, contype=contype)

    def get(self, timeout: float | None = None) -> NoReturn:
        """Protect get method not supported for ZeroQueuePub."""
        msg = "ZeroQueuePub does not support get() method."
        raise NotImplementedError(msg)

    def get_nowait(self) -> NoReturn:
        """Protect get_nowait method not supported for ZeroQueuePub."""
        msg = "ZeroQueuePub does not support get_nowait() method."
        raise NotImplementedError(msg)
