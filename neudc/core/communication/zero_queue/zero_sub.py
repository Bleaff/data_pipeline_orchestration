"""ZeroQueueSub class for receiving data from another ZeroQueuePublisher.

This class extends the ZeroQueue class to receive data from a ZeroQueuePublisher.
"""

from typing import Any, NoReturn

from neudc.core.communication.zero_queue.zero_queue import ZeroQueue
from neudc.core.communication.zero_queue.zmq_state import ZeroQueueConnectionType, ZeroQueueMode


class ZeroQueueSub(ZeroQueue):
    """ZeroQueueSub class for receiving data from another ZeroQueuePublisher.

    This class extends the ZeroQueue class to receive data from a ZeroQueuePublisher.

    Args:
    ----
            port (int): Port for PUB/SUB communication.
            mode (ZeroQueueMode): Mode of the queue (PUB(publisher) or SUB(subscriber)).

    """

    def __init__(
        self,
        port: int = -1,
        contype: ZeroQueueConnectionType = ZeroQueueConnectionType.BIND,
    ) -> "ZeroQueueSub":
        """Initialize the ZeroQueueSub.

        This class is used to receive data from a ZeroQueuePublisher. It always binds to a random port.

        """
        super().__init__(port=port, mode=ZeroQueueMode.SUB, contype=contype)

    def put(self, item: Any) -> NoReturn:
        """Protect put method not supported for ZeroQueueSub."""
        msg = f"ZeroQueueSub does not support put() method. Cannot put item typeof {type(item)}."
        raise NotImplementedError(msg)

    def put_nowait(self, item: Any) -> NoReturn:
        """Protect put_nowait method not supported for ZeroQueueSub."""
        msg = f"ZeroQueueSub does not support put_nowait() method. Cannot put item typeof {type(item)}."
        raise NotImplementedError(msg)
