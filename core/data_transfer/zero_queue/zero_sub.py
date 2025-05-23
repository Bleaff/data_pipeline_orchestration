"""ZeroQueueSub class for receiving data from another ZeroQueuePublisher.

This class extends the ZeroQueue class to receive data from a ZeroQueuePublisher.
"""

from core.data_transfer.zero_queue.zero_queue import ZeroQueue
from core.data_transfer.zero_queue.zmq_state import ZeroQueueConnectionType
from core.data_transfer.zero_queue.zmq_state import ZeroQueueMode

class ZeroQueueSub(ZeroQueue):
    """ZeroQueueSub class for receiving data from another ZeroQueuePublisher.

    This class extends the ZeroQueue class to receive data from a ZeroQueuePublisher.

    Args:
    ----
            port (int): Port for PUB/SUB communication.
            mode (ZeroQueueMode): Mode of the queue (PUB(publisher) or SUB(subscriber)).

    """
    def __init__(self, port: int = -1, contype: ZeroQueueConnectionType = ZeroQueueConnectionType.BIND) -> None:
        super().__init__(port, mode=ZeroQueueMode.SUB, contype=contype)
        sockets:dict = {}
        