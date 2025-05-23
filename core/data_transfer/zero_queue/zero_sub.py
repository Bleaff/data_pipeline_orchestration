"""ZeroQueueSub class for receiving data from another ZeroQueuePublisher.

This class extends the ZeroQueue class to receive data from a ZeroQueuePublisher.
"""

from .zero_queue import ZeroQueue


class ZeroQueueSub(ZeroQueue):
    """ZeroQueueSub class for receiving data from another ZeroQueuePublisher.

    This class extends the ZeroQueue class to receive data from a ZeroQueuePublisher.

    Args:
    ----
            port (int): Port for PUB/SUB communication.
            mode (ZeroQueueMode): Mode of the queue (PUB(publisher) or SUB(subscriber)).

    """
