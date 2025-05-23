"""ZeroQueuePub class for publishing data to another ZeroQueueSubscriber.

This class extends the ZeroQueue class to publish data to a ZeroQueueSubscriber.
"""

from .zero_queue import ZeroQueue


class ZeroQueuePub(ZeroQueue):
    """Publish data to another ZeroQueueSubscriber.

    Extends the ZeroQueue class to publish data to a ZeroQueueSubscriber.

    Args:
    ----
        port (int): Port for PUB/SUB communication.
        mode (ZeroQueueMode): Mode of the queue (PUB(publisher) or SUB(subscriber)).

    """
