"""ZeroQueuePub class for publishing data to another ZeroQueueSubscriber.

This class extends the ZeroQueue class to publish data to a ZeroQueueSubscriber.
"""

from zero_queue import ZeroQueue
from zmq_state import ZeroQueueConnectionType
from zmq_state import ZeroQueueMode


class ZeroQueuePub(ZeroQueue):
    """Publish data to another ZeroQueueSubscriber.

    Extends the ZeroQueue class to publish data to a ZeroQueueSubscriber.
    """
    def __init__(self, port = -1, contype = ZeroQueueConnectionType.CONNECT):
        super().__init__(port, mode=ZeroQueueMode.PUB, contype=contype)
    
    def get(self, timeout = None):
        raise NotImplementedError("ZeroQueuePub does not support get() method.")
    def get_nowait(self):
        raise NotImplementedError("ZeroQueuePub does not support get_nowait() method.")
        
if __name__ == "__main__":
    import logging
    import time

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    port = int(input("Enter port: "))
    # Example usage
    pub = ZeroQueuePub(port=port)
    logger.info(f"ZeroQueuePub bound to port {pub.port}")

    while True:
        message = input("Enter message to send: ")
        ret = pub.put(message)
        time.sleep(1)