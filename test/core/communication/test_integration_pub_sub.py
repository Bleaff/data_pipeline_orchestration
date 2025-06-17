import logging
import threading
import time

from neudc.core.communication.zero_queue import ZeroQueuePub, ZeroQueueSub

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_pub_to_bound_sub() -> None:
    message = {"type": "ping", "payload": "hello"}
    received = []

    sub = ZeroQueueSub()
    port = sub.port
    logger.info(f"ZeroQueueSub bound to port {port}")

    def run_sub(timeout=2) -> None:
        time_start = time.time()
        logger.info("ZeroQueueSub started")
        while time.time() - time_start < timeout:
            msg = sub.get(timeout=0.1)
            if msg:
                logger.info(f"ZeroQueueSub received message: {msg}")
                received.append(msg)
                break

    def run_pub() -> None:
        pub = ZeroQueuePub(port=port)
        logger.info("ZeroQueuePub started")
        time.time()

        pub.put(message)
        logger.info(f"ZeroQueuePub sent message: {message}")

    thread_sub = threading.Thread(target=run_sub)
    thread_pub = threading.Thread(target=run_pub)

    thread_sub.start()
    thread_pub.start()

    thread_sub.join()
    thread_pub.join()

    assert received == [message]


if __name__ == "__main__":
    test_pub_to_bound_sub()
