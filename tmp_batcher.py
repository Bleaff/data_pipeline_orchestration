import logging
import queue
import threading
from time import sleep, time

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def worker(
    mailbox: ZMQMailbox,
    stop_event: threading.Event,
    threading_queue: queue.Queue,
    sleep_time: float = 0.01,
    max_wait: float = 10,
    batch_size: int = 10,
) -> None:
    while not stop_event.is_set():
        batch = []
        start = time()
        while batch_size > len(batch) and time() - start < max_wait:
            try:
                message = mailbox.receive()
                if message is not None:
                    batch.append(message)
            except Exception as e:
                logger.exception(f"Error receiving message: {e}")
                break
        if len(batch):
            logger.debug(f"Received batch of size {len(batch)} for {time() - start:.2f} seconds: {batch}")
            threading_queue.put(batch.copy())


def queue_watchdog(threading_queue: queue.Queue, stop_event: threading.Event, check_interval: float = 0.1) -> None:
    while not stop_event.is_set():
        if not threading_queue.empty():
            logger.debug(f"Threading queue size({threading_queue.qsize()}): {threading_queue.queue}")
            threading_queue.get()
        sleep(check_interval)


def main(stop_event: threading.Event):
    mailbox = ZMQMailbox(name="test_mailbox")
    threading_queue = queue.Queue()
    threads = []
    thread_worker = threading.Thread(target=worker, args=(mailbox, stop_event, threading_queue))
    thread_watchdog = threading.Thread(target=queue_watchdog, args=(threading_queue, stop_event, 2))
    threads.append(thread_worker)
    threads.append(thread_watchdog)
    thread_worker.start()
    thread_watchdog.start()
    return threads


if __name__ == "__main__":
    stop_event = threading.Event()
    threads = main(stop_event)
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=1)
