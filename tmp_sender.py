import random
import uuid
from threading import Event, Thread
from time import sleep

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox


def worker(
    mailbox: ZMQMailbox, stop_event: Event, sleep_time: float = 0.01, max_wait: float = 10, batch_size: int = 10,
) -> None:
    while not stop_event.is_set():
        object_send = str(uuid.uuid4().hex)[:5]
        mailbox.send(object_send)
        sleep(random.uniform(0.1, 2))


def main(port) -> None:
    mailbox = ZMQMailbox(name="BatchSender")
    mailbox.add_publisher(port)
    stop_event = Event()
    thread = Thread(target=worker, args=(mailbox, stop_event))
    thread.start()
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        thread.join(timeout=1)


if __name__ == "__main__":
    port = int(input("Enter the port number for the mailbox (default 54074): ") or 54074)
    main(port)
    # This will run the main function to start the sender
    # and allow it to send messages until interrupted.
    # The sender will send random strings to the mailbox.
    # You can adjust the sleep time and batch size as needed.
    # The mailbox will handle the messages sent by the worker.
    # The worker will send a random string to the mailbox every 0.001 to 0.01 seconds.
    # The main function will run until interrupted by a keyboard interrupt (Ctrl+C).
    # The worker will stop gracefully when the stop event is set.
