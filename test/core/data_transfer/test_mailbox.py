import threading
import time

from core.data_transfer.mailbox.zmq_mailbox import ZMQMailbox


def test_add_publisher_and_send() -> None:
    mailbox = ZMQMailbox()
    port = mailbox.consume_port

    received = []

    def reader() -> None:
        msg = mailbox.receive()
        received.append(msg)

    t = threading.Thread(target=reader)
    t.start()

    time.sleep(0.1)  # дать сабскрайберу время
    mailbox.add_publisher(port)
    mailbox.send({"msg": "test"})

    t.join(timeout=1.0)

    assert received
    assert received[0]["msg"] == "test"


def test_receive_message() -> None:
    mailbox = ZMQMailbox()
    port = mailbox.consume_port
    mailbox.add_publisher(port)

    time.sleep(0.1)
    mailbox.send({"event": "ping"})
    msg = mailbox.receive()

    assert msg["event"] == "ping"


def test_has_working_port_sharing() -> None:
    mailbox = ZMQMailbox()
    port = mailbox.consume_port
    mailbox.add_publisher(port)
    mailbox.add_publisher(port)  # overwrite test

    time.sleep(0.1)
    mailbox.send({"data": 123})
    msg = mailbox.receive()

    assert msg == {"data": 123}


def test_remove_publisher() -> None:
    mailbox = ZMQMailbox()
    port = mailbox.consume_port

    mailbox.add_publisher(port)
    assert port in mailbox.pub_sockets

    mailbox.remove_publisher(port)
    assert port not in mailbox.pub_sockets
