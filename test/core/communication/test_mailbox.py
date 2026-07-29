from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox


def test_add_publisher_and_send() -> None:
    received = []
    m_1 = ZMQMailbox()
    m_2 = ZMQMailbox()
    port_1 = m_1.consume_port
    port_2 = m_2.consume_port

    m_1.add_publisher("m_2", port_2)
    m_2.add_publisher("m_1", port_1)

    m_1.send({"msg": "test"})
    received.append(m_2.receive())

    assert received
    assert received[0]["msg"] == "test"


def test_has_working_port_sharing() -> None:
    mailbox = ZMQMailbox()
    pub_2 = ZMQMailbox()
    pub_3 = ZMQMailbox()

    port = mailbox.consume_port
    pub_2.add_publisher("mailbox", port)
    pub_3.add_publisher("mailbox", port)

    pub_3.send({"msg": "test_1"})
    pub_2.send({"msg": "test_2"})

    import time

    received = []
    deadline = time.time() + 5
    while len(received) < 2 and time.time() < deadline:
        msg = mailbox.receive(timeout=0.1)
        if msg is not None:
            received.append(msg)

    # Two independent producers feed one consumer; PUSH/PULL fair-queues across them,
    # so the interleaving order is not defined. What matters is that both arrive.
    assert {frozenset(m.items()) for m in received} == {
        frozenset({"msg": "test_1"}.items()),
        frozenset({"msg": "test_2"}.items()),
    }


def test_remove_publisher() -> None:
    mailbox = ZMQMailbox()
    target = ZMQMailbox()

    mailbox.add_publisher("target", target.consume_port)
    assert "target" in mailbox.pub_sockets

    mailbox.remove_publisher("target")
    assert "target" not in mailbox.pub_sockets

    target.stop()
