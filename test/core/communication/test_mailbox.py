from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox


def test_add_publisher_and_send() -> None:
    received = []
    m_1 = ZMQMailbox()
    m_2 = ZMQMailbox()
    port_1 = m_1.consume_port
    port_2 = m_2.consume_port

    m_1.add_publisher(port_2)
    m_2.add_publisher(port_1)

    m_1.send({"msg": "test"})
    received.append(m_2.receive())

    assert received
    assert received[0]["msg"] == "test"


def test_has_working_port_sharing() -> None:
    mailbox = ZMQMailbox()
    pub_2 = ZMQMailbox()
    pub_3 = ZMQMailbox()

    port = mailbox.consume_port
    pub_2.add_publisher(port)
    pub_3.add_publisher(port)  # overwrite test

    pub_3.send({"msg": "test_1"})
    pub_2.send({"msg": "test_2"})

    recieved = [mailbox.receive() for _ in range(2)]

    assert recieved == [{"msg": "test_1"}, {"msg": "test_2"}]


def test_remove_publisher() -> None:
    mailbox = ZMQMailbox()
    port = mailbox.consume_port

    mailbox.add_publisher(port)
    assert port in mailbox.pub_sockets

    mailbox.remove_publisher(port)
    assert port not in mailbox.pub_sockets
