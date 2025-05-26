import threading
import time

from core.data_transfer.mailbox.zmq_mailbox import ZMQMailbox


# def test_add_publisher_and_send() -> None:
#     m_1 = ZMQMailbox()
#     m_2 = ZMQMailbox()
#     port_m1 = m_1.consume_port
#     port_m2 = m_2.consume_port
    
#     m_1.add_publisher(port_m2)
#     m_2.add_publisher(port_m1)


#     received = []

#     def reader() -> None:
#         time.sleep(0.1)
#         msg = m_1.receive()
#         received.append(msg)

#     t = threading.Thread(target=reader)
#     t.start()

#     time.sleep(0.1)  # дать сабскрайберу время
#     m_2.send({"msg": "test"})

#     t.join(timeout=1.0)

#     assert received
#     assert received[0]["msg"] == "test"

if __name__ == "__main__":
    # m_1 = ZMQMailbox()
    # m_2 = ZMQMailbox()
    # port_1 = m_1.consume_port
    # port_2 = m_2.consume_port

    # m_1.add_publisher(port_2)
    # m_2.add_publisher(port_1)
    
    # m_1.send({"msg": "test"})
    # msg = m_2.receive()
    # print(msg)
    mailbox = ZMQMailbox()
    port = mailbox.consume_port

    mailbox.add_publisher(port)
    assert port in mailbox.pub_sockets

    mailbox.remove_publisher(port)
    assert port not in mailbox.pub_sockets