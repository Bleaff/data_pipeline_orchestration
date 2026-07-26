"""Tests for pickling semantics of ZMQMailbox (Stage 2).

__getstate__ must be a pure serializer (no side effects), and from_state must
rebuild a mailbox that can actually receive on the original consume port.
"""

from __future__ import annotations

import time

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox


def test_getstate_does_not_stop_the_mailbox() -> None:
    mailbox = ZMQMailbox()
    assert mailbox._running is True

    state = mailbox.__getstate__()

    # Pickling must not mutate/stop the object.
    assert mailbox._running is True
    assert isinstance(state, dict)
    assert state["consume_port"] == mailbox.consume_port
    assert state["pub_sockets"] == []

    mailbox.stop()


def test_from_state_rebuilds_a_working_mailbox() -> None:
    receiver = ZMQMailbox()
    port = receiver.consume_port
    state = receiver.__getstate__()
    receiver.stop()  # explicit hand-off: free the bound port for the rebuilt mailbox

    rebuilt = ZMQMailbox.from_state(state)
    assert rebuilt.consume_port == port

    sender = ZMQMailbox()
    sender.add_publisher(rebuilt.consume_port)
    sender.send({"x": 1})

    got = None
    deadline = time.time() + 5
    while got is None and time.time() < deadline:
        got = rebuilt.receive(timeout=0.1)

    assert got == {"x": 1}
    sender.stop()
    rebuilt.stop()
