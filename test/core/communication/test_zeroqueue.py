"""Tests for ZeroQueue, ZeroQueuePub, and ZeroQueueSub classes.

Using mocking, NotImplementedError is raised for put, put_nowait, get, and get_nowait methods. Checks timeout behavior.
"""

from unittest.mock import MagicMock

import pytest
import zmq

from neudc.core.communication.messaging import codec
from neudc.core.communication.zero_queue.zero_pub import ZeroQueuePub
from neudc.core.communication.zero_queue.zero_queue import ZeroQueue
from neudc.core.communication.zero_queue.zero_sub import ZeroQueueSub
from neudc.core.communication.zero_queue.zmq_state import ZeroQueueConnectionType, ZeroQueueMode


@pytest.fixture()
def mock_zmq(monkeypatch):
    context_mock = MagicMock()
    socket_mock = MagicMock()
    poller_mock = MagicMock()

    context_mock.socket.return_value = socket_mock
    monkeypatch.setattr("neudc.core.communication.zero_queue.zero_queue.zmq.Context", lambda: context_mock)
    monkeypatch.setattr("neudc.core.communication.zero_queue.zero_queue.zmq.Poller", lambda: poller_mock)

    return {
        "context": context_mock,
        "socket": socket_mock,
        "poller": poller_mock,
    }


def test_zeroqueuepub_raises_on_get(mock_zmq) -> None:
    pub = ZeroQueuePub(port=5555)
    with pytest.raises(NotImplementedError):
        pub.get()
    with pytest.raises(NotImplementedError):
        pub.get_nowait()


def test_zeroqueuesub_raises_on_put(mock_zmq) -> None:
    sub = ZeroQueueSub()
    with pytest.raises(NotImplementedError):
        sub.put("test")
    with pytest.raises(NotImplementedError):
        sub.put_nowait("test")


def test_zeroqueue_put_sends_object(mock_zmq) -> None:
    pub = ZeroQueue(port=5555, mode=ZeroQueueMode.PUB, contype=ZeroQueueConnectionType.CONNECT)
    test_obj = {"key": "value"}
    pub.put(test_obj)
    mock_zmq["socket"].send.assert_called_once_with(codec.dumps(test_obj))


def test_zeroqueue_get_nowait_receives_object(mock_zmq) -> None:
    test_msg = {"msg": "hello"}
    socket = mock_zmq["socket"]
    poller = mock_zmq["poller"]

    socket.recv.return_value = codec.dumps(test_msg)
    poller.poll.return_value = [(socket, zmq.POLLIN)]  # <-- вот это фиксит

    sub = ZeroQueue(port=5555, mode=ZeroQueueMode.SUB, contype=ZeroQueueConnectionType.CONNECT)
    result = sub.get_nowait()

    assert result == test_msg


def test_zeroqueue_get_timeout_returns_none(mock_zmq) -> None:
    mock_zmq["poller"].poll.return_value = []

    sub = ZeroQueue(port=5555, mode=ZeroQueueMode.SUB, contype=ZeroQueueConnectionType.CONNECT)
    result = sub.get(timeout=0.1)
    assert result is None


def test_zeroqueue_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        ZeroQueue(port=5555, mode="INVALID", contype=ZeroQueueConnectionType.CONNECT)


def test_zeroqueue_invalid_connection_type(monkeypatch) -> None:
    from neudc.core.communication.zero_queue.zero_queue import ZeroQueue

    class FakeEnum:
        pass

    with pytest.raises(ValueError):
        ZeroQueue(port=-1, mode=ZeroQueueMode.SUB, contype=FakeEnum)
