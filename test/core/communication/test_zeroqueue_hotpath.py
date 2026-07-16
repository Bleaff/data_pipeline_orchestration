"""Regression test: ZeroQueue.put must not sleep per message (Stage 1)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from neudc.core.communication.zero_queue.zero_queue import ZeroQueue
from neudc.core.communication.zero_queue.zmq_state import ZeroQueueConnectionType, ZeroQueueMode


def test_put_does_not_call_time_sleep(monkeypatch) -> None:
    context_mock = MagicMock()
    socket_mock = MagicMock()
    poller_mock = MagicMock()
    context_mock.socket.return_value = socket_mock
    monkeypatch.setattr("neudc.core.communication.zero_queue.zero_queue.zmq.Context", lambda: context_mock)
    monkeypatch.setattr("neudc.core.communication.zero_queue.zero_queue.zmq.Poller", lambda: poller_mock)

    sleep_calls: list = []
    monkeypatch.setattr(time, "sleep", lambda *a, **k: sleep_calls.append(a))

    pub = ZeroQueue(port=5555, mode=ZeroQueueMode.PUB, contype=ZeroQueueConnectionType.CONNECT)
    sleep_calls.clear()

    for i in range(50):
        pub.put({"i": i})

    assert socket_mock.send.call_count == 50
    assert sleep_calls == []  # no per-message sleep


def test_put_forwards_object(monkeypatch) -> None:
    context_mock = MagicMock()
    socket_mock = MagicMock()
    poller_mock = MagicMock()
    context_mock.socket.return_value = socket_mock
    monkeypatch.setattr("neudc.core.communication.zero_queue.zero_queue.zmq.Context", lambda: context_mock)
    monkeypatch.setattr("neudc.core.communication.zero_queue.zero_queue.zmq.Poller", lambda: poller_mock)

    pub = ZeroQueue(port=5555, mode=ZeroQueueMode.PUB, contype=ZeroQueueConnectionType.CONNECT)
    payload = {"key": "value"}
    pub.put(payload)
    from neudc.core.communication.messaging import codec

    socket_mock.send.assert_called_once_with(codec.dumps(payload))
