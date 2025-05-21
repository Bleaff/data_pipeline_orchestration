"""Tests for ZeroQueue."""

import time

from core.data_transfer.zero_queue.zero_queue import ZeroQueue


def test_queue_put_and_get() -> None:
    """Test put() and get() methods of ZeroQueue."""
    queue = ZeroQueue()

    test_msg = {"foo": "bar"}
    queue.put(test_msg)

    # ZeroMQ PUB/SUB требует времени на подписку
    time.sleep(0.05)

    result = queue.get(timeout=1)
    assert result is None, "SUB не должен получать собственное сообщение"


def test_cross_instance_put_get() -> None:
    """Test put() and get() methods across different ZeroQueue instances."""
    queue_sender = ZeroQueue()
    queue_receiver = ZeroQueue(port=queue_sender.port)

    time.sleep(0.1)  # дать время на соединение SUB сокета

    test_msg = {"msg": "hello"}
    queue_sender.put(test_msg)

    result = queue_receiver.get(timeout=1)
    assert result == test_msg
