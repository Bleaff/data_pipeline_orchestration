"""Tests for RoutingFactory wiring `message_queue_size`/`queue_policy` into each node's
mailbox (#38).

Mirrors `test_routing_factory_control.py`: scoped to `RoutingFactory.create_mailboxes()`
directly (no NodeFactory/real nodes needed), since this is purely a mailbox-graph concern.
"""  # noqa: D205

from __future__ import annotations

from neudc.core.communication.mailbox.zmq_mailbox import QueuePolicy
from neudc.core.communication.messaging.routing_factory import RoutingFactory


def _stop_all(mailboxes) -> None:
    for replicas in mailboxes.values():
        for mailbox in replicas:
            mailbox.stop()


def test_default_config_gets_the_historical_block_policy_and_size() -> None:
    # Regression: a config that never mentions these keys must build exactly as before.
    config = {"nodes": [{"id": "reader", "type": "FolderImageNode", "outputs": []}]}
    mailboxes = RoutingFactory(config).create_mailboxes()
    try:
        mailbox = mailboxes["reader"][0]
        assert mailbox.queue_policy is QueuePolicy.BLOCK
        assert mailbox._message_queue.maxsize == 20
    finally:
        _stop_all(mailboxes)


def test_queue_policy_and_size_are_threaded_into_the_mailbox() -> None:
    config = {
        "nodes": [
            {
                "id": "reader",
                "type": "FolderImageNode",
                "outputs": [],
                "message_queue_size": 3,
                "queue_policy": "drop_oldest",
            },
        ],
    }
    mailboxes = RoutingFactory(config).create_mailboxes()
    try:
        mailbox = mailboxes["reader"][0]
        assert mailbox.queue_policy is QueuePolicy.DROP_OLDEST
        assert mailbox._message_queue.maxsize == 3
    finally:
        _stop_all(mailboxes)


def test_queue_policy_applies_to_every_replica() -> None:
    config = {
        "nodes": [
            {
                "id": "detector",
                "type": "SaveImageNode",
                "outputs": [],
                "replicas": 3,
                "message_queue_size": 4,
                "queue_policy": "conflate",
            },
        ],
    }
    mailboxes = RoutingFactory(config).create_mailboxes()
    try:
        assert len(mailboxes["detector"]) == 3
        for mailbox in mailboxes["detector"]:
            assert mailbox.queue_policy is QueuePolicy.CONFLATE
            assert mailbox._message_queue.maxsize == 4
    finally:
        _stop_all(mailboxes)
