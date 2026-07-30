"""Tests for RoutingFactory wiring of the priority control channel (#41).

Mirrors how `outputs` wiring is exercised end-to-end in `test/core/test_pipeline_e2e.py`,
but scoped to `RoutingFactory.create_mailboxes()` directly (no NodeFactory/real nodes
needed) since wiring is purely a mailbox-graph concern.
"""

from __future__ import annotations

import time

import pytest

from neudc.core.communication.messaging.routing_factory import RoutingFactory
from neudc.core.communication.messaging.types import ControlAction, ControlMessage


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_control_outputs_are_wired_to_the_targets_control_port() -> None:
    config = {
        "nodes": [
            {"id": "reader", "type": "FolderImageNode", "outputs": [], "control_outputs": ["controller"]},
            {"id": "controller", "type": "SaveImageNode", "outputs": [], "control_outputs": []},
        ],
    }
    router = RoutingFactory(config)
    mailboxes = router.create_mailboxes()
    try:
        assert "controller" in mailboxes["reader"][0].control_pub_sockets

        mailboxes["reader"][0].send_control(
            ControlMessage(timestamp=1.0, action=ControlAction.CANCEL, session_id="s", turn_id=1)
        )
        assert _wait_until(lambda: mailboxes["controller"][0].is_cancelled("s", 1))
    finally:
        for replicas in mailboxes.values():
            for mailbox in replicas:
                mailbox.stop()


def test_control_output_to_a_replicated_target_reaches_every_replica() -> None:
    # Interaction between #15 (replicas) and #41 (control channel): a control message
    # can't be round-robined like data, since we don't know in advance which replica is
    # processing the turn being cancelled. Every replica's control port must be wired.
    config = {
        "nodes": [
            {"id": "reader", "type": "FolderImageNode", "outputs": [], "control_outputs": ["controller"]},
            {"id": "controller", "type": "SaveImageNode", "outputs": [], "control_outputs": [], "replicas": 3},
        ],
    }
    router = RoutingFactory(config)
    mailboxes = router.create_mailboxes()
    try:
        assert len(mailboxes["controller"]) == 3
        assert len(mailboxes["reader"][0].control_pub_sockets["controller"]) == 3

        mailboxes["reader"][0].send_control(
            ControlMessage(timestamp=1.0, action=ControlAction.CANCEL, session_id="s", turn_id=7)
        )
        for replica in mailboxes["controller"]:
            assert _wait_until(lambda replica=replica: replica.is_cancelled("s", 7))
    finally:
        for replicas in mailboxes.values():
            for mailbox in replicas:
                mailbox.stop()


def test_dangling_control_output_raises() -> None:
    config = {
        "nodes": [
            {"id": "reader", "type": "FolderImageNode", "outputs": [], "control_outputs": ["ghost"]},
        ],
    }
    router = RoutingFactory(config)
    with pytest.raises(ValueError, match="not found in mailboxes"):
        router.create_mailboxes()


def test_no_control_outputs_configured_leaves_control_channel_unwired() -> None:
    # The regression case (#41, done-when criterion 3): a pipeline that never mentions
    # control_outputs must build and wire exactly as it did before this feature.
    config = {
        "nodes": [
            {"id": "reader", "type": "FolderImageNode", "outputs": ["saver"]},
            {"id": "saver", "type": "SaveImageNode", "outputs": []},
        ],
    }
    router = RoutingFactory(config)
    mailboxes = router.create_mailboxes()
    try:
        assert mailboxes["reader"][0].control_pub_sockets == {}
        assert mailboxes["saver"][0].control_pub_sockets == {}
        # Data wiring is unaffected.
        assert mailboxes["reader"][0].pub_sockets
    finally:
        for replicas in mailboxes.values():
            for mailbox in replicas:
                mailbox.stop()
