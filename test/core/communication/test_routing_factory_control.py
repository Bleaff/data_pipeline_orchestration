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
        assert "controller" in mailboxes["reader"].control_pub_sockets

        mailboxes["reader"].send_control(
            ControlMessage(timestamp=1.0, action=ControlAction.CANCEL, session_id="s", turn_id=1)
        )
        assert _wait_until(lambda: mailboxes["controller"].is_cancelled("s", 1))
    finally:
        for mailbox in mailboxes.values():
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
        assert mailboxes["reader"].control_pub_sockets == {}
        assert mailboxes["saver"].control_pub_sockets == {}
        # Data wiring is unaffected.
        assert mailboxes["reader"].pub_sockets
    finally:
        for mailbox in mailboxes.values():
            mailbox.stop()
