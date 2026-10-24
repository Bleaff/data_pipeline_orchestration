"""Tests for VadNode `from_config`: config dict -> wired VadFilterMixin instance."""

from __future__ import annotations

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.node.filters.vad_node import VadNode


def test_from_config_wires_energy_threshold_and_session_id() -> None:
    node = VadNode.from_config(
        {
            "mailbox": ZMQMailbox(),
            "energy_threshold": 0.02,
            "session_id": "sess-1",
        },
    )

    assert node.energy_threshold == 0.02
    assert node.session_id == "sess-1"


def test_from_config_defaults_session_id() -> None:
    node = VadNode.from_config({"mailbox": ZMQMailbox()})

    assert node.session_id == "default"
    assert node.energy_threshold == 0.01


def test_from_config_wires_barge_in_options() -> None:
    node = VadNode.from_config(
        {
            "mailbox": ZMQMailbox(),
            "barge_in_energy_threshold": 0.3,
            "barge_in_cooldown_sec": 2.5,
        },
    )

    assert node.barge_in_energy_threshold == 0.3
    assert node.barge_in_cooldown_sec == 2.5


def test_from_config_defaults_barge_in_threshold_to_energy_threshold() -> None:
    node = VadNode.from_config({"mailbox": ZMQMailbox(), "energy_threshold": 0.04})

    assert node.barge_in_energy_threshold == 0.04
    assert node.barge_in_cooldown_sec == 1.5
