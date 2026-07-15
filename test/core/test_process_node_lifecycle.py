"""Parent-side lifecycle tests for BaseProcessNode (Stage 2).

These cover the construction hand-off and the readiness event without spawning a
child process (spawn + locally-defined classes is inherently flaky). The
cross-process readiness path is exercised indirectly by the multipipe manager test.
"""

from __future__ import annotations

from typing import Any

from neudc.core.base.base_process import BaseProcessNode
from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox


class _EchoProcessNode(BaseProcessNode):
    """Minimal concrete process node used only for construction-time assertions."""

    def process(self, item: Any) -> Any:
        return item


def test_construction_captures_config_and_releases_parent_mailbox() -> None:
    mailbox = ZMQMailbox()
    assert mailbox._running is True

    node = _EchoProcessNode(mailbox, id="echo")

    # A picklable snapshot is captured for the child...
    assert isinstance(node.mailbox_config, dict)
    assert node.mailbox_config["consume_port"] == mailbox.consume_port
    # ...and the parent mailbox is explicitly released so the child can rebind.
    assert mailbox._running is False


def test_ready_event_starts_unset() -> None:
    mailbox = ZMQMailbox()
    node = _EchoProcessNode(mailbox, id="echo")

    assert hasattr(node, "_ready_event")
    assert node._ready_event.is_set() is False
    # Not started yet, so waiting with a zero timeout reports "not ready".
    assert node.wait_ready(timeout=0) is False
