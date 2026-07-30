"""The error policy as the node run loops actually use it (Stage 9, #11).

Both loops — the threaded one in ``BaseNode._run`` and the process one in
``BaseProcessNode.run`` — must go through the same ``_handle``. They drifted apart
once already (#12), so these tests pin the shared path rather than one loop's copy.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from neudc.core.base.base_process import BaseProcessNode
from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.node.node_factory import NodeFactory
from neudc.core.policy import ErrorAction, ErrorPolicy


class _BoomThreadedNode(BaseThreadedNode):
    """Threaded node whose process() always fails, fed by a canned message."""

    def __init__(self, mailbox: Any) -> None:
        super().__init__(mailbox, id="boom")
        self.calls = 0

    def _collect_data(self) -> Any:
        # Paced deliberately: a real node blocks on mailbox.receive(), and an
        # unpaced always-failing loop just floods the log.
        time.sleep(0.01)
        return "msg"

    def process(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        msg = "process exploded"
        raise RuntimeError(msg)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _BoomThreadedNode:
        return cls(config["mailbox"])


class _BoomProcessNode(BaseProcessNode):
    """Process node used for parent-side assertions only (no spawn — see #12 test notes)."""

    def process(self, item: Any) -> Any:
        msg = "process exploded"
        raise RuntimeError(msg)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _BoomProcessNode:
        return cls(config["mailbox"])


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# === Threaded loop ===


def test_skip_policy_keeps_the_node_running() -> None:
    # The historical behaviour: a failing message is dropped, the node carries on.
    node = _BoomThreadedNode(ZMQMailbox())
    try:
        node.start()
        assert _wait_until(lambda: node.error_policy.stats.dropped >= 2)

        assert node.failed() is False
        assert node.is_running is True
    finally:
        node.stop()


def test_fail_policy_stops_the_node_and_flags_it() -> None:
    node = _BoomThreadedNode(ZMQMailbox())
    node.error_policy = ErrorPolicy.from_config({"on_error": "fail"}, node_id="boom")
    try:
        node.start()

        assert _wait_until(node.failed), "node should be flagged as failed"
        assert _wait_until(lambda: node.is_running is False), "node should stop itself"
        # It gave up on the first message rather than spinning on more of them.
        assert node.error_policy.stats.failures == 1
    finally:
        node.stop()


def test_retry_policy_reattempts_before_dropping() -> None:
    node = _BoomThreadedNode(ZMQMailbox())
    node.error_policy = ErrorPolicy.from_config(
        {"on_error": "retry", "max_retries": 2, "initial_backoff": 0.001, "max_backoff": 0.01},
        node_id="boom",
    )
    try:
        node.start()
        assert _wait_until(lambda: node.error_policy.stats.dropped >= 1)

        # Three process() calls per dropped message: the attempt plus two retries.
        assert node.calls >= 3
        assert node.error_policy.stats.retries >= 2
    finally:
        node.stop()


# === Process loop (parent side, no spawn) ===


def test_process_node_handle_uses_the_same_policy_path() -> None:
    node = _BoomProcessNode(ZMQMailbox(), id="boom")
    node.error_policy = ErrorPolicy.from_config({"on_error": "skip"}, node_id="boom")

    assert node._handle("msg") is None
    assert node.error_policy.stats.dropped == 1
    assert node.failed() is False


def test_process_node_fail_policy_sets_the_shared_failed_event() -> None:
    # The event must be a multiprocessing one: the parent has to see what the child sets.
    node = _BoomProcessNode(ZMQMailbox(), id="boom")
    node.error_policy = ErrorPolicy.from_config({"on_error": "fail"}, node_id="boom")

    node._handle("msg")

    assert node.failed() is True
    # The run loop's own exit condition, so the process winds down on its next pass.
    assert node.stop_event.is_set() is True


def test_process_node_uses_multiprocessing_primitives() -> None:
    # BaseNode creates threading events, which a process node must swap out: they are
    # not picklable for spawn, and a threading event set in the child would be
    # invisible to the parent anyway.
    import multiprocessing.synchronize as mp_sync

    node = _BoomProcessNode(ZMQMailbox(), id="boom")

    assert isinstance(node._failed_event, mp_sync.Event)
    assert isinstance(node._stop_event, mp_sync.Event)
    assert not isinstance(node._failed_event, threading.Event)


# === Factory wiring ===


def test_factory_builds_the_policy_from_node_config(tmp_path) -> None:
    # Wiring lives in the factory, so no node's from_config has to know about it.
    node = NodeFactory.create(
        {
            "id": "reader",
            "type": "FolderImageNode",
            "folder_path": str(tmp_path),
            "outputs": [],
            "error_policy": {"on_error": "retry", "max_retries": 4},
        },
        mailbox=ZMQMailbox(),
    )

    assert node.error_policy.config.on_error is ErrorAction.RETRY
    assert node.error_policy.config.max_retries == 4
    # The policy tags its dead-letter records and logs with the node's configured id,
    # which the factory now propagates (nodes used to keep the base class default).
    assert node.error_policy.node_id == "reader"
    assert node.id == "reader"


def test_factory_gives_an_unconfigured_node_the_historical_behaviour(tmp_path) -> None:
    node = NodeFactory.create(
        {"id": "reader", "type": "FolderImageNode", "folder_path": str(tmp_path), "outputs": []},
        mailbox=ZMQMailbox(),
    )

    assert node.error_policy.config.on_error is ErrorAction.SKIP
    assert node.error_policy.config.max_retries == 0


def test_factory_wires_health_overrides_onto_the_node(tmp_path) -> None:
    # Same pop-then-wire pattern as error_policy (#39): the keys must not leak into
    # from_config's kwargs and must land on the constructed instance.
    node = NodeFactory.create(
        {
            "id": "reader",
            "type": "FolderImageNode",
            "folder_path": str(tmp_path),
            "outputs": [],
            "health_timeout": 30,
            "health_check_interval": 2,
        },
        mailbox=ZMQMailbox(),
    )

    assert node.health_timeout == 30
    assert node.health_check_interval == 2


def test_factory_leaves_health_defaults_when_unconfigured(tmp_path) -> None:
    node = NodeFactory.create(
        {"id": "reader", "type": "FolderImageNode", "folder_path": str(tmp_path), "outputs": []},
        mailbox=ZMQMailbox(),
    )

    # BaseThreadedNode has no health_timeout attribute at all -- the factory must not
    # invent one when the config doesn't mention it.
    assert not hasattr(node, "health_timeout")


def test_factory_pops_queue_policy_keys_before_from_config(tmp_path, monkeypatch) -> None:
    # `message_queue_size`/`queue_policy` (#38) are routing-only config, consumed by
    # RoutingFactory when building the node's mailbox -- they must not leak into the
    # node class's own `from_config`, same treatment as `replicas`/`autoscale`.
    from neudc.core.node.readers.image_reader import FolderImageNode

    captured: dict = {}
    real_from_config = FolderImageNode.from_config

    def spy(config):
        captured.update(config)
        return real_from_config(config)

    monkeypatch.setattr(FolderImageNode, "from_config", staticmethod(spy))

    NodeFactory.create(
        {
            "id": "reader",
            "type": "FolderImageNode",
            "folder_path": str(tmp_path),
            "outputs": [],
            "message_queue_size": 5,
            "queue_policy": "conflate",
        },
        mailbox=ZMQMailbox(),
    )

    assert "message_queue_size" not in captured
    assert "queue_policy" not in captured
