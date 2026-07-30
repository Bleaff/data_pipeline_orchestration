"""Tests for per-edge inbound queue policy: block / drop_oldest / conflate (#38).

``block`` is the historical, lossless default (regression-tested elsewhere, notably
``test_mailbox_hotpath.py::test_full_internal_queue_applies_backpressure_without_none``).
The tests here cover the two new, lossy policies plus the crux of the issue: dropping a
message already sitting in ``_message_queue`` never leaks a shared-memory segment,
because by the time a message is queued there ``codec.loads`` has already copied its
out-of-band buffers into owned memory and unlinked the backing segment (see
``codec.py``'s "Ownership contract").
"""

from __future__ import annotations

import itertools
import sys
import time

import numpy as np
import pytest
from prometheus_client import REGISTRY

from neudc.core.communication.mailbox import zmq_mailbox
from neudc.core.communication.mailbox.zmq_mailbox import QueuePolicy, ZMQMailbox
from neudc.core.communication.messaging import codec
from neudc.core.communication.messaging.types import Frame

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shared memory only")

_counter = itertools.count()


def _node_id(prefix: str) -> str:
    """Build a fresh, collision-free node label so Prometheus's global registry
    (shared across the whole test run) can't leak values between tests.
    """  # noqa: D205
    return f"{prefix}-{next(_counter)}"


def _drops(node: str, reason: str) -> float:
    return REGISTRY.get_sample_value("neudc_node_queue_drops_total", {"node": node, "reason": reason}) or 0.0


def _big_image(side: int = 256) -> np.ndarray:
    # Comfortably above SHM_MIN_BYTES (64 KiB) so pixels go out-of-band to shared memory.
    return (np.arange(side * side * 3, dtype=np.uint8) % 255).reshape(side, side, 3)


def _make_frame(image: np.ndarray, frame_id: int) -> Frame:
    return Frame(
        image=image,
        timestamp=1.5,
        source_frame="src.png",
        frame_id=frame_id,
        boxes=[],
        embedding=None,
        frame_id_last=frame_id,
        drop=False,
    )


# --- Config coercion ---------------------------------------------------------------


def test_queue_policy_defaults_to_block() -> None:
    mailbox = ZMQMailbox(name=_node_id("default-policy"))
    try:
        assert mailbox.queue_policy is QueuePolicy.BLOCK
    finally:
        mailbox.stop()


def test_queue_policy_accepts_plain_string() -> None:
    mailbox = ZMQMailbox(name=_node_id("string-policy"), queue_policy="drop_oldest")
    try:
        assert mailbox.queue_policy is QueuePolicy.DROP_OLDEST
    finally:
        mailbox.stop()


def test_invalid_queue_policy_string_is_rejected() -> None:
    with pytest.raises(ValueError, match="bogus"):
        ZMQMailbox(name=_node_id("bad-policy"), queue_policy="bogus")


# --- drop_oldest ---------------------------------------------------------------------


def test_drop_oldest_evicts_the_oldest_message_and_counts_drops() -> None:
    node_id = _node_id("drop-oldest")
    receiver = ZMQMailbox(name=node_id, message_queue_size=2, queue_policy="drop_oldest")
    sender = ZMQMailbox(name=_node_id("drop-oldest-sender"))
    try:
        sender.add_publisher("receiver", receiver.consume_port)

        n = 5
        for i in range(n):
            sender.send({"i": i})

        # capacity 2: messages 0,1 fill the queue; 2,3,4 each evict the current oldest.
        expected_drops = n - 2
        deadline = time.time() + 5
        while _drops(node_id, "drop_oldest") < expected_drops and time.time() < deadline:
            time.sleep(0.01)

        assert _drops(node_id, "drop_oldest") == expected_drops
        assert receiver.queue_depth == 2

        remaining = []
        while True:
            msg = receiver.receive(timeout=0.2)
            if msg is None:
                break
            remaining.append(msg["i"])

        # The two newest messages survive, oldest-first (it's still a FIFO queue).
        assert remaining == [3, 4]
    finally:
        sender.stop()
        receiver.stop()


def test_drop_oldest_does_not_block_the_receiver_thread() -> None:
    """A full drop_oldest queue must accept new messages without ever blocking."""
    node_id = _node_id("drop-oldest-nonblock")
    receiver = ZMQMailbox(name=node_id, message_queue_size=1, queue_policy="drop_oldest")
    sender = ZMQMailbox(name=_node_id("drop-oldest-nonblock-sender"))
    try:
        sender.add_publisher("receiver", receiver.consume_port)

        n = 50
        start = time.time()
        for i in range(n):
            sender.send({"i": i})

        deadline = time.time() + 5
        while _drops(node_id, "drop_oldest") < n - 1 and time.time() < deadline:
            time.sleep(0.01)
        elapsed = time.time() - start

        assert _drops(node_id, "drop_oldest") == n - 1
        # Nothing here should ever block on a full queue; this should be fast.
        assert elapsed < 5
        assert receiver.queue_depth == 1
        got = receiver.receive(timeout=0.5)
        assert got == {"i": n - 1}
    finally:
        sender.stop()
        receiver.stop()


# --- conflate ------------------------------------------------------------------------


def test_conflate_keeps_only_the_most_recent_message() -> None:
    node_id = _node_id("conflate")
    receiver = ZMQMailbox(name=node_id, message_queue_size=20, queue_policy="conflate")
    sender = ZMQMailbox(name=_node_id("conflate-sender"))
    try:
        sender.add_publisher("receiver", receiver.consume_port)

        n = 5
        for i in range(n):
            sender.send({"i": i})

        # Every message but the first evicts exactly one already-queued item.
        expected_drops = n - 1
        deadline = time.time() + 5
        while _drops(node_id, "conflate") < expected_drops and time.time() < deadline:
            time.sleep(0.01)

        assert _drops(node_id, "conflate") == expected_drops
        assert receiver.queue_depth == 1

        got = receiver.receive(timeout=0.5)
        assert got == {"i": n - 1}
        # Only one message was ever retrievable.
        assert receiver.receive(timeout=0.2) is None
    finally:
        sender.stop()
        receiver.stop()


def test_conflate_does_not_count_a_drop_when_queue_starts_empty() -> None:
    node_id = _node_id("conflate-empty")
    receiver = ZMQMailbox(name=node_id, queue_policy="conflate")
    sender = ZMQMailbox(name=_node_id("conflate-empty-sender"))
    try:
        sender.add_publisher("receiver", receiver.consume_port)
        sender.send({"i": 0})

        deadline = time.time() + 5
        while receiver.queue_depth < 1 and time.time() < deadline:
            time.sleep(0.01)

        assert receiver.queue_depth == 1
        assert _drops(node_id, "conflate") == 0
    finally:
        sender.stop()
        receiver.stop()


# --- SHM: dropping a queued message must never leak a segment ------------------------


def _install_segment_spies(monkeypatch) -> tuple[set[str], set[str]]:
    """Track every shared-memory segment name created vs. consumed (copied + unlinked).

    ``_create_segment``/``_consume_segment`` are called by bare name from within
    ``codec.py`` (``_dumps_shm``/``_loads_shm``), which resolves them via the module's
    own globals — so monkeypatching the attribute on the ``codec`` module object is
    enough to intercept every call, including calls the mailbox's receiver thread makes
    indirectly through ``ZeroQueue.get()`` -> ``codec.loads()``.
    """
    created: set[str] = set()
    consumed: set[str] = set()

    real_create_segment = codec._create_segment
    real_consume_segment = codec._consume_segment

    def spy_create_segment(nbytes: int):
        shm = real_create_segment(nbytes)
        created.add(shm.name)
        return shm

    def spy_consume_segment(name: str, nbytes: int):
        result = real_consume_segment(name, nbytes)
        consumed.add(name)
        return result

    monkeypatch.setattr(codec, "_create_segment", spy_create_segment)
    monkeypatch.setattr(codec, "_consume_segment", spy_consume_segment)
    return created, consumed


@posix_only
@pytest.mark.parametrize("policy", ["drop_oldest", "conflate"])
def test_shm_segments_are_not_leaked_when_queued_messages_are_dropped(monkeypatch, policy: str) -> None:
    monkeypatch.setattr(zmq_mailbox.codec, "SHM_ENABLED", True)
    created, consumed = _install_segment_spies(monkeypatch)

    node_id = _node_id(f"shm-{policy}")
    receiver = ZMQMailbox(name=node_id, message_queue_size=2, queue_policy=policy)
    sender = ZMQMailbox(name=_node_id(f"shm-{policy}-sender"))
    try:
        sender.add_publisher("receiver", receiver.consume_port)

        img = _big_image()
        n = 8
        for i in range(n):
            sender.send(_make_frame(img, frame_id=i))

        # Wait until the receiver thread has pulled (and therefore SHM-consumed) every
        # message off the socket, regardless of how many of them then got dropped from
        # `_message_queue` under the policy under test.
        deadline = time.time() + 10
        while len(consumed) < n and time.time() < deadline:
            time.sleep(0.01)

        assert len(created) == n, "expected one shared-memory segment per large frame sent"
        # This is the crux of the test: every created segment was also consumed (copied
        # out and unlinked) by codec.loads -- *before* the mailbox's queue policy ever
        # got a chance to drop the corresponding already-deserialized message. No segment
        # is left dangling just because the queued Python object it once backed was
        # evicted by drop_oldest/conflate.
        assert consumed == created, f"leaked shared-memory segments: {created - consumed}"

        # Sanity: the policy actually did drop something in this scenario (otherwise the
        # test would not be exercising what it claims to).
        assert _drops(node_id, policy) > 0
    finally:
        sender.stop()
        receiver.stop()
