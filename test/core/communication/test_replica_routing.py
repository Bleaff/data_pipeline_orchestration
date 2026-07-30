"""Tests for worker-replica round-robin routing (#15).

Covers three layers:
1. Transport: a single ``ZeroQueuePub`` connected to N ``ZeroQueueSub`` ports
   actually round-robins messages fairly across all N receivers.
2. ``ZMQMailbox``: growing/shrinking a replica group's connections at runtime.
3. ``RoutingFactory``: a node config with ``replicas: N`` produces N mailboxes with
   distinct consume ports, and the producer's mailbox keeps exactly one
   ``ZeroQueuePub`` entry (not N) connected to every replica's port — the invariant
   the SHM single-consumer fast path in ``ZMQMailbox.send`` relies on.
"""

from __future__ import annotations

import time

import pytest

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging.routing_factory import RoutingFactory
from neudc.core.communication.zero_queue import ZeroQueuePub, ZeroQueueSub


def _drain(sub: ZeroQueueSub, expected: int, timeout: float = 8.0) -> list:
    received: list = []
    deadline = time.time() + timeout
    while len(received) < expected and time.time() < deadline:
        msg = sub.get(timeout=0.1)
        if msg is not None:
            received.append(msg)
    return received


# === 1. Transport: one PUSH -> N PULL round-robins ===


def test_one_pub_round_robins_across_n_sub_ports() -> None:
    n_workers = 3
    subs = [ZeroQueueSub() for _ in range(n_workers)]
    pub = ZeroQueuePub(ports=[sub.port for sub in subs])
    try:
        n_messages = 90
        for i in range(n_messages):
            pub.put({"i": i})

        received_per_worker: list[list] = [[] for _ in subs]
        deadline = time.time() + 8
        total = 0
        while total < n_messages and time.time() < deadline:
            for idx, sub in enumerate(subs):
                msg = sub.get(timeout=0.05)
                if msg is not None:
                    received_per_worker[idx].append(msg)
                    total += 1

        counts = [len(msgs) for msgs in received_per_worker]
        assert sum(counts) == n_messages
        # Fair round-robin: every worker gets a non-trivial share, not all-or-nothing.
        min_expected_share = n_messages / n_workers * 0.5
        assert all(c >= min_expected_share for c in counts), counts
    finally:
        pub.stop()
        for sub in subs:
            sub.stop()


def test_connect_additional_grows_the_fanout() -> None:
    """A replica group can grow at runtime (autoscale scale-up) without losing messages."""
    sub_a = ZeroQueueSub()
    pub = ZeroQueuePub(port=sub_a.port)
    sub_b = ZeroQueueSub()
    try:
        assert pub.ports == [sub_a.port]
        pub.connect_additional(sub_b.port)
        assert sorted(pub.ports) == sorted([sub_a.port, sub_b.port])

        n = 40
        for i in range(n):
            pub.put({"i": i})

        received: list = []
        deadline = time.time() + 8
        while len(received) < n and time.time() < deadline:
            for sub in (sub_a, sub_b):
                msg = sub.get(timeout=0.05)
                if msg is not None:
                    received.append(msg)

        assert len(received) == n
    finally:
        pub.stop()
        sub_a.stop()
        sub_b.stop()


def test_disconnect_additional_shrinks_the_fanout() -> None:
    sub_a = ZeroQueueSub()
    sub_b = ZeroQueueSub()
    pub = ZeroQueuePub(ports=[sub_a.port, sub_b.port])
    try:
        pub.disconnect_additional(sub_b.port)
        assert pub.ports == [sub_a.port]
    finally:
        pub.stop()
        sub_a.stop()
        sub_b.stop()


def test_disconnect_additional_refuses_to_remove_the_last_port() -> None:
    sub = ZeroQueueSub()
    pub = ZeroQueuePub(port=sub.port)
    try:
        with pytest.raises(ValueError, match="only remaining port"):
            pub.disconnect_additional(sub.port)
    finally:
        pub.stop()
        sub.stop()


# === 2. ZMQMailbox: add_publisher groups replicas under one target id ===


def test_mailbox_add_publisher_with_replica_ports_uses_one_pub_socket() -> None:
    sender = ZMQMailbox()
    workers = [ZMQMailbox() for _ in range(3)]
    try:
        worker_ports = [w.consume_port for w in workers]
        sender.add_publisher("worker", worker_ports)

        # One logical target -> one ZeroQueuePub, connected to every replica's port.
        assert list(sender.pub_sockets.keys()) == ["worker"]
        assert sorted(sender.pub_sockets["worker"].ports) == sorted(worker_ports)

        n = 30
        for i in range(n):
            sender.send({"i": i})

        received: list = []
        deadline = time.time() + 8
        while len(received) < n and time.time() < deadline:
            for w in workers:
                msg = w.receive(timeout=0.05)
                if msg is not None:
                    received.append(msg)

        # Total delivered is what matters here; per-worker fairness is covered by
        # the transport-level round-robin test above.
        assert len(received) == n
    finally:
        sender.stop()
        for w in workers:
            w.stop()


# === 3. RoutingFactory: `replicas: N` builds N mailboxes, one shared ZeroQueuePub ===


def test_routing_factory_builds_replica_mailboxes_with_distinct_ports() -> None:
    config = {
        "nodes": [
            {"id": "producer", "outputs": ["worker"]},
            {"id": "worker", "replicas": 3},
        ],
    }
    router = RoutingFactory(config)
    mailboxes = router.create_mailboxes()
    try:
        assert len(mailboxes["worker"]) == 3
        assert len(mailboxes["producer"]) == 1

        worker_ports = [mb.consume_port for mb in mailboxes["worker"]]
        assert len(set(worker_ports)) == 3  # distinct ports

        producer_mailbox = mailboxes["producer"][0]
        # SHM invariant (#15): one logical downstream target -> one pub_sockets entry,
        # even though it fans out to three physical replica ports.
        assert len(producer_mailbox.pub_sockets) == 1
        assert sorted(producer_mailbox.pub_sockets["worker"].ports) == sorted(worker_ports)
    finally:
        for mbs in mailboxes.values():
            for mb in mbs:
                mb.stop()


def test_routing_factory_default_replicas_is_a_single_mailbox() -> None:
    """No `replicas` key must behave exactly as before: one node id -> one mailbox."""
    config = {
        "nodes": [
            {"id": "reader", "outputs": ["saver"]},
            {"id": "saver", "outputs": []},
        ],
    }
    router = RoutingFactory(config)
    mailboxes = router.create_mailboxes()
    try:
        assert len(mailboxes["reader"]) == 1
        assert len(mailboxes["saver"]) == 1
        assert len(mailboxes["reader"][0].pub_sockets) == 1
    finally:
        for mbs in mailboxes.values():
            for mb in mbs:
                mb.stop()


def test_routing_factory_replica_group_round_robins_messages() -> None:
    """End-to-end: a producer replica's PUSH fans messages out across all consumer replicas."""
    config = {
        "nodes": [
            {"id": "producer", "outputs": ["worker"]},
            {"id": "worker", "replicas": 3},
        ],
    }
    router = RoutingFactory(config)
    mailboxes = router.create_mailboxes()
    try:
        producer_mailbox = mailboxes["producer"][0]
        workers = mailboxes["worker"]

        n = 60
        for i in range(n):
            producer_mailbox.send({"i": i})

        received_per_worker: list[list] = [[] for _ in workers]
        total = 0
        deadline = time.time() + 8
        while total < n and time.time() < deadline:
            for idx, w in enumerate(workers):
                msg = w.receive(timeout=0.05)
                if msg is not None:
                    received_per_worker[idx].append(msg)
                    total += 1

        counts = [len(msgs) for msgs in received_per_worker]
        assert sum(counts) == n
        min_expected_share = n / len(workers) * 0.3
        assert all(c >= min_expected_share for c in counts), counts
    finally:
        for mbs in mailboxes.values():
            for mb in mbs:
                mb.stop()
