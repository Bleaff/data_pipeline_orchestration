"""Unit tests for the queue-depth autoscaler decision engine (#15).

``ReplicaAutoscaler`` is exercised entirely through dependency injection here: a
fake "replica" is just an object with a settable queue depth, `scale_up`/`scale_down`
are plain callables recording their calls. This proves the scale-up/scale-down
*decision logic* (the boundary conditions around `queue_depth_high`/`_low` and
`min_replicas`/`max_replicas`) in isolation, without a real mailbox, ZMQ socket, or
running node — see the module docstring in `neudc/core/utils/autoscaler.py` and the
task's scope note: this suite intentionally does not wire a live pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neudc.core.utils.autoscale import AutoscaleConfig
from neudc.core.utils.autoscaler import ReplicaAutoscaler


@dataclass
class FakeReplica:
    """Minimal stand-in for a running node instance: just an id and a queue depth."""

    name: str
    depth: int = 0


@dataclass
class FakeCluster:
    """Records scale-up/scale-down actions so tests can assert on them."""

    created: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    _next_id: int = 0

    def scale_up(self) -> FakeReplica:
        self._next_id += 1
        name = f"replica-{self._next_id}"
        self.created.append(name)
        return FakeReplica(name=name)

    def scale_down(self, replica: FakeReplica) -> None:
        self.stopped.append(replica.name)


def _make_autoscaler(
    replicas: list[FakeReplica],
    cluster: FakeCluster,
    *,
    min_replicas: int = 1,
    max_replicas: int = 4,
    queue_depth_high: int = 15,
    queue_depth_low: int = 2,
) -> ReplicaAutoscaler[FakeReplica]:
    config = AutoscaleConfig(
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        queue_depth_high=queue_depth_high,
        queue_depth_low=queue_depth_low,
    )
    return ReplicaAutoscaler(
        replicas=replicas,
        config=config,
        queue_depth_reader=lambda r: r.depth,
        scale_up=cluster.scale_up,
        scale_down=cluster.scale_down,
    )


def test_scales_up_when_all_replicas_are_above_high_watermark() -> None:
    cluster = FakeCluster()
    replicas = [FakeReplica("a", depth=20), FakeReplica("b", depth=16)]
    autoscaler = _make_autoscaler(replicas, cluster, max_replicas=4, queue_depth_high=15)

    result = autoscaler.check_and_scale()

    assert result == "up"
    assert len(autoscaler.replicas) == 3
    assert cluster.created == ["replica-1"]
    assert cluster.stopped == []


def test_does_not_scale_up_if_any_replica_is_below_high_watermark() -> None:
    cluster = FakeCluster()
    replicas = [FakeReplica("a", depth=20), FakeReplica("b", depth=3)]
    autoscaler = _make_autoscaler(replicas, cluster, queue_depth_high=15)

    result = autoscaler.check_and_scale()

    assert result is None
    assert len(autoscaler.replicas) == 2
    assert cluster.created == []


def test_does_not_scale_up_past_max_replicas() -> None:
    cluster = FakeCluster()
    replicas = [FakeReplica("a", depth=100), FakeReplica("b", depth=100)]
    autoscaler = _make_autoscaler(replicas, cluster, max_replicas=2, queue_depth_high=15)

    result = autoscaler.check_and_scale()

    assert result is None
    assert len(autoscaler.replicas) == 2
    assert cluster.created == []


def test_scales_down_when_all_replicas_are_below_low_watermark() -> None:
    cluster = FakeCluster()
    replicas = [FakeReplica("a", depth=0), FakeReplica("b", depth=1), FakeReplica("c", depth=2)]
    autoscaler = _make_autoscaler(replicas, cluster, min_replicas=1, queue_depth_low=2)

    result = autoscaler.check_and_scale()

    assert result == "down"
    assert len(autoscaler.replicas) == 2
    assert len(cluster.stopped) == 1
    # The least-loaded replica ("a", depth=0) is the one removed.
    assert cluster.stopped == ["a"]


def test_does_not_scale_down_if_any_replica_is_above_low_watermark() -> None:
    cluster = FakeCluster()
    replicas = [FakeReplica("a", depth=0), FakeReplica("b", depth=5)]
    autoscaler = _make_autoscaler(replicas, cluster, queue_depth_low=2)

    result = autoscaler.check_and_scale()

    assert result is None
    assert len(autoscaler.replicas) == 2
    assert cluster.stopped == []


def test_does_not_scale_down_past_min_replicas() -> None:
    cluster = FakeCluster()
    replicas = [FakeReplica("a", depth=0)]
    autoscaler = _make_autoscaler(replicas, cluster, min_replicas=1, queue_depth_low=2)

    result = autoscaler.check_and_scale()

    assert result is None
    assert len(autoscaler.replicas) == 1
    assert cluster.stopped == []


def test_at_exact_high_watermark_boundary_scales_up() -> None:
    """`>= queue_depth_high` (not `>`) is the trigger."""
    cluster = FakeCluster()
    replicas = [FakeReplica("a", depth=15), FakeReplica("b", depth=15)]
    autoscaler = _make_autoscaler(replicas, cluster, queue_depth_high=15)

    assert autoscaler.check_and_scale() == "up"


def test_at_exact_low_watermark_boundary_scales_down() -> None:
    """`<= queue_depth_low` (not `<`) is the trigger."""
    cluster = FakeCluster()
    replicas = [FakeReplica("a", depth=2), FakeReplica("b", depth=2)]
    autoscaler = _make_autoscaler(replicas, cluster, queue_depth_low=2)

    assert autoscaler.check_and_scale() == "down"


def test_no_scaling_in_the_dead_zone_between_watermarks() -> None:
    cluster = FakeCluster()
    replicas = [FakeReplica("a", depth=8), FakeReplica("b", depth=9)]
    autoscaler = _make_autoscaler(replicas, cluster, queue_depth_high=15, queue_depth_low=2)

    assert autoscaler.check_and_scale() is None
    assert cluster.created == []
    assert cluster.stopped == []


def test_repeated_checks_scale_up_by_one_step_at_a_time() -> None:
    cluster = FakeCluster()
    replicas = [FakeReplica("a", depth=20)]
    autoscaler = _make_autoscaler(replicas, cluster, max_replicas=3, queue_depth_high=15)

    # New replicas start idle (depth 0), so after the first scale-up the *group* is no
    # longer uniformly above the high watermark, and scaling stops until load catches up.
    assert autoscaler.check_and_scale() == "up"
    assert len(autoscaler.replicas) == 2
    assert autoscaler.check_and_scale() is None
    assert len(autoscaler.replicas) == 2


def test_background_thread_start_stop_does_not_hang() -> None:
    """The periodic runner starts/stops cleanly; not asserting on timing-sensitive scaling."""
    cluster = FakeCluster()
    replicas = [FakeReplica("a", depth=0)]
    config = AutoscaleConfig(min_replicas=1, max_replicas=2, check_interval_s=0.05)
    autoscaler = ReplicaAutoscaler(
        replicas=replicas,
        config=config,
        queue_depth_reader=lambda r: r.depth,
        scale_up=cluster.scale_up,
        scale_down=cluster.scale_down,
    )

    autoscaler.start()
    autoscaler.start()  # calling twice must be a no-op, not a second thread
    autoscaler.stop()
    autoscaler.stop()  # calling twice must not raise
