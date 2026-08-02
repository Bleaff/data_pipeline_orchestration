"""Prometheus exposition and JSON snapshotting for the control-plane API (#22)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import REGISTRY, generate_latest

from neudc.core.observability.metrics import build_metrics_registry

if TYPE_CHECKING:
    from prometheus_client.registry import CollectorRegistry

__all__ = ("render_prometheus_text", "snapshot_metrics")


def _active_registry() -> CollectorRegistry:
    """Return the registry to read from: multiprocess-aware if configured, else the default."""
    return build_metrics_registry() or REGISTRY


def render_prometheus_text() -> bytes:
    """Render the active registry in the standard Prometheus text exposition format."""
    return generate_latest(_active_registry())


def snapshot_metrics() -> dict[str, dict[str, float]]:
    """Collect the active registry into `{metric_name: {node_label: value}}`.

    Every metric in `neudc.core.observability.metrics` carries a single `node` label
    (`neudc_node_queue_drops_total` also carries `reason`, folded into the node key as
    `f"{node}:{reason}"` so the snapshot stays a flat two-level mapping for the WS feed).

    `metric_name` matches the name a client would see in `/metrics`: a `collect()`
    family's own `.name` has the `_total` suffix stripped for counters (prometheus_client
    convention), so it's restored here — otherwise the two endpoints would disagree on
    what a given metric is called. Histograms are skipped: their per-bucket/count/sum
    samples don't reduce to one `node -> value` series, and nothing reads them today.
    """
    snapshot: dict[str, dict[str, float]] = {}
    for family in _active_registry().collect():
        if not family.name.startswith("neudc_node_") or family.type in ("histogram", "summary"):
            continue
        name = f"{family.name}_total" if family.type == "counter" else family.name
        series = snapshot.setdefault(name, {})
        for sample in family.samples:
            node = sample.labels.get("node", "")
            reason = sample.labels.get("reason")
            key = f"{node}:{reason}" if reason else node
            series[key] = sample.value
    return snapshot
