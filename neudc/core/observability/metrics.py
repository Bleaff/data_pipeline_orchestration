"""Unified Prometheus metrics registry for pipeline observability (#14).

Every metric here carries a single ``node`` label, so one node's full picture —
queue depth, throughput, latency, errors, health — can be sliced in Grafana without
cross-referencing separate registries. This is deliberately separate from
:mod:`neudc.profilers.profiler_metrics`, which counts CV-specific frame/box
statistics inside filter and postprocess profilers; the metrics here describe the
*run loop* itself and are wired from ``core/base`` and ``core/policy``, not from a
node's own processing logic.
"""

from __future__ import annotations

import os

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

__all__ = (
    "DEAD_LETTERED",
    "DROPPED",
    "ERRORS",
    "FAILURES",
    "HEALTH",
    "MESSAGES_PROCESSED",
    "PROCESS_LATENCY",
    "QUEUE_DEPTH",
    "QUEUE_DROPS",
    "QUEUE_HWM",
    "RETRIES",
    "build_metrics_registry",
)

# --- Queues -----------------------------------------------------------------------

# `multiprocess_mode="livemostrecent"`: these gauges describe *current state of this
# node*, not a quantity to combine across processes. The default `"all"` mode keeps a
# separate series per pid forever (including long-dead ones) in `PROMETHEUS_MULTIPROC_DIR`,
# so a restarted node's fresh reading sits alongside its predecessor's last-known value
# with no ordering guarantee between them — a multiprocess reader (e.g. the
# control-plane API's `/metrics`/`/ws/metrics`) can end up surfacing the stale one.
# "livemostrecent" keeps only the latest write from a still-running process.
QUEUE_DEPTH = Gauge(
    "neudc_node_queue_depth",
    "Current depth of a node's internal (in-process) message queue",
    ["node"],
    multiprocess_mode="livemostrecent",
)
QUEUE_HWM = Gauge(
    "neudc_node_queue_hwm",
    "Configured ZMQ high-water mark for a node's inbound mailbox socket",
    ["node"],
    multiprocess_mode="livemostrecent",
)
QUEUE_DROPS = Counter(
    "neudc_node_queue_drops_total",
    "Messages dropped from a node's inbound queue under a non-blocking queue policy",
    ["node", "reason"],
)

# --- Throughput / latency -----------------------------------------------------------

MESSAGES_PROCESSED = Counter(
    "neudc_node_messages_processed_total",
    "Messages a node's process() completed successfully",
    ["node"],
)
PROCESS_LATENCY = Histogram(
    "neudc_node_process_latency_seconds",
    "process() wall-clock latency per attempt (including retried attempts)",
    ["node"],
)

# --- Errors / drops (mirrors neudc.core.policy.error_policy.ErrorStats) -------------

ERRORS = Counter(
    "neudc_node_errors_total",
    "process() exceptions raised, including attempts that were later retried",
    ["node"],
)
RETRIES = Counter(
    "neudc_node_retries_total",
    "Retry attempts made under an on_error: retry policy",
    ["node"],
)
DROPPED = Counter(
    "neudc_node_dropped_total",
    "Messages dropped after the error policy gave up on them",
    ["node"],
)
DEAD_LETTERED = Counter(
    "neudc_node_dead_lettered_total",
    "Messages written to a node's dead-letter sink",
    ["node"],
)
FAILURES = Counter(
    "neudc_node_failures_total",
    "Times a node gave up processing under an on_error: fail policy",
    ["node"],
)

# --- Health ---------------------------------------------------------------------

HEALTH = Gauge(
    "neudc_node_health",
    "Node health as observed by its own run loop (1=healthy, 0=unhealthy)",
    ["node"],
    multiprocess_mode="livemostrecent",
)


def build_metrics_registry() -> CollectorRegistry | None:
    """Return the registry metrics should be read from, per `PROMETHEUS_MULTIPROC_DIR`.

    Process nodes (`BaseProcessNode`) run in a separate OS process, so metrics they
    record live in that process's own private `prometheus_client` registry by default
    — invisible to whichever process is exposing `/metrics`. If the deployer set
    `PROMETHEUS_MULTIPROC_DIR` (per `prometheus_client`'s own multiprocess-mode
    contract: a writable, empty directory, set *before* any pipeline starts), this
    returns a registry that reads every node's metrics back from files in that
    directory instead. Returns `None` when the env var is unset, meaning: use
    `prometheus_client`'s implicit default (in-process) registry — the historical,
    single-process behaviour.

    Returns
    -------
        CollectorRegistry | None: A multiprocess-aware registry, or `None` for the default one.

    """
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not multiproc_dir:
        return None

    from prometheus_client import multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=multiproc_dir)
    return registry
