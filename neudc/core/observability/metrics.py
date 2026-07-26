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

from prometheus_client import Counter, Gauge, Histogram

# --- Queues -----------------------------------------------------------------------

QUEUE_DEPTH = Gauge(
    "neudc_node_queue_depth",
    "Current depth of a node's internal (in-process) message queue",
    ["node"],
)
QUEUE_HWM = Gauge(
    "neudc_node_queue_hwm",
    "Configured ZMQ high-water mark for a node's inbound mailbox socket",
    ["node"],
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
)
