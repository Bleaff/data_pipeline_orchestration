"""Cross-cutting observability: the unified per-node Prometheus metrics registry (#14).

Lives alongside ``core/policy`` rather than inside ``core/base``: the base classes stay
wiring (they call into this module), the metric definitions and any recording logic
live here.
"""

from neudc.core.observability.metrics import (
    DEAD_LETTERED,
    DROPPED,
    ERRORS,
    FAILURES,
    HEALTH,
    MESSAGES_PROCESSED,
    PROCESS_LATENCY,
    QUEUE_DEPTH,
    QUEUE_HWM,
    RETRIES,
)

__all__ = (
    "DEAD_LETTERED",
    "DROPPED",
    "ERRORS",
    "FAILURES",
    "HEALTH",
    "MESSAGES_PROCESSED",
    "PROCESS_LATENCY",
    "QUEUE_DEPTH",
    "QUEUE_HWM",
    "RETRIES",
)
