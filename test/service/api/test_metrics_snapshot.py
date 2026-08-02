"""Unit tests for `snapshot_metrics`'s name/type handling (#22).

Regression coverage for a real bug found while wiring up the frontend (#23): a
`collect()` family's `.name` has the `_total` suffix stripped for counters, so without
restoring it, `/ws/metrics` and `/metrics` disagreed on what a counter was called and
the dashboard's "processed" count silently never appeared.
"""

from __future__ import annotations

from neudc.core.observability.metrics import MESSAGES_PROCESSED, PROCESS_LATENCY
from neudc.service.api.metrics import snapshot_metrics


def test_counter_key_keeps_the_total_suffix() -> None:
    MESSAGES_PROCESSED.labels(node="snapshot-test-counter-node").inc()

    snapshot = snapshot_metrics()

    assert "neudc_node_messages_processed_total" in snapshot
    assert "neudc_node_messages_processed" not in snapshot
    assert snapshot["neudc_node_messages_processed_total"]["snapshot-test-counter-node"] >= 1.0


def test_histogram_is_excluded_rather_than_merged_incorrectly() -> None:
    PROCESS_LATENCY.labels(node="snapshot-test-hist-node").observe(0.01)

    snapshot = snapshot_metrics()

    assert "neudc_node_process_latency_seconds" not in snapshot
