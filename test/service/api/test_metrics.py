"""Tests for the /metrics and /ws/metrics routes (#22)."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST

from neudc.core.observability.metrics import HEALTH


def test_metrics_endpoint_serves_prometheus_text_format(client) -> None:
    HEALTH.labels(node="api-metrics-test-node").set(1)

    resp = client.get("/metrics")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == CONTENT_TYPE_LATEST
    assert b"neudc_node_health" in resp.content


def test_ws_metrics_streams_a_json_snapshot(client) -> None:
    HEALTH.labels(node="api-ws-test-node").set(1)

    with client.websocket_connect("/ws/metrics") as websocket:
        snapshot = websocket.receive_json()

    assert "neudc_node_health" in snapshot
    assert snapshot["neudc_node_health"]["api-ws-test-node"] == 1.0
