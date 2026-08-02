"""Tests for the pipeline start/stop/status routes (#22), against a fake manager."""

from __future__ import annotations

_VALID_NODES = [{"id": "reader", "type": "FolderImageNode", "folder_path": "unused", "outputs": []}]
_BAD_TYPE_NODES = [{"id": "reader", "type": "NotARealNodeType", "outputs": []}]


def test_list_pipelines_starts_empty(client) -> None:
    resp = client.get("/pipelines")

    assert resp.status_code == 200
    assert resp.json() == []


def test_start_pipeline_returns_running_status(client) -> None:
    resp = client.post("/pipelines/p1/start", json={"nodes": _VALID_NODES})

    assert resp.status_code == 201
    assert resp.json() == {"name": "p1", "status": "running"}


def test_start_pipeline_rejects_invalid_config(client, fake_manager) -> None:
    resp = client.post("/pipelines/p1/start", json={"nodes": _BAD_TYPE_NODES})

    assert resp.status_code == 422
    assert "unknown type" in resp.json()["detail"]
    # A rejected config must never reach the manager.
    assert fake_manager.pipelines == {}


def test_start_pipeline_conflicts_when_already_running(client) -> None:
    client.post("/pipelines/p1/start", json={"nodes": _VALID_NODES})

    resp = client.post("/pipelines/p1/start", json={"nodes": _VALID_NODES})

    assert resp.status_code == 409
    assert "already running" in resp.json()["detail"]


def test_get_detail_of_unknown_pipeline_is_404(client) -> None:
    resp = client.get("/pipelines/ghost")

    assert resp.status_code == 404


def test_get_detail_returns_status_and_nodes(client) -> None:
    client.post("/pipelines/p1/start", json={"nodes": _VALID_NODES})

    resp = client.get("/pipelines/p1")

    assert resp.status_code == 200
    assert resp.json() == {"name": "p1", "status": "running", "nodes": _VALID_NODES}


def test_get_status_of_unknown_pipeline_is_404(client) -> None:
    resp = client.get("/pipelines/ghost/status")

    assert resp.status_code == 404


def test_get_status_of_running_pipeline(client) -> None:
    client.post("/pipelines/p1/start", json={"nodes": _VALID_NODES})

    resp = client.get("/pipelines/p1/status")

    assert resp.status_code == 200
    assert resp.json() == {"name": "p1", "status": "running"}


def test_stop_unknown_pipeline_is_404(client) -> None:
    resp = client.post("/pipelines/ghost/stop")

    assert resp.status_code == 404


def test_stop_pipeline_removes_it_from_the_list(client) -> None:
    client.post("/pipelines/p1/start", json={"nodes": _VALID_NODES})

    resp = client.post("/pipelines/p1/stop")

    assert resp.status_code == 204
    assert client.get("/pipelines").json() == []
