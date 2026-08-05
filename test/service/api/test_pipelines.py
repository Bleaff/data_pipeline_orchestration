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


# === /register and /unregister: caller-owned pipelines (e.g. live mic/speaker device
# objects that can't be started via /start's JSON body) made visible to the dashboard.


def test_register_external_pipeline_appears_as_running(client) -> None:
    resp = client.post("/pipelines/voice-assistant/register", json={"nodes": _VALID_NODES})

    assert resp.status_code == 201
    assert resp.json() == {"name": "voice-assistant", "status": "running"}
    assert client.get("/pipelines").json() == [{"name": "voice-assistant", "status": "running"}]


def test_register_external_pipeline_rejects_invalid_config(client, fake_manager) -> None:
    resp = client.post("/pipelines/voice-assistant/register", json={"nodes": _BAD_TYPE_NODES})

    assert resp.status_code == 422
    assert fake_manager.external_pipelines == {}


def test_register_external_pipeline_conflicts_with_a_managed_pipeline(client) -> None:
    client.post("/pipelines/p1/start", json={"nodes": _VALID_NODES})

    resp = client.post("/pipelines/p1/register", json={"nodes": _VALID_NODES})

    assert resp.status_code == 409


def test_registered_pipeline_detail_returns_its_nodes(client) -> None:
    client.post("/pipelines/voice-assistant/register", json={"nodes": _VALID_NODES})

    resp = client.get("/pipelines/voice-assistant")

    assert resp.status_code == 200
    assert resp.json() == {"name": "voice-assistant", "status": "running", "nodes": _VALID_NODES}


def test_unregister_removes_it_from_the_list(client) -> None:
    client.post("/pipelines/voice-assistant/register", json={"nodes": _VALID_NODES})

    resp = client.post("/pipelines/voice-assistant/unregister")

    assert resp.status_code == 204
    assert client.get("/pipelines").json() == []


def test_unregister_unknown_pipeline_is_404(client) -> None:
    resp = client.post("/pipelines/ghost/unregister")

    assert resp.status_code == 404


def test_stop_does_not_affect_an_externally_registered_pipeline(client) -> None:
    # /stop is for manager-owned processes; an external registration has none to stop.
    client.post("/pipelines/voice-assistant/register", json={"nodes": _VALID_NODES})

    resp = client.post("/pipelines/voice-assistant/stop")

    assert resp.status_code == 404
    assert client.get("/pipelines").json() == [{"name": "voice-assistant", "status": "running"}]
