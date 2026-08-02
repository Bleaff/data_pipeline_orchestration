"""Tests that a browser-origin frontend (e.g. the Next.js dev server) can call the API (#23)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from neudc.service.api import create_app


def test_cors_preflight_allows_any_origin_by_default(fake_manager) -> None:
    client = TestClient(create_app(manager=fake_manager))

    resp = client.options(
        "/pipelines",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )

    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"


def test_cors_can_be_restricted_to_specific_origins(fake_manager) -> None:
    client = TestClient(create_app(manager=fake_manager, cors_origins=["http://trusted.example"]))

    resp = client.options(
        "/pipelines",
        headers={"Origin": "http://untrusted.example", "Access-Control-Request-Method": "GET"},
    )

    assert "access-control-allow-origin" not in resp.headers
