"""Shared fixtures for control-plane API tests: a fake manager, no real subprocess pipelines."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from neudc.service.api import create_app


class FakePipelineServiceManager:
    """Duck-types `PipelineServiceManager` without spawning real `BasePipeline` processes.

    Mirrors `start_pipeline`/`stop_pipeline`/`register_external_pipeline`/
    `unregister_external_pipeline`/`status`/`pipelines` exactly, so the API layer can
    be exercised at full speed; a slower true-subprocess integration test lives in
    `test_app_integration.py` for the one path worth proving end-to-end.
    """

    def __init__(self) -> None:
        self.pipelines: dict[str, str] = {}
        self.external_pipelines: dict[str, list[dict[str, Any]]] = {}

    def start_pipeline(
        self, name: str, nodes_config: list[dict[str, Any]], task_config: dict[str, Any] | None = None
    ) -> None:
        if name in self.pipelines or name in self.external_pipelines:
            msg = f"Pipeline '{name}' already running"
            raise ValueError(msg)
        self.pipelines[name] = "running"

    def stop_pipeline(self, name: str) -> None:
        if name not in self.pipelines:
            msg = f"Pipeline '{name}' not found"
            raise ValueError(msg)
        del self.pipelines[name]

    def register_external_pipeline(self, name: str, nodes_config: list[dict[str, Any]]) -> None:
        if name in self.pipelines or name in self.external_pipelines:
            msg = f"Pipeline '{name}' already running"
            raise ValueError(msg)
        self.external_pipelines[name] = nodes_config

    def unregister_external_pipeline(self, name: str) -> None:
        if name not in self.external_pipelines:
            msg = f"External pipeline '{name}' not found"
            raise ValueError(msg)
        del self.external_pipelines[name]

    def status(self) -> dict[str, str]:
        statuses = dict(self.pipelines)
        statuses.update(dict.fromkeys(self.external_pipelines, "running"))
        return statuses


@pytest.fixture
def fake_manager() -> FakePipelineServiceManager:
    return FakePipelineServiceManager()


@pytest.fixture
def client(fake_manager: FakePipelineServiceManager) -> TestClient:
    return TestClient(create_app(manager=fake_manager))
