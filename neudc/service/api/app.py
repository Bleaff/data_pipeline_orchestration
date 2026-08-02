"""FastAPI control-plane app: REST + WS surface over `PipelineServiceManager` (#22)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neudc.service.api.routes import config, observability, pipelines
from neudc.service.pipeline_manager import PipelineServiceManager

__all__ = ("create_app",)


def create_app(manager: PipelineServiceManager | None = None, cors_origins: list[str] | None = None) -> FastAPI:
    """Build the control-plane FastAPI app.

    Args:
    ----
        manager (PipelineServiceManager | None): Manager to expose; a fresh one is
            created if omitted. Injectable so tests can supply a lightweight fake
            instead of spinning real subprocess pipelines.
        cors_origins (list[str] | None): Origins allowed to call this API from a
            browser (e.g. the frontend dev server). Defaults to `["*"]` — this is an
            internal control-plane tool with no cookie-based auth, so a permissive
            default is safe; tighten it for a real deployment.

    Returns:
    -------
        FastAPI: The configured application.

    """
    app = FastAPI(title="neudc control plane")
    app.state.manager = manager or PipelineServiceManager()
    # Per-pipeline nodes_config, kept only so `/pipelines/{name}/preview` can locate a
    # SaveImageNode's save_dir; PipelineServiceManager itself doesn't retain it.
    app.state.pipeline_nodes_configs: dict[str, list[dict[str, Any]]] = {}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Frame-Name"],
    )

    app.include_router(pipelines.router)
    app.include_router(config.router)
    app.include_router(observability.router)

    return app
