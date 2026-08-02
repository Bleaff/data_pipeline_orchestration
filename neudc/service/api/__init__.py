"""Control-plane API: FastAPI over `PipelineServiceManager` (#22).

REST for pipeline start/stop/status and config read/validate, a Prometheus `/metrics`
endpoint, and a `/ws/metrics` WebSocket feed. Optional layer — importing this package
pulls in `fastapi`; CPU-only pipeline runs (`neudc/entrypoints/main.py`,
`main_multipipe.py`) never import it.
"""

from __future__ import annotations

from neudc.service.api.app import create_app

__all__ = ("create_app",)
