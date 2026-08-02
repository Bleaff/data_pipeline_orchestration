"""Prometheus `/metrics` and `/ws/metrics` streaming routes (#22)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST

from neudc.service.api.metrics import render_prometheus_text, snapshot_metrics
from neudc.utils import LOGGER

__all__ = ("router",)

# How often the WS metrics feed pushes a fresh snapshot to a connected client.
METRICS_STREAM_INTERVAL_SEC = 1.0

router = APIRouter()


@router.get("/metrics")
def get_metrics() -> Response:
    return Response(content=render_prometheus_text(), media_type=CONTENT_TYPE_LATEST)


@router.websocket("/ws/metrics")
async def stream_metrics(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(snapshot_metrics())
            await asyncio.sleep(METRICS_STREAM_INTERVAL_SEC)
    except WebSocketDisconnect:
        LOGGER.debug("Metrics WS client disconnected")
