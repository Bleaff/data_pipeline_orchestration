"""Pipeline start/stop/status/preview routes (#22)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from neudc.core.utils.config_schema import ConfigError, validate_pipeline_config
from neudc.service.api.preview import find_latest_frame
from neudc.service.api.schemas import PipelineDetailResponse, PipelineStartRequest, PipelineStatusResponse

if TYPE_CHECKING:
    from neudc.service.pipeline_manager import PipelineServiceManager

__all__ = ("router",)

router = APIRouter()


@router.get("/pipelines")
def list_pipelines(request: Request) -> list[PipelineStatusResponse]:
    manager: PipelineServiceManager = request.app.state.manager
    return [PipelineStatusResponse(name=name, status=status) for name, status in manager.status().items()]


@router.get("/pipelines/{name}/status")
def get_pipeline_status(name: str, request: Request) -> PipelineStatusResponse:
    manager: PipelineServiceManager = request.app.state.manager
    statuses = manager.status()
    if name not in statuses:
        raise HTTPException(status_code=404, detail=f"Pipeline '{name}' not found")
    return PipelineStatusResponse(name=name, status=statuses[name])


@router.get("/pipelines/{name}")
def get_pipeline_detail(name: str, request: Request) -> PipelineDetailResponse:
    manager: PipelineServiceManager = request.app.state.manager
    statuses = manager.status()
    if name not in statuses:
        raise HTTPException(status_code=404, detail=f"Pipeline '{name}' not found")
    nodes_config = request.app.state.pipeline_nodes_configs.get(name, [])
    return PipelineDetailResponse(name=name, status=statuses[name], nodes=nodes_config)


@router.post("/pipelines/{name}/start", status_code=201)
def start_pipeline(name: str, body: PipelineStartRequest, request: Request) -> PipelineStatusResponse:
    try:
        validate_pipeline_config({"nodes": body.nodes})
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    manager: PipelineServiceManager = request.app.state.manager
    try:
        manager.start_pipeline(name=name, nodes_config=body.nodes, task_config=body.task)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    request.app.state.pipeline_nodes_configs[name] = body.nodes
    return PipelineStatusResponse(name=name, status=manager.status()[name])


@router.post("/pipelines/{name}/stop", status_code=204)
def stop_pipeline(name: str, request: Request) -> Response:
    manager: PipelineServiceManager = request.app.state.manager
    try:
        manager.stop_pipeline(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    request.app.state.pipeline_nodes_configs.pop(name, None)
    return Response(status_code=204)


@router.get("/pipelines/{name}/preview")
def preview_pipeline(name: str, request: Request) -> FileResponse:
    nodes_config = request.app.state.pipeline_nodes_configs.get(name)
    if nodes_config is None:
        raise HTTPException(status_code=404, detail=f"Pipeline '{name}' not found")

    frame_path = find_latest_frame(nodes_config)
    if frame_path is None:
        raise HTTPException(status_code=404, detail=f"Pipeline '{name}' has no SaveImageNode output yet")
    # Exposed so a polling client can tell two responses apart without comparing bytes.
    return FileResponse(frame_path, media_type="image/jpeg", headers={"X-Frame-Name": frame_path.name})
