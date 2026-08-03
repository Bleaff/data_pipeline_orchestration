"""Pre-label review routes (#24): list/view/correct a pipeline's `CreateDataset` output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from neudc.service.api.labels import (
    find_dataset_root,
    get_frame_detail,
    get_frame_image_path,
    list_label_frames,
    save_frame_boxes,
)

# Route return/body types must stay real (not TYPE_CHECKING-only) imports: FastAPI
# resolves them via runtime reflection for OpenAPI schema generation and response
# serialization, even under `from __future__ import annotations` — ruff's TC001 can't
# tell a route handler apart from a plain function and flags these as if they were
# type-checking-only, which would raise at app startup if "fixed".
from neudc.service.api.schemas import (  # noqa: TC001
    LabelFrameDetail,
    LabelFrameSummary,
    LabelUpdateRequest,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ("router",)

router = APIRouter()


def _dataset_root_or_404(name: str, request: Request) -> Path:
    nodes_config = request.app.state.pipeline_nodes_configs.get(name)
    if nodes_config is None:
        raise HTTPException(status_code=404, detail=f"Pipeline '{name}' not found")
    dataset_root = find_dataset_root(nodes_config)
    if dataset_root is None:
        raise HTTPException(status_code=404, detail=f"Pipeline '{name}' has no CreateDataset node")
    return dataset_root


@router.get("/pipelines/{name}/labels")
def list_labels(name: str, request: Request) -> list[LabelFrameSummary]:
    dataset_root = _dataset_root_or_404(name, request)
    return list_label_frames(dataset_root)


@router.get("/pipelines/{name}/labels/{frame_id}")
def get_label(name: str, frame_id: str, request: Request) -> LabelFrameDetail:
    dataset_root = _dataset_root_or_404(name, request)
    detail = get_frame_detail(dataset_root, frame_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Frame '{frame_id}' not found")
    return detail


@router.get("/pipelines/{name}/labels/{frame_id}/image")
def get_label_image(name: str, frame_id: str, request: Request) -> FileResponse:
    dataset_root = _dataset_root_or_404(name, request)
    image_path = get_frame_image_path(dataset_root, frame_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Frame '{frame_id}' not found")
    return FileResponse(image_path, media_type="image/jpeg")


@router.put("/pipelines/{name}/labels/{frame_id}")
def update_label(name: str, frame_id: str, body: LabelUpdateRequest, request: Request) -> LabelFrameDetail:
    dataset_root = _dataset_root_or_404(name, request)
    detail = save_frame_boxes(dataset_root, frame_id, body.boxes)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Frame '{frame_id}' not found")
    return detail
