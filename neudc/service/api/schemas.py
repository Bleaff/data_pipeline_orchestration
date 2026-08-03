"""Request/response models for the control-plane API (#22)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

__all__ = (
    "BoxSpec",
    "ConfigValidateRequest",
    "ConfigValidateResponse",
    "LabelFrameDetail",
    "LabelFrameSummary",
    "LabelUpdateRequest",
    "PipelineDetailResponse",
    "PipelineStartRequest",
    "PipelineStatusResponse",
)


class PipelineStartRequest(BaseModel):
    """Body of `POST /pipelines/{name}/start`."""

    nodes: list[dict[str, Any]]
    task: dict[str, Any] | None = None


class PipelineStatusResponse(BaseModel):
    """A single pipeline's reported status."""

    name: str
    status: Literal["running", "stopped"]


class PipelineDetailResponse(BaseModel):
    """A single pipeline's status plus its node config, for rendering the graph view (#23)."""

    name: str
    status: Literal["running", "stopped"]
    nodes: list[dict[str, Any]]


class ConfigValidateRequest(BaseModel):
    """Body of `POST /config/validate`: a raw pipeline config (a `nodes` list)."""

    nodes: list[dict[str, Any]]


class ConfigValidateResponse(BaseModel):
    """Result of validating a `ConfigValidateRequest` against `config_schema`."""

    valid: bool
    error: str | None = None


class BoxSpec(BaseModel):
    """One YOLO-format box: normalized (0-1) center/size, same convention `CreateDatasetMixin` writes."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class LabelFrameSummary(BaseModel):
    """One frame in a pipeline's `CreateDataset` output, for the review queue list (#24)."""

    frame_id: str
    box_count: int
    reviewed: bool


class LabelFrameDetail(BaseModel):
    """A single frame's boxes, for the review UI."""

    frame_id: str
    boxes: list[BoxSpec]
    reviewed: bool


class LabelUpdateRequest(BaseModel):
    """Body of `PUT /pipelines/{name}/labels/{frame_id}`: the human-corrected boxes."""

    boxes: list[BoxSpec]
