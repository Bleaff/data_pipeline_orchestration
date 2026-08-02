"""Config read/validate routes (#22)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from neudc.core.utils.config_loader import load_config
from neudc.core.utils.config_schema import ConfigError, validate_pipeline_config
from neudc.service.api.schemas import ConfigValidateRequest, ConfigValidateResponse

__all__ = ("router",)

router = APIRouter()


@router.post("/config/validate")
def validate_config(body: ConfigValidateRequest) -> ConfigValidateResponse:
    try:
        validate_pipeline_config({"nodes": body.nodes})
    except ConfigError as exc:
        return ConfigValidateResponse(valid=False, error=str(exc))
    return ConfigValidateResponse(valid=True)


@router.get("/config")
def read_config(path: str) -> JSONResponse:
    try:
        config = load_config(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(content=config)
