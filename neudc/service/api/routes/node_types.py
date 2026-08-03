"""Node-type catalog route, for the visual config builder (#24)."""

from __future__ import annotations

from fastapi import APIRouter

from neudc.service.api.node_catalog import NodeTypeSpec, get_node_catalog

__all__ = ("router",)

router = APIRouter()


@router.get("/node-types")
def list_node_types() -> list[NodeTypeSpec]:
    return get_node_catalog()
