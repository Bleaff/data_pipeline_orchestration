"""Declarative validation for pipeline configuration files.

The runtime accepts fairly free-form YAML (each node carries its own parameters),
so instead of enumerating every node's parameters we validate the *structure* that
the router and factory rely on:

- there is a top-level ``nodes`` list;
- every node has a unique ``id`` and a ``type`` known to the NodeFactory;
- every entry in a node's ``outputs`` references an existing node ``id``.

Node-specific parameters are preserved (``extra="allow"``) and checked later by each
node's ``from_config``. Validation failures raise :class:`ConfigError` with a message
that names the offending node, so a bad config fails fast and legibly at startup.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from neudc.core.node.node_factory import NodeFactory


class ConfigError(ValueError):
    """Raised when a pipeline configuration is structurally invalid."""


class NodeSpec(BaseModel):
    """A single node entry. Node-specific parameters are kept as extra fields."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    outputs: list[str] = []


class PipelineConfig(BaseModel):
    """A single pipeline: a list of nodes plus any extra top-level keys (e.g. prometheus)."""

    model_config = ConfigDict(extra="allow")

    nodes: list[NodeSpec]

    @model_validator(mode="after")
    def _check_graph(self) -> PipelineConfig:
        ids = [node.id for node in self.nodes]
        duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
        if duplicates:
            msg = f"Duplicate node ids: {duplicates}"
            raise ConfigError(msg)

        known_types = set(NodeFactory.NODE_IMPORTS)
        id_set = set(ids)
        for node in self.nodes:
            if node.type not in known_types:
                msg = f"Node '{node.id}': unknown type '{node.type}'. Known types: {sorted(known_types)}"
                raise ConfigError(msg)
            for target in node.outputs:
                if target not in id_set:
                    msg = f"Node '{node.id}': output '{target}' does not reference any node id"
                    raise ConfigError(msg)
        return self


def validate_pipeline_config(raw: Any) -> PipelineConfig:
    """Validate a single-pipeline config dict and return the parsed model.

    Args:
    ----
        raw: The loaded YAML config (expected to be a mapping with a ``nodes`` list).

    Returns:
    -------
        PipelineConfig: The validated configuration.

    Raises:
    ------
        ConfigError: If the config is not a mapping, lacks ``nodes``, or fails validation.

    """
    if not isinstance(raw, dict):
        msg = f"Config must be a mapping, got {type(raw).__name__}"
        raise ConfigError(msg)
    if "nodes" not in raw:
        msg = "Config must contain a top-level 'nodes' list"
        raise ConfigError(msg)
    try:
        return PipelineConfig(**raw)
    except ConfigError:
        raise
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
