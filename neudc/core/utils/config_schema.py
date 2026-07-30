"""Declarative validation for pipeline configuration files.

The runtime accepts fairly free-form YAML (each node carries its own parameters),
so instead of enumerating every node's parameters we validate the *structure* that
the router and factory rely on:

- there is a top-level ``nodes`` list;
- every node has a unique ``id`` and a ``type`` known to the NodeFactory;
- every entry in a node's ``outputs`` references an existing node ``id``;
- every edge's producer ``emits`` a payload type the consumer ``accepts`` (#35/#36).

Node-specific parameters are preserved (``extra="allow"``) and checked later by each
node's ``from_config``. Validation failures raise :class:`ConfigError` with a message
that names the offending node, so a bad config fails fast and legibly at startup.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from neudc.core.node.node_factory import NodeFactory
from neudc.core.policy import ErrorPolicyConfig
from neudc.core.utils.autoscale import AutoscaleConfig


class ConfigError(ValueError):
    """Raised when a pipeline configuration is structurally invalid."""


class NodeSpec(BaseModel):
    """A single node entry. Node-specific parameters are kept as extra fields."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    outputs: list[str] = []
    control_outputs: list[str] = []
    """Optional priority control-plane edges (e.g. turn cancellation / barge-in, #41).

    Wired by :class:`~neudc.core.communication.messaging.routing_factory.RoutingFactory`
    into the target's control channel, independent of ``outputs``/the data FIFO. A node
    with no ``control_outputs`` gets a control mailbox that is simply never wired to
    anything, so an untouched config runs exactly as before.
    """
    #: Number of worker replicas for this node (#15). 1 (default) is the historical,
    #: single-instance behaviour. `autoscale` (an extra field, validated below the
    #: same way `error_policy` is) may grow/shrink this at runtime between
    #: `autoscale.min_replicas` and `autoscale.max_replicas`.
    replicas: int = 1
    #: Per-node override of BaseProcessNode.HEALTH_TIMEOUT / HEALTH_CHECK_INTERVAL
    #: (#39). `None` (default) falls back to the class-constant defaults; wired onto
    #: the constructed node by `NodeFactory.create`, same as `error_policy`. Only
    #: meaningful for process nodes, but harmless (ignored) on other node types.
    health_timeout: float | None = None
    health_check_interval: float | None = None


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
        id_to_node = {n.id: n for n in self.nodes}
        for node in self.nodes:
            if node.type not in known_types:
                msg = f"Node '{node.id}': unknown type '{node.type}'. Known types: {sorted(known_types)}"
                raise ConfigError(msg)
            for target in node.outputs:
                if target not in id_set:
                    msg = f"Node '{node.id}': output '{target}' does not reference any node id"
                    raise ConfigError(msg)
                _check_payload_compat(node, id_to_node[target])
            for target in node.control_outputs:
                if target not in id_set:
                    msg = f"Node '{node.id}': control_output '{target}' does not reference any node id"
                    raise ConfigError(msg)
            _check_error_policy(node)
            _check_replicas(node)
            _check_autoscale(node)
            _check_health_config(node)
        return self


def _check_payload_compat(node: NodeSpec, target: NodeSpec) -> None:
    """Validate that ``node``'s declared ``emits`` is compatible with ``target``'s ``accepts`` (#35/#36).

    Kept out of :class:`NodeSpec`, same reason as :func:`_check_error_policy`: the
    message should name both node ids the user wrote, not a positional index.
    """
    emits = NodeFactory.get_emits(node.type)
    accepts = NodeFactory.get_accepts(target.type)
    if not any(issubclass(e, a) for e in emits for a in accepts):
        emits_names = [t.__name__ for t in emits]
        accepts_names = [t.__name__ for t in accepts]
        msg = f"Node '{node.id}' emits {emits_names} but '{target.id}' only accepts {accepts_names}"
        raise ConfigError(msg)


def _check_error_policy(node: NodeSpec) -> None:
    """Validate a node's optional ``error_policy`` block, naming the node on failure.

    Kept out of :class:`NodeSpec` so the message points at the node id the user wrote
    rather than at a positional index in the ``nodes`` list.
    """
    raw = getattr(node, "error_policy", None)
    if raw is None:
        return
    if not isinstance(raw, dict):
        msg = f"Node '{node.id}': 'error_policy' must be a mapping, got {type(raw).__name__}"
        raise ConfigError(msg)
    try:
        ErrorPolicyConfig(**raw)
    except ValidationError as exc:
        msg = f"Node '{node.id}': invalid 'error_policy': {exc}"
        raise ConfigError(msg) from exc


def _check_replicas(node: NodeSpec) -> None:
    """Validate a node's ``replicas`` count, naming the node on failure (#15)."""
    if node.replicas < 1:
        msg = f"Node '{node.id}': 'replicas' must be >= 1, got {node.replicas}"
        raise ConfigError(msg)


def _check_autoscale(node: NodeSpec) -> None:
    """Validate a node's optional ``autoscale`` block, naming the node on failure (#15).

    Kept out of :class:`NodeSpec`, same reason as :func:`_check_error_policy`: the
    message should point at the node id the user wrote, not a positional index.
    """
    raw = getattr(node, "autoscale", None)
    if raw is None:
        return
    if not isinstance(raw, dict):
        msg = f"Node '{node.id}': 'autoscale' must be a mapping, got {type(raw).__name__}"
        raise ConfigError(msg)
    try:
        AutoscaleConfig(**raw)
    except ValidationError as exc:
        msg = f"Node '{node.id}': invalid 'autoscale': {exc}"
        raise ConfigError(msg) from exc


def _check_health_config(node: NodeSpec) -> None:
    """Validate a node's optional ``health_timeout``/``health_check_interval`` (#39).

    Kept out of :class:`NodeSpec` as a field-level constraint, same reason as
    :func:`_check_error_policy`: the message should name the node id the user wrote,
    not a positional index. Both are optional; when omitted the node falls back to
    `BaseProcessNode.HEALTH_TIMEOUT`/`HEALTH_CHECK_INTERVAL`.
    """
    if node.health_timeout is not None and node.health_timeout <= 0:
        msg = f"Node '{node.id}': 'health_timeout' must be > 0, got {node.health_timeout}"
        raise ConfigError(msg)
    if node.health_check_interval is not None and node.health_check_interval <= 0:
        msg = f"Node '{node.id}': 'health_check_interval' must be > 0, got {node.health_check_interval}"
        raise ConfigError(msg)


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
