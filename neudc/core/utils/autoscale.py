"""Config model for a node's optional ``autoscale`` block (#15).

Mirrors how :class:`~neudc.core.policy.error_policy.ErrorPolicyConfig` lives
alongside the behaviour it configures and is only *referenced* from
:mod:`neudc.core.utils.config_schema`, not defined there: the schema module stays a
thin structural validator, and each block's own module owns its shape and invariants.

``replicas: N`` in a node config (see ``NodeSpec.replicas``) sets the static/starting
worker count for that node. ``autoscale`` is optional and, when present, lets
:class:`~neudc.core.utils.autoscaler.ReplicaAutoscaler` grow or shrink that count at
runtime based on queue depth, between ``min_replicas`` and ``max_replicas``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AutoscaleConfig(BaseModel):
    """Per-node ``autoscale`` block from the pipeline YAML.

    ``queue_depth_high``/``queue_depth_low`` are read against
    :attr:`~neudc.core.communication.mailbox.zmq_mailbox.ZMQMailbox.queue_depth` (the
    node's own inbound message queue), not a Prometheus gauge, so the decision does
    not depend on metrics being enabled.
    """

    model_config = ConfigDict(extra="forbid")

    min_replicas: int = Field(default=1, ge=1)
    max_replicas: int = Field(default=1, ge=1)
    queue_depth_high: int = Field(default=15, ge=0)
    queue_depth_low: int = Field(default=2, ge=0)
    check_interval_s: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> AutoscaleConfig:
        if self.min_replicas > self.max_replicas:
            msg = f"min_replicas ({self.min_replicas}) must be <= max_replicas ({self.max_replicas})"
            raise ValueError(msg)
        if self.queue_depth_low >= self.queue_depth_high:
            msg = f"queue_depth_low ({self.queue_depth_low}) must be < queue_depth_high ({self.queue_depth_high})"
            raise ValueError(msg)
        return self
