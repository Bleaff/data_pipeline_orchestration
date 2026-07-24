"""Execution policies applied around a node's own logic.

Currently the error policy (Stage 9): what a node does when ``process()`` raises —
retry with backoff, skip the message, or fail the pipeline — plus the dead-letter
sink that records whatever could not be processed.

The behaviour lives here rather than in ``core/base`` so the base classes stay
wiring: their loops call the policy, they do not implement it.
"""

from neudc.core.policy.dead_letter import DeadLetterSink
from neudc.core.policy.error_policy import ErrorAction, ErrorPolicy, ErrorPolicyConfig, ErrorStats, NodeFailure

__all__ = (
    "DeadLetterSink",
    "ErrorAction",
    "ErrorPolicy",
    "ErrorPolicyConfig",
    "ErrorStats",
    "NodeFailure",
)
