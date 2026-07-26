"""Threaded node wiring `ActiveLearningMixin` selection logic into the pipeline."""

from __future__ import annotations

from typing import Any

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.node.model.mixins.active_learning_mixin import ActiveLearningMixin


class ActiveLearning(ActiveLearningMixin, BaseThreadedNode):
    """Threaded node that runs uncertainty-diversity active-learning selection on frames."""

    def __init__(self, num_to_select: int, conf_strategy: str, conf_weight: float, *args: Any, **kwargs: Any) -> None:
        """Initialize the node by wiring both parent classes' constructors.

        Args:
        ----
            num_to_select (int): How many images to pick once all expected frames are gathered.
            conf_strategy (str): Confidence aggregation strategy forwarded to `ActiveLearningMixin`.
            conf_weight (float): Uncertainty-vs-distance weight forwarded to `ActiveLearningMixin`.
            *args (Any): Positional arguments forwarded to `BaseThreadedNode`.
            **kwargs (Any): Keyword arguments forwarded to `BaseThreadedNode`.

        """
        ActiveLearningMixin.__init__(
            self, num_to_select=num_to_select, conf_strategy=conf_strategy, conf_weight=conf_weight
        )
        BaseThreadedNode.__init__(self, *args, **kwargs)

    @classmethod
    def from_config(cls: type[ActiveLearning], config: dict[str, Any]) -> ActiveLearning:
        """Build an `ActiveLearning` node from a configuration dictionary."""
        return ActiveLearning(
            mailbox=config["mailbox"],
            num_to_select=config["num_to_select"],
            conf_strategy=config["conf_strategy"],
            conf_weight=config["conf_weight"],
        )
