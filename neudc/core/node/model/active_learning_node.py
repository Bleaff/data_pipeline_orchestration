from __future__ import annotations

from typing import Any

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.node.model.mixins.active_learning_mixin import ActiveLearningMixin


class ActiveLearning(ActiveLearningMixin, BaseThreadedNode):
    def __init__(self, num_to_select: int, conf_strategy: str, conf_weight: float, *args: Any, **kwargs: Any):

        ActiveLearningMixin.__init__(
            self, num_to_select=num_to_select, conf_strategy=conf_strategy, conf_weight=conf_weight
        )
        BaseThreadedNode.__init__(self, *args, **kwargs)

    @classmethod
    def from_config(cls: type[ActiveLearning], config: dict[str, Any]) -> ActiveLearning:

        return ActiveLearning(
            mailbox=config["mailbox"],
            num_to_select=config["num_to_select"],
            conf_strategy=config["conf_strategy"],
            conf_weight=config["conf_weight"],
        )
