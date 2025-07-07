from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neudc.core.base.base_process import BaseProcessNode
from neudc.nn.backends.torch import TorchBackend
from neudc.nn.models.det.yolo import YOLOv8

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import Frame


class ProcessInference(BaseProcessNode):
    def __init__(self, mailbox: Any, logger: Any) -> None:
        super().__init__(mailbox, logger)
        self.model = None

    def process(self, item: Frame) -> Any:
        if self.model is None:
            self.model = YOLOv8("/Users/bleaf/Desktop/Job/neudc/models/yolov8n.torchscript", TorchBackend, device_id=-1)
            self.model.warmup(iters=10)
        image = item.image
        self.model(
            [
                image,
            ],
        )
        return item

    @classmethod
    def from_config(cls: type[BaseProcessNode], config: dict[str, Any]) -> BaseProcessNode:
        """From config-based constructor for building node with specified config."""
        return ProcessInference(config["mailbox"], config["logger"])
