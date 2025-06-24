"""ProcessDetInference class.

This class is a base class for process nodes that perform object detection.
The class is designed to be used as a node in a dataflow graph.
The class is thread-safe and can be used with asyncio.
"""

from typing import Any

from neudc.core.communication.messaging.types import Box, Frame
from neudc.core.node.model.base_process_inference import BaseProcessInference


class ProcessDetInference(BaseProcessInference):
    """ProcessDetInference class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ProcessDetInference class."""
        super().__init__(*args, **kwargs)

    def postprocess_result(self, result: Any, item: Frame):
        """Put the result to the item."""
        # self.logger.info(f'Model detection result: {result}')
        boxes = []
        for box_ in result[0]:
            x, y, x2, y2, score, class_id = box_
            box = Box(
                x1=int(x),
                y1=int(y),
                x2=int(x2),
                y2=int(y2),
                score=score,
                class_id=self.model.names[int(class_id)],  # type: ignore[attr-defined]
            )
            boxes.append(box)
        item.boxes.extend(boxes)
        return item
