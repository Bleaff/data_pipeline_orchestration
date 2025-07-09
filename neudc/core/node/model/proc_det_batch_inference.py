from typing import Any

from neudc.core.communication.messaging.types import Batch, Box
from neudc.core.node.model.base_batch_process_inference import BaseBatchProcessInference


class ProcessDetBatchInference(BaseBatchProcessInference):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def postprocess_result(self, result: Any, item: Batch):
        """Put the result to the item"""
        # self.logger.info(f'Model detection result: {result}')
        for res, frame in zip(result, item.frames):
            boxes = []
            for box_ in res:
                x, y, x2, y2, score, class_id = box_
                box = Box(
                    x1=int(x), y1=int(y), x2=int(x2), y2=int(y2), score=score, class_id=self.model.names[int(class_id)]
                )
                boxes.append(box)
            frame.boxes.extend(boxes)
        return item
