from typing import Any

from neudc.core.communication.messaging.types import Batch, Box
from neudc.core.node.model.base_batch_process_inference import BaseBatchProcessInference


class ProcessDetBatchInference(BaseBatchProcessInference):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model_cfg = kwargs.get("model_config", {})
        self.extract_embeddings: bool = model_cfg.get("extract_embeddings", False)

    def postprocess_result(self, result: Any, item: Batch):
        """Put the result to the item"""
        if self.extract_embeddings:
            preds, embs = result
        else:
            preds, embs = result, None

        for i, (dets, frame) in enumerate(zip(preds, item.frames)):
            frame.boxes.extend([
                Box(
                    x1=int(x), y1=int(y), x2=int(x2), y2=int(y2),
                    score=float(score),
                    class_id=self.model.names[int(cls_id)],
                )
                for x, y, x2, y2, score, cls_id in dets
            ])

            if self.extract_embeddings:
                frame.embedding = embs[i]
        return item