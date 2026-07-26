"""Batched object-detection inference node: converts raw model output into `Box` annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from neudc.core.communication.messaging.types import Batch, Box
from neudc.core.node.model.base_batch_process_inference import BaseBatchProcessInference
from neudc.profilers.postprocess_detector_profiler import PostprocessProfiler

if TYPE_CHECKING:
    from neudc.nn.models.det.base import BaseDetector


class ProcessDetBatchInference(BaseBatchProcessInference):
    """Batch process-inference node running a detector model over a batch of frames."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the node, forwarding arguments to `BaseBatchProcessInference`."""
        super().__init__(*args, **kwargs)
        model_cfg = kwargs.get("model_config", {})
        self.extract_embeddings: bool = model_cfg.get("extract_embeddings", False)

    @PostprocessProfiler()
    def postprocess_result(self, result: Any, item: Batch) -> Batch:
        """Convert raw detector output into `Box` annotations and attach them to each frame."""
        assert self.model is not None
        # Cast, not isinstance: tests may exercise this with a lightweight duck-typed
        # stub (just a `.names` dict), not a real BaseDetector subclass.
        model = cast("BaseDetector", self.model)
        if self.extract_embeddings:
            preds, embs = result
        else:
            preds, embs = result, None

        for i, (dets, frame) in enumerate(zip(preds, item.frames, strict=True)):
            frame.boxes.extend(
                [
                    Box(
                        x1=int(x),
                        y1=int(y),
                        x2=int(x2),
                        y2=int(y2),
                        score=float(score),
                        class_id_name=model.names[int(cls_id)],
                        class_id_num=int(cls_id),
                    )
                    for x, y, x2, y2, score, cls_id in dets
                ]
            )

            if self.extract_embeddings:
                frame.embedding = embs[i]
        return item
