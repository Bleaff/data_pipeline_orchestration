"""Single-frame object-detection inference node: converts raw model output into `Box` annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from neudc.core.communication.messaging.types import Box, Frame
from neudc.core.node.model.base_process_inference import BaseProcessInference
from neudc.profilers.postprocess_detector_profiler import PostprocessProfiler

if TYPE_CHECKING:
    from neudc.nn.models.det.base import BaseDetector


class ProcessDetInference(BaseProcessInference):
    """Process-inference node running a detector model on individual frames."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the node by forwarding all arguments to `BaseProcessInference`."""
        super().__init__(*args, **kwargs)

    @PostprocessProfiler()
    def postprocess_result(self, result: Any, item: Frame) -> Frame:
        """Convert raw detector output into `Box` annotations and attach them to the frame."""
        assert self.model is not None
        # Cast, not isinstance: tests exercise this with a lightweight duck-typed
        # stub (just a `.names` dict), not a real BaseDetector subclass.
        model = cast("BaseDetector", self.model)
        boxes = []
        for box_ in result[0]:
            x, y, x2, y2, score, class_id = box_
            box = Box(
                x1=int(x),
                y1=int(y),
                x2=int(x2),
                y2=int(y2),
                score=float(score),
                class_id_name=model.names[int(class_id)],
                class_id_num=int(class_id),
            )
            boxes.append(box)
        item.boxes.extend(boxes)
        return item
