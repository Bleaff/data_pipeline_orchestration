"""Regression test: single-frame detection postprocess builds valid Box objects (Stage 3).

ProcessDetInference.postprocess_result used to construct Box(class_id=...), which the
schema rejects (it requires class_id_name + class_id_num). This verifies the fix.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging.types import Frame
from neudc.core.node.model.proc_det_inference import ProcessDetInference


class _StubModel:
    names: ClassVar = {0: "cat", 1: "dog"}


def _frame() -> Frame:
    return Frame(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp=0.0,
        source_frame="x",
        frame_id=0,
        boxes=[],
        frame_id_last=1,
        drop=False,
    )


def test_postprocess_builds_valid_boxes() -> None:
    mailbox = ZMQMailbox()  # stopped during node construction (process-node hand-off)
    node = ProcessDetInference(model_config={}, mailbox=mailbox, id="det")
    node.model = _StubModel()

    result = [np.array([[10, 20, 30, 40, 0.9, 1]], dtype=float)]
    out = node.postprocess_result(result, _frame())

    assert len(out.boxes) == 1
    box = out.boxes[0]
    assert box.class_id_name == "dog"
    assert box.class_id_num == 1
    assert (box.x1, box.y1, box.x2, box.y2) == (10, 20, 30, 40)
    assert abs(box.score - 0.9) < 1e-6
