"""Regression test for ProcessEmbeddingInference.cluster_and_select.

`far_idx` was only assigned inside the `not np.allclose(d2c, 0)` branch but read
unconditionally a few lines later, so a cluster whose points all coincide with their
centroid (e.g. a single-frame cluster) raised UnboundLocalError instead of selecting it.
"""

from __future__ import annotations

import numpy as np

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox
from neudc.core.communication.messaging.types import Frame
from neudc.core.node.model.proc_embedding_inference import ProcessEmbeddingInference


def _frame(frame_id: int) -> Frame:
    return Frame(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp=0.0,
        source_frame="videos/clip_a",
        frame_id=frame_id,
        boxes=[],
        frame_id_last=1,
        drop=False,
    )


def _node() -> ProcessEmbeddingInference:
    mailbox = ZMQMailbox()  # stopped during node construction (process-node hand-off)
    return ProcessEmbeddingInference(
        batch_size=1,
        model_config={"eps": 0.015, "min_samples": 1, "num_extremes": 1},
        mailbox=mailbox,
        id="embedding",
    )


def test_cluster_and_select_handles_single_point_cluster() -> None:
    node = _node()
    source = "videos/clip_a"
    node.frame_cache[source] = {0: _frame(0)}
    node.emb_cache[source] = {0: np.ones(8, dtype=np.float32)}

    result = node.cluster_and_select(source)

    assert len(result.frames) == 1
    assert result.frames[0].frame_id == 0
