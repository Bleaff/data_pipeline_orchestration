"""Mixin implementing per-folder frame collection and uncertainty-diversity active-learning selection."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from neudc.core.communication.messaging.types import Batch, Frame
from neudc.utils import LOGGER

# Below this uncertainty spread, treat all frames as equally uncertain (avoid a division blow-up).
_UNCERTAINTY_SPREAD_EPS = 1e-8


class ActiveLearningMixin:
    """Collects frames coming from different folders, waits until every folder is complete.

    Once every folder (i.e. video / dataset chunk) has delivered all its images, selects a
    subset for annotation with an *uncertainty x diversity* strategy.
    """

    def __init__(self, num_to_select: int = 20, conf_strategy: str = "sum", conf_weight: float = 1.0) -> None:
        """Initialize the active-learning mixin.

        Args:
        ----
            num_to_select (int): How many images to pick once all expected frames are gathered.
            conf_strategy (str): How to aggregate box-confidence values into a single
                "uncertainty" number per image. One of {"sum", "avg", "max"}.
            conf_weight (float): Relative importance of uncertainty vs. embedding distance
                when the final score is computed.

        """
        self.num_to_select: int = num_to_select
        self.conf_strategy: str = conf_strategy
        self.conf_weight: float = conf_weight

        self.cache: dict[str, dict[int, Frame]] = defaultdict(dict)
        self.expected: dict[str, int] = {}
        self.remaining: dict[str, set[int]] = {}

    def process(self, frame: Frame) -> Batch[Frame] | Frame | None:
        """Handle one incoming frame.

        - Add the frame to the per-directory cache.
        - Once *all* directories have delivered all their frames (`ready`),
          run active-learning selection and return the chosen subset.
        - Otherwise return `None` so the pipeline knows to keep waiting.
        """
        src_dir = Path(frame.source_frame).parent.as_posix()  #'assets/images'
        if src_dir not in self.remaining:
            last_id = frame.frame_id_last
            self.remaining[src_dir] = set(range(last_id))
            self.expected[src_dir] = last_id

        self.cache[src_dir][frame.frame_id] = frame
        self.remaining[src_dir].discard(frame.frame_id)

        left_per_dir_cnt = {k: len(v) for k, v in self.remaining.items()}
        left_per_dir_ids = {k: sorted(v) for k, v in self.remaining.items()}
        left_total = sum(left_per_dir_cnt.values())

        LOGGER.info(
            "Waiting frames: total %d |  pending IDs %s",
            left_total,
            left_per_dir_ids,
        )

        ready = all(len(frames) == self.expected[src] for src, frames in self.cache.items())

        if not ready:
            return None

        LOGGER.info("All frames collected - starting active-learning")

        all_frames: list[Frame] = []

        for frames in self.cache.values():
            for k in sorted(frames):
                f = frames[k]
                if not getattr(f, "drop", False):
                    all_frames.append(f)

        selected = self.active_select(all_frames)

        self.cache.clear()
        self.expected.clear()
        self.remaining.clear()

        return selected

    def active_select(self, frames: list[Frame]) -> Batch[Frame]:
        """Run uncertainty-diversity sampling over the collected frames.

        1.  Build an embedding matrix `embs` of shape (N, C).
        2.  Compute per-image uncertainty using self.conf_strategy on the box
            confidence scores.  Normalise into [0, 1].
        3.  Greedily pick `num_to_select` images that maximise
            distance-to-nearest-selected + conf_weight x uncertainty.
        """
        n_total = len(frames)
        n_take = min(self.num_to_select, n_total)

        if n_take <= 0:
            return Batch(frames=[])

        # Active-learning selection assumes upstream embedding extraction already ran, so every
        # frame here carries a real embedding; Frame.embedding is optional only for other paths.
        embs = np.vstack([f.embedding for f in frames])  # type: ignore[misc]
        uncert = np.zeros(n_total, dtype=float)
        for i, f in enumerate(frames):
            confs = np.array([b.score for b in f.boxes]) if f.boxes else np.array([])
            if confs.size == 0:
                uncert[i] = 1.0
            else:
                inv = 1.0 - confs
                if self.conf_strategy == "sum":
                    uncert[i] = inv.sum()
                elif self.conf_strategy == "avg":
                    uncert[i] = inv.mean()
                elif self.conf_strategy == "max":
                    uncert[i] = inv.max()
                else:
                    msg = f"Unknown conf_strategy {self.conf_strategy}"
                    raise ValueError(msg)
        if np.ptp(uncert) > _UNCERTAINTY_SPREAD_EPS:
            uncert = (uncert - uncert.min()) / np.ptp(uncert)

        selected_idx: list[int] = []
        for _ in range(n_take):
            if selected_idx:
                chosen = embs[selected_idx]
                d = cdist(embs, chosen).min(axis=1)
            else:
                d = np.zeros(n_total)

            scores = d + self.conf_weight * uncert
            scores[selected_idx] = -np.inf
            best = int(np.argmax(scores))
            selected_idx.append(best)

        LOGGER.info(f"Selected {len(selected_idx)}/{n_total} frames")
        return Batch(frames=[frames[i] for i in selected_idx])
