"""ProcessEmbeddingInference: post-process embedding vectors to select representative frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from neudc.core.communication.messaging.types import Batch, Frame
from neudc.core.node.model.base_batch_process_inference import BaseBatchProcessInference
from neudc.utils import LOGGER


class ProcessEmbeddingInference(BaseBatchProcessInference):
    """Post-processes embedding vectors to select non-duplicate frames.

    Expects a configuration dict in args[0] with keys:
      - eps (float): DBSCAN epsilon parameter (cosine distance). Default: 0.015.
      - min_samples (int): minimum cluster size for DBSCAN. Default: 10.
      - num_extremes (int): number of points farthest from the cluster medoid
                            to include per cluster. Default: 2.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize caches and clustering parameters.

        Args:
        ----
            *args: positional arguments forwarded to BaseBatchProcessInference
            **kwargs: keyword arguments, expects 'model_config' dict

        """
        self.emb_cache: dict[str, dict[int, np.ndarray]] = {}
        self.frame_cache: dict[str, dict[int, Frame]] = {}

        self.eps = kwargs["model_config"].get("eps", 0.015)
        self.min_samples = kwargs["model_config"].get("min_samples", 2)
        self.num_extremes = kwargs["model_config"].get("num_extremes", 1)
        super().__init__(*args, **kwargs)

    def postprocess_result(self, result: np.ndarray, item: Batch) -> Batch | None | Frame:
        """Accumulate embeddings and frame metadata, then cluster when complete.

        Parameters
        ----------
        result : np.ndarray
            Embedding vectors of shape (batch_size, D).
        item : Batch[Frame]
            Batch of Frame objects corresponding to each embedding.

        Returns
        -------
        Optional[Batch[Frame]]
            A Batch of selected unique frames for any source that has
            reached its last_id, or None if not yet ready.

        """
        for emb, frame_item in zip(result, item):
            src = str(Path(frame_item.source_frame).parent)
            frame_id = frame_item.frame_id
            last_id = frame_item.frame_id_last

            self.emb_cache.setdefault(src, {})[frame_id] = emb
            self.frame_cache.setdefault(src, {})[frame_id] = frame_item
            if len(self.emb_cache[src]) == last_id:
                return self.cluster_and_select(src)
        return None

    def cluster_and_select(self, source: str) -> Batch | None | Frame:  # noqa: C901
        """1) Run DBSCAN (cosine) on all cached embeddings for the source.

        2) For each cluster label >= 0:
             - pick the medoid (closest to centroid)
             - pick the point farthest from that medoid
             - greedily pick up to self.num_extremes further points by
               farthest-first: each new point maximizes its minimum
               distance to all already selected points
           For label == -1 (noise), include all frames.
        3) Clear caches for the source.
        4) Deduplicate by (source_frame, frame_id).
        """
        emb_dict = self.emb_cache.pop(source)
        frame_dict = self.frame_cache.pop(source)
        fids = sorted(emb_dict.keys())
        all_emb = np.stack([emb_dict[fid] for fid in fids], axis=0)

        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine").fit_predict(all_emb)

        unique = Batch(frames=[])

        for lbl in sorted(set(labels)):
            members = [fid for fid, lb in zip(fids, labels) if lb == lbl]

            if lbl == -1:
                unique.frames.extend(frame_dict[fid] for fid in members)
                continue

            idxs = [fids.index(fid) for fid in members]
            embs = all_emb[idxs, :]  # shape (k, D)
            k = embs.shape[0]

            centroid = embs.mean(axis=0)
            d2c = np.linalg.norm(embs - centroid[None, :], axis=1)

            medoid_idx = int(np.argmin(d2c))
            unique.frames.append(frame_dict[members[medoid_idx]])

            if np.allclose(d2c, 0):
                continue

            dist_to_medoid = np.linalg.norm(embs - embs[medoid_idx : medoid_idx + 1, :], axis=1)
            far_idx = int(np.argmax(dist_to_medoid))
            if far_idx != medoid_idx:
                unique.frames.append(frame_dict[members[far_idx]])

            diffs = embs[:, None, :] - embs[None, :, :]
            pdist = np.linalg.norm(diffs, axis=2)

            selected = {medoid_idx, far_idx}
            while len(selected) < 1 + self.num_extremes:
                best_j, best_dist = None, -1.0
                for j in range(k):
                    if j in selected:
                        continue
                    dist_to_set = min(pdist[j, i] for i in selected)
                    if dist_to_set > best_dist:
                        best_dist = dist_to_set
                        best_j = j
                if best_j is None:
                    break
                selected.add(best_j)
                unique.frames.append(frame_dict[members[best_j]])

        seen = set()
        filtered = Batch(frames=[])
        for frame in unique.frames:
            key = (frame.source_frame, frame.frame_id)
            if key not in seen:
                seen.add(key)
                filtered.frames.append(frame)

        return filtered
