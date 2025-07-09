from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.cluster import DBSCAN

from neudc.core.communication.messaging.types import Batch, Frame
from neudc.core.node.model.base_batch_process_inference import BaseBatchProcessInference


class ProcessEmbeddingInference(BaseBatchProcessInference):
    """
    Post-processes embedding vectors to select non-duplicate frames.

    Expects a configuration dict in args[0] with keys:
      - eps (float): DBSCAN epsilon parameter (cosine distance). Default: 0.015.
      - min_samples (int): minimum cluster size for DBSCAN. Default: 10.
      - num_extremes (int): number of points farthest from the cluster medoid
                            to include per cluster. Default: 2.
    """

    def __init__(self, *args, **kwargs) -> None:

        self.emb_cache: dict[str, dict[int, np.ndarray]] = {}
        self.frame_cache: dict[str, dict[int, Frame]] = {}
        cfg = {}
        if args and isinstance(args[0], dict):
            cfg = args[0]
        self.eps = cfg.get("eps", 0.015)
        self.min_samples = cfg.get("min_samples", 10)
        self.num_extremes = cfg.get("num_extremes", 2)
        super().__init__(*args, **kwargs)

    def postprocess_result(self, result: Any, item: Batch) -> Optional[Batch] | Frame | None:
        """
        Accumulates embeddings and Frame metadata by source folder.
        As soon as a frame with frame_id == frame_id_last is seen for a source,
        immediately performs clustering and selection of non-duplicates.

        Parameters:
          result: np.ndarray of shape (batch_size, D) — embedding vectors.
          items:  List[Frame], same length as result — frame metadata.

        Returns:
          List[Frame] of selected unique frames for the completed source,
          or None if no source has yet reached its last frame in this batch.
        """
        for emb, frame_item in zip(result, item):
            src = str(Path(frame_item.source_frame).parent)
            frame_id = frame_item.frame_id
            last_id = frame_item.frame_id_last

            self.emb_cache.setdefault(src, {})[frame_id] = emb
            self.frame_cache.setdefault(src, {})[frame_id] = frame_item
            self.logger.debug(f"{frame_id=}/{last_id=}")
            if frame_id == last_id:
                unique = self.cluster_and_select(src)
                return unique
        return

    def cluster_and_select(self, source: str) -> Optional[Batch] | Frame | None:
        """
        1) Runs DBSCAN (cosine metric) on all cached embeddings for the source.
        2) For each cluster label >= 0:
             - add the medoid (point closest to the cluster centroid)
             - add self.num_extremes points farthest from the medoid
           For label == -1 (noise), add all frames.
        3) Clear caches for the source.
        4) Remove any duplicate frames from the final list.

        Returns:
          List[Frame] of non-duplicate representative frames.
        """
        emb_dict = self.emb_cache[source]
        frame_dict = self.frame_cache[source]
        frame_ids = sorted(emb_dict.keys())
        all_embds = np.stack([emb_dict[id] for id in frame_ids], axis=0)

        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine").fit_predict(all_embds)

        unique: Batch[Frame] = Batch(frames=list())
        for cluster_label in sorted(set(labels)):
            idxs = [i for i, lbl in zip(frame_ids, labels) if lbl == cluster_label]

            if cluster_label == -1:
                unique.frames.extend(frame_dict[i] for i in idxs)
                continue

            embs = np.stack([emb_dict[i] for i in idxs], axis=0)
            centroid = embs.mean(axis=0)

            dists = np.linalg.norm(embs - centroid[None, :], axis=1)
            medoid_loc = int(np.argmin(dists))
            unique.frames.append(frame_dict[idxs[medoid_loc]])

            sorted_locs = np.argsort(-dists)
            count = 0
            for loc in sorted_locs:
                if loc == medoid_loc:
                    continue
                unique.frames.append(frame_dict[idxs[loc]])
                count += 1
                if count >= self.num_extremes:
                    break

        del self.emb_cache[source]
        del self.frame_cache[source]

        seen = set()
        filtered: Batch[Frame] = Batch(frames=list())
        for f in unique:
            key = (f.source_frame, f.frame_id)
            if key not in seen:
                seen.add(key)
                filtered.frames.append(f)

        return filtered
