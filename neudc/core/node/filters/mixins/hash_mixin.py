"""Hash Filter Mixin."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import imagehash
from PIL import Image

from neudc.utils import LOGGER

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import Frame


class HashFilterMixin:
    """Mixin for filtering out near-duplicate frames based on different hash algorithms.

    Attributes
    ----------
        cache: Mapping from source_frame (str) to a dict of frame_id (int) → hash
               (either imagehash.ImageHash or MD5 hex string).
        delta: Maximum Hamming distance for perceptual hashes to be considered duplicates.
        type_of_hash: One of 'ahash', 'dhash', 'phash', or 'md5'.
        hash_funcs: Mapping from hash_type to the corresponding hashing method.
        hash_size: Size parameter for aHash/dHash/pHash algorithms.

    """

    HASH_TYPES: ClassVar = ["colorhash", "dhash", "phash", "average_hash", "phash_simple", "dhash_vertical", "whash"]

    def __init__(
        self,
        delta: int = 5,
        hash_type: str = "dhash",
        hash_size: int = 8,
    ) -> HashFilterMixin:
        """Initialize the mixin.

        Args:
        ----
            delta: Maximum Hamming distance for perceptual hash comparisons.
            hash_type: Which hash to compute—'ahash', 'dhash', 'phash', or 'md5'.
            hash_size: size of hash for aHash/dHash/pHash.

        """
        self.cache: dict[str, dict[int, imagehash.ImageHash | str]] = defaultdict(dict)
        self.delta: int = delta
        self.hash_size: int = hash_size

        if hash_type not in self.HASH_TYPES:
            LOGGER.warning(f"WARNING ⚠️ Invalid hash type: {hash_type}. Switching to default `dhash`.")
            hash_type = "dhash"

        self.hash_func = getattr(imagehash, hash_type)

    def process(self, frame: Frame) -> Frame | None:
        """Filter out near-duplicate frames.

        1. Compute the current frame's hash.
        2. Compare against all stored hashes for this frame.source_frame:
           - For perceptual hashes, drop if Hamming distance ≤ delta.
        3. If no duplicate found, store the new hash and return the frame.
        4. Otherwise return None.

        Args:
        ----
            frame: Incoming Frame to process.

        Returns:
        -------
            The same Frame if unique; otherwise None.

        """
        pil_image = Image.fromarray(frame.image)
        current_hash = self.hash_func(pil_image, hash_size=self.hash_size)

        key_folder = Path(frame.source_frame).parent.name

        for old_hash in self.cache[key_folder].values():
            if (current_hash - old_hash) <= self.delta:
                frame.drop = True
                return frame
        self.cache[key_folder][frame.frame_id] = current_hash
        return frame
