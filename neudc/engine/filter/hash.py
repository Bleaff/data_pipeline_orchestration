import hashlib
from collections import defaultdict

import numpy as np

from neudc.utils.types import BoolMask, UInt8HWC

from .base import BaseFilter

__all__ = ("HashFilter",)


class HashFilter(BaseFilter):
    def __init__(
        self,
        separate_by_loader: bool = False,
    ) -> "HashFilter":
        """
        Args:
            separate_by_loader (bool): Whether to separate seen hashes by loader.
        """
        super().__init__(separate_by_loader)
        self.cache = defaultdict(set) if separate_by_loader else set()

    def __call__(self, ims: list[UInt8HWC], loader_ids: list[int]) -> BoolMask:
        """
        Args:
            ims (list): List of images.
            loader_ids (list): List of loader IDs.
        Returns:
            BoolMask: Boolean mask indicating if the image is unique.
        """
        unique_image = np.ones(len(ims), dtype=bool)
        for i, (im, loader_id) in enumerate(zip(ims, loader_ids)):
            im_hash = hashlib.md5(im.tobytes()).hexdigest()
            container = self.cache[loader_id] if self.separate_by_loader else self.cache
            if im_hash in container:
                unique_image[i] = False
            else:
                container.add(im_hash)

        return unique_image
