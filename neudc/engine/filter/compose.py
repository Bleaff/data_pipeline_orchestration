import itertools

import numpy as np

from neudc.utils.types import BoolMask, UInt8HWC

from .base import BaseFilter

__all__ = ("ComposeFilter",)


class ComposeFilter:
    def __init__(
        self,
        filters: list[BaseFilter],
    ) -> "ComposeFilter":
        """
        Args:
            filters (list): List of filters to compose.
        """
        self.filters = filters

    def __call__(self, ims: list[UInt8HWC], loader_ids: list[int]) -> BoolMask:
        """
        Args:
            ims (list): List of images.
            loader_ids (list): List of loader IDs.
        Returns:
            BoolMask: Boolean mask indicating image should be saved or filtered out.
        """
        final_mask = np.ones(len(ims), dtype=bool)
        for im_filter in self.filters:
            # Filter images and loader IDs based on the final mask
            ims_filtered = list(itertools.compress(ims, final_mask))
            loader_ids_filtered = list(itertools.compress(loader_ids, final_mask))
            # Get current mask result only for valid images
            current_mask = im_filter(ims_filtered, loader_ids_filtered)
            # Use boolean indexing to update final_mask
            final_mask[final_mask] &= current_mask
        return final_mask
