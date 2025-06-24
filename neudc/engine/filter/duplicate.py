from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .base import BaseFilter

if TYPE_CHECKING:
    from neudc.utils.types import BoolMask, UInt8HWC

__all__ = ("NeuralDuplicateFilter",)


class NeuralDuplicateFilter(BaseFilter):
    def __init__(
        self,
        separate_by_loader: bool = False,
    ) -> NeuralDuplicateFilter:
        super().__init__(separate_by_loader)
        self.cache = defaultdict(list) if separate_by_loader else []

    def __call__(self, ims: list[UInt8HWC], loader_ids: list[int]) -> BoolMask: ...
