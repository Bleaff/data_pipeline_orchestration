from abc import ABC, abstractmethod

from neudc.utils.types import BoolMask, UInt8HWC

__all__ = ("BaseFilter",)


class BaseFilter(ABC):
    """Base class for all filters."""

    @abstractmethod
    def __init__(
        self,
        separate_by_loader: bool = False,
    ) -> "BaseFilter":
        """
        Args:
            separate_by_loader (bool): Whether to separate seen hashes by loader.
        """
        self.separate_by_loader = separate_by_loader

    @abstractmethod
    def __call__(self, ims: list[UInt8HWC], loader_ids: list[int]) -> BoolMask:
        """
        Args:
            ims (list): List of images.
            loader_ids (list): List of loader IDs.
        Returns:
            BoolMask: Boolean mask indicating image should be saved or filtered out.
        """
        ...
