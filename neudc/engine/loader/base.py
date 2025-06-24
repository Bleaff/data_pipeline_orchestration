from abc import ABC, abstractmethod
from typing import Any

__all__ = ("BaseLoader",)


class BaseLoader(ABC):
    """Base Loader with custom iterations."""

    @abstractmethod
    def __init__(self, path: str, *args, **kwargs) -> "BaseLoader": ...

    @abstractmethod
    def __iter__(self) -> Any: ...

    @abstractmethod
    def __del__(self) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...
