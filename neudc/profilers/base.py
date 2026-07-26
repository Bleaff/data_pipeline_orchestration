"""Abstract base class shared by the profiler decorators/context managers in this package."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType


class BaseProfiler(ABC):
    """Common interface for profilers usable both as decorators and as context managers."""

    @abstractmethod
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``func`` so that each call is profiled."""
        raise NotImplementedError

    @abstractmethod
    def __enter__(self) -> Self:
        """Start timing the enclosed block."""
        raise NotImplementedError

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop timing the enclosed block and record the elapsed time."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return the profiler's class name."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Return the same representation as :meth:`__str__`."""
        return self.__str__()
