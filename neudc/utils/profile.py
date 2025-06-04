"""Profiling utilities for NEUDC system.

This module defines two context managers/decorators:
- `Profile` for measuring execution time, optionally with CUDA sync and logging
- `NoProfile` to disable profiling temporarily
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any, Callable, Self

import torch

from .logger import LOGGER

if TYPE_CHECKING:
    import logging


try:
    import cuda

    CUDA_PROFILE_ENABLE = True
except ImportError:
    CUDA_PROFILE_ENABLE = False
    LOGGER.info("Cuda library is not installed. Check your installation carefully.")

if CUDA_PROFILE_ENABLE:
    from cuda import cuda, cudart

__all__ = ("Profile", "NoProfile")


class Profile(contextlib.ContextDecorator):
    """A profiling class to measure and accumulate the execution time of code blocks or functions.

    Can also log the average execution time after a specified number of calls, including the function name.

    Usage as a decorator:
        @Profile(logger=my_logger, freq=10, use_cuda=True)
        def my_func():
            # code to profile

    Usage as a context manager:
        with Profile(logger=my_logger, freq=10, use_cuda=True, name="my_block") as p:
            # code to profile
    """

    enabled = True  # Class-level flag to enable/disable profiling globally

    def __init__(
        self,
        *,
        t: float = 0.0,
        use_cuda: bool = True,
        use_torch: bool = False,
        logger: logging.Logger | None = None,
        freq: int | None = None,
        max_calls: int = 100_000,
        name: str | None = None,
    ) -> None:
        """Initialize the Profile class.

        Args:
        ----
            t (float): Initial accumulated time. Defaults to 0.0.
            use_cuda (bool): Whether to synchronize with CUDA device. Defaults to True.
            use_torch (bool): Whether to use torch.cuda for sync if CUDA_PROFILE_ENABLE is False.
            logger (Logger, optional): Logger to log profile results.
            freq (int, optional): Log frequency (calls). None means no logging.
            max_calls (int): Maximum calls before reset. Defaults to 100_000.
            name (str, optional): Name of the profiling scope. Defaults to None.

        """
        self.t = t
        self.use_cuda = use_cuda
        self.use_torch = use_torch
        self.logger = logger
        self.freq = freq
        self.max_calls = max_calls
        self.call_count = 0
        self.start = 0.0
        self.dt = 0.0
        self.func_name = name

    def __call__(self, func: Callable) -> Callable:  # type: ignore[override]
        """Allow this class to be used as a decorator."""
        self.func_name = self.func_name or func.__name__

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not Profile.enabled:
                return func(*args, **kwargs)
            with self:
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self) -> Self:
        """Start timing."""
        if Profile.enabled:
            self.start = self.time()
        return self

    def __exit__(self, _type: type[BaseException] | None, _value: BaseException | None, _traceback: object) -> None:
        """Stop timing, accumulate time, and possibly log average execution time."""
        if Profile.enabled and self.start:
            self.dt = self.time() - self.start
            self.t += self.dt
            self.call_count += 1

            if self.logger and self.freq and (self.call_count % self.freq == 0):
                avg_time = self.t / self.call_count
                self.logger.info(f"Average execution time for '{self.func_name}': {avg_time:.6f} s")

                if self.call_count > self.max_calls:
                    self.logger.info(f"Stats for '{self.func_name}' has been reset.")
                    self.call_count = 0
                    self.t = 0.0

    def __str__(self) -> str:
        """Return a human-readable string of the accumulated elapsed time."""
        return f"Elapsed time for '{self.func_name}' is {self.t} s"

    def time(self) -> float:
        """Get the current time, synchronizing with CUDA if needed."""
        if self.use_cuda:
            sync_success = False
            if CUDA_PROFILE_ENABLE:
                (err,) = cudart.cudaDeviceSynchronize()
                sync_success = err == cuda.CUresult.CUDA_SUCCESS
            elif self.use_torch:
                try:
                    torch.cuda.synchronize()
                    sync_success = True
                except (ImportError, AttributeError, RuntimeError) as e:
                    if self.logger:
                        self.logger.exception("Torch CUDA sync failed.", exc_info=e)

            if not sync_success and self.logger:
                self.logger.error("CUDA device synchronization failed.")

        return time.time()


class NoProfile(contextlib.ContextDecorator):
    """A context manager and decorator that disables the Profile class's functionality within its scope or decorated function.

    When used, Profile.enabled is set to False, preventing
    any profiling from taking place inside the decorated function or context block, and then
    restored upon exit.
    """

    def __call__(self, func: Callable) -> Callable:  # type: ignore[override]
        """Call the decorator that disables profiling for the wrapped function.

        Args:
        ----
            func (Callable): The function to be wrapped.

        Returns:
        -------
            Callable: The wrapped function with profiling disabled.

        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Disable profiling before function call and restore it after."""
            prev_state = Profile.enabled
            Profile.enabled = False
            res = func(*args, **kwargs)
            Profile.enabled = prev_state
            return res

        return wrapper

    def __enter__(self) -> Self:
        """Enter the context manager, disabling the Profile."""
        self.prev_state = Profile.enabled
        Profile.enabled = False
        return self

    def __exit__(self, _type: type[BaseException] | None, _value: BaseException | None, _traceback: object) -> None:
        """Exit the context manager, restoring previous Profile state."""
        Profile.enabled = self.prev_state


NoProfile = NoProfile()  # type: ignore[assignment, misc]
