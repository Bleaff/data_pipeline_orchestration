"""Profiling utilities for NEUDC system.

This module defines two context managers/decorators:
- `Profile` for measuring execution time, optionally with CUDA sync and logging
- `NoProfile` to disable profiling temporarily
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Self

import torch
from numba import jit

from .logger import LOGGER, USE_NUMBA

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from cuda import cuda, cudart

    CUDA_PROFILE_ENABLE = True
except ImportError:
    CUDA_PROFILE_ENABLE = False
    LOGGER.info("Cuda library is not installed. Check your installation carefully.")

__all__ = ("NoProfile", "Profile", "conditional_jit", "get_profile", "toggle_jit")


class Profile(contextlib.ContextDecorator):
    """A profiling class to measure and accumulate the execution time of code blocks or functions.

    Can also log the average execution time after a specified number of calls, including the function name.

    Usage as a decorator:
        @Profile(freq=10, use_cuda=True)
        def my_func():
            # code to profile

    Usage as a context manager:
        with Profile(freq=10, use_cuda=True, name="my_block") as p:
            # code to profile
    """

    enabled = True  # Class-level flag to enable/disable profiling globally

    def __init__(
        self,
        *,
        t: float = 0.0,
        use_cuda: bool = True,
        use_torch: bool = False,
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
            freq (int, optional): Log frequency (calls). None means no logging.
            max_calls (int): Maximum calls before reset. Defaults to 100_000.
            name (str, optional): Name of the profiling scope. Defaults to None.

        """
        self.t = t
        self.use_cuda = use_cuda
        self.use_torch = use_torch
        self.freq = freq
        self.max_calls = max_calls
        self.call_count = 0
        self.start = 0.0
        self.dt = 0.0
        self.func_name = name

    def __call__(self, func: Callable) -> Callable:  # type: ignore[override]
        """Allow this class to be used as a decorator."""
        self.func_name = self.func_name or func.__name__

        @wraps(func)
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

            if self.freq and (self.call_count % self.freq == 0):
                avg_time = self.t / self.call_count
                LOGGER.info(f"Average execution time for '{self.func_name}': {avg_time:.6f} s")

                if self.call_count > self.max_calls:
                    LOGGER.info(f"Stats for '{self.func_name}' has been reset.")
                    self.call_count = 0
                    self.t = 0.0

    def __str__(self) -> str:
        """Return a human-readable string of the accumulated elapsed time."""
        return f"Elapsed time for '{self.func_name}' is {self.t} s"

    def time(self) -> float:
        """Get the current time, synchronizing with CUDA if a device is actually present.

        The sync decision is made at call time (not decoration time), so profiling a
        model that ends up on CPU is a no-op instead of spamming errors, while GPU runs
        are correctly synchronized before the timestamp is taken.
        """
        if self.use_cuda:
            if CUDA_PROFILE_ENABLE:
                (err,) = cudart.cudaDeviceSynchronize()
                if err != cuda.CUresult.CUDA_SUCCESS:
                    LOGGER.error("CUDA device synchronization failed.")
            elif self.use_torch and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except (AttributeError, RuntimeError) as e:
                    LOGGER.exception("Torch CUDA sync failed.", exc_info=e)

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

        @wraps(func)
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

# Registry to track toggleable functions
_JIT_REGISTRY: dict[int, dict[str, Any]] = {}


# Public API (exported via __all__); changing the call convention could break external callers.
def toggle_jit(enable: bool) -> None:  # noqa: FBT001
    """Globally enable/disable JIT compilation at runtime."""
    global USE_NUMBA  # noqa: PLW0603 -- USE_NUMBA is deliberately module-level global state, toggled process-wide.
    old_setting = USE_NUMBA
    USE_NUMBA = enable

    # Re-register all tracked functions
    for func_info in _JIT_REGISTRY.values():
        module = func_info["module"]
        func_name = func_info["name"]
        original_func = func_info["original"]
        jit_args = func_info["jit_args"]
        jit_kwargs = func_info["jit_kwargs"]

        # Create new implementation
        if enable:
            new_func = jit(*jit_args, **jit_kwargs)(original_func)
            LOGGER.info(f"JIT ENABLED for {func_name}")
        else:
            new_func = original_func
            LOGGER.info(f"JIT DISABLED for {func_name}")

        # Replace function in module
        setattr(module, func_name, new_func)

    LOGGER.info(
        f"JIT globally {'ENABLED' if enable else 'DISABLED'} "
        f"(previously {'enabled' if old_setting else 'disabled'})",
    )


def conditional_jit(*args: Any, **kwargs: Any) -> Callable:
    """Build a decorator that JIT-compiles the wrapped function, toggleable at runtime."""
    turn_on = kwargs.pop("turn_on", USE_NUMBA)

    def decorator(func: Callable) -> Callable:
        # Store original function and parameters
        module = sys.modules[func.__module__]
        func_id = id(func)
        _JIT_REGISTRY[func_id] = {
            "module": module,
            "name": func.__name__,
            "original": func,
            "jit_args": args,
            "jit_kwargs": kwargs,
            "turn_on": turn_on,
        }

        # Apply initial JIT setting
        if turn_on:
            LOGGER.info(f"JIT ENABLED for {func.__name__}")
            return jit(*args, **kwargs)(func)
        LOGGER.info(f"JIT DISABLED for {func.__name__}")
        return func

    return decorator


_PROFILE_REGISTRY: dict[str, Profile] = {}


def get_profile(name: str, **kwargs: Any) -> Profile:
    """Get a named Profile instance, create if missing."""
    if name not in _PROFILE_REGISTRY:
        _PROFILE_REGISTRY[name] = Profile(**kwargs, name=name)
    return _PROFILE_REGISTRY[name]
