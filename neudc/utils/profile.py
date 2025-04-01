import contextlib
import logging
import time
from typing import Callable, Optional

import torch

from neudc.utils import CUDA_PROFILE_ENABLE

if CUDA_PROFILE_ENABLE:
    from cuda import cuda, cudart

__all__ = ("Profile", "NoProfile")


class Profile(contextlib.ContextDecorator):
    """
    A profiling class to measure and accumulate the execution time of code blocks or functions.
    Can also log the average execution time after a specified number of calls, including the function name.

    Usage as a decorator:
        @Profile(logger=my_logger, log_frequency=10, use_cuda=True)
        def my_func():
            # code to profile

    Usage as a context manager:
        with Profile(logger=my_logger, log_frequency=10, use_cuda=True, name="my_block") as p:
            # code to profile
        # p contains total elapsed time and will have logged stats if frequency was reached
    """

    enabled = True  # Class-level flag to enable/disable profiling globally

    def __init__(
        self,
        t: float = 0.0,
        use_cuda: bool = True,
        use_torch: bool = False,
        logger: Optional[logging.Logger] = None,
        freq: Optional[int] = None,
        max_calls: int = 1e5,
        name: Optional[str] = None,
    ) -> "Profile":
        """
        Initialize the Profile class.

        Args:
            t (float): Initial accumulated time. Defaults to 0.0.
            use_cuda (bool): Whether to synchronize with CUDA device. Defaults to False.
            logger (logging.Logger): Logger instance to log average times. Defaults to None (no logging).
            log_frequency (int): Number of calls after which to log average time. Defaults to None (no logging).
            max_num_calls (int): Number of maximum calls, after it stats will be reset.
            name (str): Optional name for the profiled code (if used as context manager). Defaults to None (will use 'context').
        """
        self.t = t
        self.use_cuda = use_cuda
        self.use_torch = use_torch
        self.logger = logger
        self.freq = freq
        self.max_calls = max_calls
        self.call_count = 0
        self.start = None
        self.dt = None
        self.func_name = name

    def __call__(self, func: Callable) -> Callable:
        """Allow this class to be used as a decorator."""
        # Capture the function name from the decorated function
        self.func_name = self.func_name or func.__name__

        def wrapper(*args, **kwargs) -> Callable:
            if not Profile.enabled:
                return func(*args, **kwargs)
            with self:  # Use the context manager to measure time
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self) -> "Profile":
        """Start timing."""
        if Profile.enabled:
            self.start = self.time()
        return self

    def __exit__(self, type, value, traceback) -> None:  # noqa
        """Stop timing, accumulate time, and possibly log average execution time."""
        if Profile.enabled and self.start:
            self.dt = self.time() - self.start  # delta-time
            self.t += self.dt  # accumulate total time
            self.call_count += 1

            # Log if conditions are met
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
                except:
                    pass

            if not sync_success and self.logger:
                self.logger.error("CUDA device synchronization failed.")

        return time.time()


class NoProfile(contextlib.ContextDecorator):
    """
    A context manager and decorator that disables the Profile class's functionality within
    its scope or decorated function. When used, Profile.enabled is set to False, preventing
    any profiling from taking place inside the decorated function or context block, and then
    restored upon exit.
    """

    def __call__(self, func: Callable) -> Callable:
        """
        Allow this class to be used as a decorator that disables profiling for the decorated function.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: The wrapped function that runs with Profile disabled.
        """

        def wrapper(*args, **kwargs) -> Callable:
            """
            Wrapper function that disables profiling before the function call and restores the previous
            profiling state afterwards.

            Args:
                args: Positional arguments matching the decorated function.
                kwargs: Keyword arguments matching the decorated function.

            Returns:
                Any: The result of the decorated function.
            """

            # Disable profiler
            prev_state = Profile.enabled
            Profile.enabled = False

            res = func(*args, **kwargs)

            # Enable profiler
            Profile.enabled = prev_state

            return res

        return wrapper

    def __enter__(self) -> "NoProfile":
        """
        Enter the context manager, disabling the Profile.

        Returns:
            NoProfile(): The context manager instance.
        """

        self.prev_state = Profile.enabled
        Profile.enabled = False
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        """
        Exit the context manager, restoring the previous Profile state.

        Args:
            type (Type[BaseException]): The exception type if an exception is raised in the context.
            value (BaseException): The exception instance if an exception is raised in the context.
            traceback (TracebackType): The traceback associated with an exception raised in the context.
        """

        Profile.enabled = self.prev_state


NoProfile = NoProfile()
