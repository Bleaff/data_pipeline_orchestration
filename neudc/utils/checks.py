"""Small validation/compatibility helpers shared across model and profiling code."""

from __future__ import annotations

from typing import Any

import torch

from neudc.utils import LOGGER, WINDOWS


def check_class_names(
    names: list | dict,
) -> dict:
    """Check class names.

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.
    """
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            msg = (
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
            raise KeyError(
                msg,
            )
    return names


def default_class_names() -> dict[int, str]:
    """Return numerical fallback class names (class0, class1, ...) for up to 999 classes."""
    return {i: f"class{i}" for i in range(999)}  # return default if above errors


def to_tuple(data: int | tuple[int, int]) -> tuple[int, int]:
    """Convert 0d or 1d data to 1d copying the data once."""
    if isinstance(data, int):
        return (data, data)
    return data


def torch_compile(*args: Any, **kwargs: Any) -> Any:
    """Safe torch.compile with backward compatibility for PyTorch 1.x."""
    if not hasattr(torch, "compile"):
        # Backward compatibility for PyTorch 1.x
        LOGGER.warning(
            "PyTorch 1.x will no longer be supported by NeuDC. Please upgrade to PyTorch 2.x.",
            DeprecationWarning,
        )
        if args and isinstance(args[0], torch.nn.Module):
            return args[0]
        return torch.jit.script
    if WINDOWS:
        # torch.compile is not supported on Windows
        # https://github.com/orgs/pytorch/projects/27
        LOGGER.warning(
            "Windows does not yet support torch.compile and the performance will be affected.",
            RuntimeWarning,
        )
        if args and isinstance(args[0], torch.nn.Module):
            return args[0]
        return lambda x: x
    return torch.compile(*args, **kwargs)
