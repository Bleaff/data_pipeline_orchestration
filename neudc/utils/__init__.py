"""Utilities and configuration for the NEUDC system.

This module provides logging setup, platform-specific constants,
and optional profiling/logging integration with external libraries
such as TensorRT and ONNX.
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path

import pytz  # type: ignore[import-untyped]

from .logger import LOGGER, USE_NUMBA
from .profile import CUDA_PROFILE_ENABLE, NoProfile, Profile, conditional_jit, get_profile, toggle_jit

# System metadata and environment flags
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
ARM64 = platform.machine() in {"arm64", "aarch64"}
TIMEZONE = pytz.timezone("Europe/Moscow")
PROFILE_FREQ = 10


try:
    import tensorrt as trt  # noqa

    TENSORRT_ENABLE = True
except ImportError:
    TENSORRT_ENABLE = False
    LOGGER.info("TensorRT library is not installed. Check your installation carefully.")
try:
    import onnxruntime as ort  # noqa

    ONNX_PROFILE_ENABLE = True
except ImportError:
    ONNX_PROFILE_ENABLE = False
    LOGGER.info("ONNX library is not installed. Check your installation carefully.")
try:
    from cuda import cuda  # noqa

    CUDA_PROFILE_ENABLE = True
except ImportError:
    CUDA_PROFILE_ENABLE = False
    LOGGER.info("Cuda library is not installed. Check your installation carefully.")

if TENSORRT_ENABLE:
    TRT_LOGGING_MAP = {
        trt.Logger.INTERNAL_ERROR: logging.CRITICAL,
        trt.Logger.ERROR: logging.ERROR,
        trt.Logger.WARNING: logging.WARNING,
        trt.Logger.INFO: logging.INFO,
        trt.Logger.VERBOSE: logging.DEBUG,
    }

    class TensorRTLogger(trt.Logger):
        """Custom TensorRT logger that routes messages to Python logging."""

        def __init__(self) -> None:
            """Initialize the TensorRT logger."""
            super().__init__(trt.Logger.VERBOSE)

        def log(self, severity: int, msg: str) -> None:
            """Log a message with the specified severity.

            Args:
            ----
                severity (int): Severity level of the message.
                msg (str): Message to log.

            """
            level = TRT_LOGGING_MAP.get(severity, logging.INFO)
            LOGGER.log(level, msg)

else:
    TensorRTLogger = LOGGER  # type: ignore[attr-defined, misc, assignment]

__all__ = (
    "Profile",
    "NoProfile",
    "LOGGER",
    "ROOT",
    "TIMEZONE",
    "TensorRTLogger",
    "TENSORRT_ENABLE",
    "CUDA_PROFILE_ENABLE",
    "ONNX_PROFILE_ENABLE",
    "CUDA_PROFILE_ENABLE",
    "get_profile",
    "USE_NUMBA",
    "conditional_jit",
    "toggle_jit",
)
