"""Utilities and configuration for the NEUDC system.

This module provides logging setup, platform-specific constants,
and optional profiling/logging integration with external libraries
such as TensorRT and ONNX.
"""

from __future__ import annotations

import importlib.util
import logging
import platform
from pathlib import Path

import pytz  # type: ignore[import-untyped]

from neudc.utils.logger import LOGGER, USE_NUMBA
from neudc.utils.profile import CUDA_PROFILE_ENABLE, NoProfile, Profile, conditional_jit, get_profile, toggle_jit

# System metadata and environment flags
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
ARM64 = platform.machine() in {"arm64", "aarch64"}
TIMEZONE = pytz.timezone("Europe/Moscow")
PROFILE_FREQ = 10


try:
    import tensorrt as trt

    TENSORRT_ENABLE = True
except ImportError:
    TENSORRT_ENABLE = False
    LOGGER.info("TensorRT library is not installed. Check your installation carefully.")

# A presence check, not a real import: nothing here needs the onnxruntime module itself,
# and a bare `import onnxruntime` used only for this check is silently stripped by
# autoflake's unused-import removal (it can't tell the import is load-bearing).
ONNX_PROFILE_ENABLE = importlib.util.find_spec("onnxruntime") is not None
if not ONNX_PROFILE_ENABLE:
    LOGGER.info("ONNX library is not installed. Check your installation carefully.")
# CUDA_PROFILE_ENABLE is computed once in neudc.utils.profile (imported above); no need
# to recompute it here.

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
    "CUDA_PROFILE_ENABLE",
    "LOGGER",
    "ONNX_PROFILE_ENABLE",
    "ROOT",
    "TENSORRT_ENABLE",
    "TIMEZONE",
    "USE_NUMBA",
    "NoProfile",
    "Profile",
    "TensorRTLogger",
    "conditional_jit",
    "get_profile",
    "toggle_jit",
)
