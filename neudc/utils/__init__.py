"""Utilities and configuration for the NEUDC system.

This module provides logging setup, platform-specific constants,
and optional profiling/logging integration with external libraries
such as TensorRT and ONNX.
"""

from __future__ import annotations

import logging
import platform
import sys
from logging import Logger
from pathlib import Path

import pytz  # type: ignore[import-untyped]

from .profile import NoProfile, Profile

# System metadata and environment flags
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
LOGGING_NAME = "neudc"
VERBOSE = True
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
ARM64 = platform.machine() in {"arm64", "aarch64"}
TIMEZONE = pytz.timezone("Europe/Moscow")
PROFILE_FREQ = 50
NUMBA_DISABLE_JIT = True


def emojis(string: str = "") -> str:
    """Return platform-dependent emoji-safe version of string."""
    return string.encode("utf-8").decode("ascii", "ignore") if WINDOWS else string


def set_logging(
    name: str = "LOGGING_NAME",
    *,
    verbose: bool = True,
    output: str | None = "output.log",
) -> Logger:
    """Set up logging with UTF-8 encoding and configurable verbosity.

    Args:
    ----
        name (str): Name of the logger.
        verbose (bool): If True, sets logging level to INFO, else ERROR.
        output (Optional[str]): File path for log output.

    Returns:
    -------
        Logger: Configured logger instance.

    Notes:
    -----
        - On Windows, attempts to configure UTF-8 for stdout.
        - Adds both stream and rotating file handlers.

    """
    level = logging.INFO if verbose else logging.ERROR
    formatter = logging.Formatter("%(message)s")

    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":

        class CustomFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                """Format log with emoji safety."""
                return emojis(super().format(record))

        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception:  # noqa: BLE001
            formatter = CustomFormatter("%(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False

    if output:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(output, maxBytes=100000, backupCount=10, encoding="utf-8")
        file_formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)
for _logger in ("sentry_sdk", "urllib3.connectionpool"):
    logging.getLogger(_logger).setLevel(logging.CRITICAL + 1)

try:
    import tensorrt as trt

    TENSORRT_ENABLE = True
except ImportError:
    TENSORRT_ENABLE = False
    LOGGER.info("TensorRT library is not installed. Check your installation carefully.")

try:
    CUDA_PROFILE_ENABLE = True
except ImportError:
    CUDA_PROFILE_ENABLE = False
    LOGGER.info("Cuda library is not installed. Check your installation carefully.")

try:
    ONNX_PROFILE_ENABLE = True
except ImportError:
    ONNX_PROFILE_ENABLE = False
    LOGGER.info("ONNX library is not installed. Check your installation carefully.")

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
)
