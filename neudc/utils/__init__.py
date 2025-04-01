import logging
import platform
import sys
from logging import Logger
from pathlib import Path
from typing import Optional

import pytz

# System
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
LOGGING_NAME = "neudc"
VERBOSE = True
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
ARM64 = platform.machine() in {"arm64", "aarch64"}  # ARM64 booleans
TIMEZONE = pytz.timezone("Europe/Moscow")  # UTC, Asia/Shanghai, Europe/Berlin
PROFILE_FREQ = 50  # Every <PROFILE_FREQ> calls of the model, the profile will be printed
NUMBA_DISABLE_JIT = True


def emojis(string: str = "") -> str:
    """Return platform-dependent emoji-safe version of string."""
    return string.encode("utf-8").decode("ascii", "ignore") if WINDOWS else string


def set_logging(
    name: str = "LOGGING_NAME",
    verbose: bool = True,
    output: Optional[str] = "output.log",
) -> Logger:
    """
    Sets up logging with UTF-8 encoding and configurable verbosity.

    This function configures logging for the stream-gui library, setting the appropriate logging level and
    formatter based on the verbosity flag and the current process rank. It handles special cases for Windows
    environments where UTF-8 encoding might not be the default.

    Args:
        name (str): Name of the logger. Defaults to "LOGGING_NAME".
        verbose (bool): Flag to set logging level to INFO if True, ERROR otherwise. Defaults to True.

    Examples:
        >>> set_logging(name="stream-gui", verbose=True)
        >>> logger = logging.getLogger("stream-gui")
        >>> logger.info("This is an info message")

    Notes:
        - On Windows, this function attempts to reconfigure stdout to use UTF-8 encoding if possible.
        - If reconfiguration is not possible, it falls back to a custom formatter that handles non-UTF-8 environments.
        - The function sets up a StreamHandler with the appropriate formatter and level.
        - The logger's propagate flag is set to False to prevent duplicate logging in parent loggers.
    """
    level = logging.INFO if verbose else logging.ERROR  # rank in world for Multi-GPU trainings

    # Configure the console (stdout) encoding to UTF-8, with checks for compatibility
    formatter = logging.Formatter("%(message)s")  # Default formatter
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                """Sets up logging with UTF-8 encoding and configurable verbosity."""
                return emojis(super().format(record))

        try:
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception as e:
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")
            formatter = CustomFormatter("%(message)s")

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False

    # set printing to the file
    if output:
        file_handler = logging.handlers.RotatingFileHandler(output, maxBytes=100000, backupCount=10, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)
for logger in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)


try:
    import tensorrt as trt  # noqa

    TENSORRT_ENABLE = True
except:
    TENSORRT_ENABLE = False
    LOGGER.info("TensorRT library is not installed. Check your installation carefully.")

try:
    import cuda  # noqa

    CUDA_PROFILE_ENABLE = True
except:
    CUDA_PROFILE_ENABLE = False
    LOGGER.info("Cuda library is not installed. Check your installation carefully.")

try:
    import onnxruntime  # noqa

    ONNX_PROFILE_ENABLE = True
except:
    ONNX_PROFILE_ENABLE = False
    LOGGER.info("ONNX library is not installed. Check your installation carefully.")

# Library
from .profile import NoProfile, Profile  # noqa

if TENSORRT_ENABLE:
    # Map TRT severity to Python logging levels
    TRT_LOGGING_MAP = {
        trt.Logger.INTERNAL_ERROR: logging.CRITICAL,
        trt.Logger.ERROR: logging.ERROR,
        trt.Logger.WARNING: logging.WARNING,
        trt.Logger.INFO: logging.INFO,
        trt.Logger.VERBOSE: logging.DEBUG,
    }

    class TensorRTLogger(trt.Logger):
        def __init__(self):
            super().__init__(trt.Logger.VERBOSE)  # set desired TRT log severity here

        def log(self, severity, msg):
            # Convert TRT severity to Python log level
            level = TRT_LOGGING_MAP.get(severity, logging.INFO)
            LOGGER.log(level, msg)

else:
    TensorRTLogger = LOGGER


__all__ = (
    "Profile",
    "NoProfile()",
    "LOGGER",
    "ROOT",
    "TIMEZONE",
    "TensorRTLogger",
    "TENSORRT_ENABLE",
    "CUDA_PROFILE_ENABLE",
    "ONNX_PROFILE_ENABLE",
)
