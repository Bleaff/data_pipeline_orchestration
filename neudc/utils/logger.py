"""neudc.utils.logger.

~~~~~~~~~~~~~~~~~~

This module provides a logger for NEUDC.

NEUDC's logger is the main entry point for logging in the NEUDC codebase.
The logger is a singleton, meaning that only a single instance of it exists.
The logger is configured to write logs to the standard error stream.

The logger is configured to emit log messages at the INFO level by default.
The user can change the log level by calling the :meth:`setLevel` method.
The user can also change the log level by setting the ``NEUDC_LOG_LEVEL``
environment variable. The possible values are:

- ``DEBUG``
- ``INFO``
- ``WARNING``
- ``ERROR``
- ``CRITICAL``

The user can also change the log format by calling the :meth:`setFormat` method.
The format string is a standard Python format string with two placeholders:
- ``%(message)s``: the log message
- ``%(filename)s:%(lineno)d``: the file and line number where the log message was emitted

The logger is thread-safe, meaning that it can be used concurrently by multiple threads.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from logging import Logger

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
LOGGING_NAME = "neudc"
VERBOSE = True
USE_NUMBA = True

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _resolve_level(verbose: bool) -> int:
    """Resolve the log level from the NEUDC_LOG_LEVEL env var, falling back to verbosity.

    The environment variable takes precedence so the hot path can be quieted in
    production without code changes. When unset, INFO is used (DEBUG only if verbose).
    """
    env_level = os.environ.get("NEUDC_LOG_LEVEL", "").upper()
    if env_level in _VALID_LEVELS:
        return getattr(logging, env_level)
    return logging.INFO if verbose else logging.ERROR


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
    level = _resolve_level(verbose)
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


# Set logger. File logging is opt-in (NEUDC_LOG_FILE=path) to keep the hot path
# free of per-message disk I/O; the level honours NEUDC_LOG_LEVEL (default INFO).
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE, output=os.environ.get("NEUDC_LOG_FILE") or None)
for _logger in ("sentry_sdk", "urllib3.connectionpool"):
    logging.getLogger(_logger).setLevel(logging.CRITICAL + 1)
