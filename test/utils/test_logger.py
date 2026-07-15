"""Tests for log-level resolution and opt-in file logging (Stage 1)."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from neudc.utils.logger import _resolve_level, set_logging


def test_env_level_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("NEUDC_LOG_LEVEL", "WARNING")
    assert _resolve_level(verbose=True) == logging.WARNING


def test_default_level_is_info_when_verbose(monkeypatch) -> None:
    monkeypatch.delenv("NEUDC_LOG_LEVEL", raising=False)
    assert _resolve_level(verbose=True) == logging.INFO


def test_invalid_env_level_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("NEUDC_LOG_LEVEL", "NOT_A_LEVEL")
    assert _resolve_level(verbose=True) == logging.INFO


def test_no_file_handler_without_output() -> None:
    logger = set_logging("neudc_test_nofile", verbose=True, output=None)
    assert not any(isinstance(h, RotatingFileHandler) for h in logger.handlers)


def test_file_handler_when_output_given(tmp_path) -> None:
    log_path = tmp_path / "out.log"
    logger = set_logging("neudc_test_withfile", verbose=True, output=str(log_path))
    assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
