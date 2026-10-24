"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def without_cv2(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Simulate an install without the `cv` extra's cv2.

    Returns ``hide(*modules)``: it hides cv2 (a ``None`` entry makes both ``import cv2`` and
    ``find_spec("cv2")`` report it missing) and drops the given already-imported modules so
    their next import re-runs against the missing cv2. Nothing changes until it is called,
    so a test can take a with-cv2 baseline first. Everything is restored after the test.
    """

    def hide(*modules: str) -> None:
        monkeypatch.setitem(sys.modules, "cv2", None)
        for module in modules:
            monkeypatch.delitem(sys.modules, module, raising=False)

    return hide
