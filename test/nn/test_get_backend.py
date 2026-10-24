"""Tests for `neudc.nn.backends.get_backend` and its optional-torch gating."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from neudc.nn import backends
from neudc.nn.backends import BackendType, get_backend


def test_torch_backend_is_rejected_with_an_install_hint_when_torch_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backends, "TORCH_AVAILABLE", False)

    with pytest.raises(RuntimeError, match=r"Install the `cv` extra"):
        get_backend(BackendType.TORCH)


def test_backends_import_without_torch_installed() -> None:
    """With torch absent (not merely failing to import), the package imports and gates TorchBackend off.

    Runs in a subprocess: torch must be hidden before `neudc.utils` is first imported.
    A `None` entry in sys.modules makes `find_spec` report the package as not installed.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "sys.modules['torch'] = None\n"
                "from neudc.nn import backends\n"
                "from neudc.utils import profile\n"
                "assert not backends.TORCH_AVAILABLE\n"
                "assert not profile.TORCH_PROFILE_ENABLE\n"
                "assert 'TorchBackend' not in backends.__all__\n"
                "try:\n"
                "    backends.get_backend(backends.BackendType.TORCH)\n"
                "except RuntimeError:\n"
                "    print('ok')\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip().splitlines()[-1] == "ok"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch installed (the `cv` extra)")
def test_broken_torch_backend_import_is_not_masked_as_torch_missing() -> None:
    """With torch installed, a failure importing `.torch` must surface, not silently disable TorchBackend.

    Otherwise the real error is lost and `get_backend` later tells the user to install
    the `cv` extra they already have.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import sys\nsys.modules['neudc.nn.backends.torch'] = None\nimport neudc.nn.backends\n"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "import of neudc.nn.backends.torch halted" in result.stderr
