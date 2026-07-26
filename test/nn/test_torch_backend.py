"""Tests for TorchBackend input staging (Stage 6).

The pinned-buffer path only activates on CUDA, so here we verify the CPU passthrough
and the pure buffer-reuse decision (which does not allocate pinned memory), plus a
real forward pass through a tiny scripted model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from neudc.nn.backends.torch import TorchBackend


class _Scale(nn.Module):
    """Identity-by-value module with a parameter (so fp16 detection works)."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w


@pytest.fixture
def backend(tmp_path):
    path = tmp_path / "scale.torchscript"
    torch.jit.save(torch.jit.script(_Scale()), str(path))
    return TorchBackend(str(path), device_id=-1)


def test_call_cpu_returns_numpy(backend) -> None:
    x = np.ones((1, 3, 8, 8), dtype=np.float32)
    out = backend(x)

    assert isinstance(out, list)
    assert isinstance(out[0], np.ndarray)
    assert np.allclose(out[0], x)


def test_stage_input_cpu_passthrough(backend) -> None:
    x = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    staged = backend._stage_input(x)

    assert staged.device.type == "cpu"
    assert torch.allclose(staged, torch.from_numpy(x))


def test_needs_new_staging_decision(backend) -> None:
    assert backend._pinned_staging is None
    assert backend._needs_new_staging(torch.empty((1, 3, 4, 4))) is True

    # A cached CPU tensor is enough to exercise the shape/dtype comparison.
    backend._pinned_staging = torch.empty((1, 3, 4, 4))
    assert backend._needs_new_staging(torch.empty((1, 3, 4, 4))) is False
    assert backend._needs_new_staging(torch.empty((2, 3, 4, 4))) is True
    assert backend._needs_new_staging(torch.empty((1, 3, 4, 4), dtype=torch.float64)) is True
