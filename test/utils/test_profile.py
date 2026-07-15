"""Tests for the Profile decorator/context manager (Stage 1).

Verifies that CUDA-sync profiling degrades to a no-op on CPU-only machines
(instead of raising / spamming errors) while still measuring wall time.
"""

from __future__ import annotations

from neudc.utils.profile import NoProfile, Profile


def test_profile_on_cpu_does_not_raise_and_measures() -> None:
    profile = Profile(use_cuda=True, use_torch=True, name="unit")
    with profile:
        _ = sum(range(10_000))

    assert profile.call_count == 1
    assert profile.t >= 0.0


def test_profile_decorator_counts_calls() -> None:
    profile = Profile(use_cuda=True, use_torch=True, name="decorated")

    @profile
    def work() -> int:
        return sum(range(1000))

    for _ in range(3):
        work()

    assert profile.call_count == 3


def test_noprofile_disables_and_restores() -> None:
    assert Profile.enabled is True

    @NoProfile
    def guarded() -> bool:
        return Profile.enabled

    assert guarded() is False
    assert Profile.enabled is True
