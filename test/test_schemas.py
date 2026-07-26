"""Tests for DataConfig.validate_paths (neudc/schemas.py).

Regression test for a latent bug: the validator checked `os.path.exists(v)` (the
whole input list) instead of `os.path.exists(path)` (each entry), which raises
TypeError on any non-empty `paths` list instead of filtering missing ones.
"""

from __future__ import annotations

from neudc.schemas import DataConfig


def test_validate_paths_keeps_existing_and_drops_missing(tmp_path) -> None:
    existing = tmp_path / "frames"
    existing.mkdir()
    missing = tmp_path / "does_not_exist"

    config = DataConfig(paths=[str(existing), str(missing)])

    assert config.paths == [str(existing)]


def test_validate_paths_with_all_existing_paths(tmp_path) -> None:
    existing = tmp_path / "frames"
    existing.mkdir()

    config = DataConfig(paths=[str(existing)])

    assert config.paths == [str(existing)]
