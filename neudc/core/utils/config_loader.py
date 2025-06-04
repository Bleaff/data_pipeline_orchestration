"""core.utils.config_loader.

This module contains a single function to load a YAML configuration file for the (NUDC) pipeline.

"""

from __future__ import annotations

from pathlib import Path

from yaml import safe_load  # type: ignore[import-untyped]


def load_config(path: str | Path) -> dict:
    """Load YAML pipeline configuration.

    Args:
    ----
        path (str | Path): Path to YAML config.

    Returns:
    -------
        dict: Parsed configuration dictionary.

    """
    path = Path(path)
    if not path.exists():
        msg = f"YAML config not found: {path}"
        raise FileNotFoundError(msg)

    with path.open() as f:
        return safe_load(f)
