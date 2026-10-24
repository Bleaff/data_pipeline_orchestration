"""Optional-dependency (pyproject extras) helpers: tell a missing extra apart from a genuine import bug."""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ("EXTRA_MODULES", "MissingExtraError", "extra_providing", "import_module_for", "require_extra")

# extra -> top-level import names it provides (keep in sync with [tool.poetry.extras] in pyproject.toml).
# PIL comes in transitively (imagehash/ultralytics/torchvision) but is imported directly by CV code.
EXTRA_MODULES: dict[str, frozenset[str]] = {
    "cv": frozenset({"cv2", "torch", "torchvision", "sklearn", "skimage", "imagehash", "ultralytics", "PIL"}),
}


class MissingExtraError(ModuleNotFoundError):
    """A package belonging to an optional extra is not installed.

    Subclasses ``ModuleNotFoundError`` (with ``name`` set to the missing module) so code
    that only cares about "this can't be imported" keeps working, while callers that
    want to degrade gracefully can catch exactly this and let genuine import bugs surface.
    """


def extra_providing(module_name: str | None) -> str | None:
    """Return the extra that provides ``module_name`` (any dotted submodule of it), or ``None``."""
    if not module_name:
        return None
    top_level = module_name.partition(".")[0]
    return next((extra for extra, modules in EXTRA_MODULES.items() if top_level in modules), None)


def require_extra(extra: str, *modules: str) -> None:
    """Raise :class:`MissingExtraError` if any of ``modules`` (from ``extra``) is not installed.

    A presence check (``find_spec``), not an import: it neither pays the import cost
    (torch, cv2) in the calling process nor masks a broken install, which still fails
    loudly on the real import later.
    """
    for module in modules:
        if importlib.util.find_spec(module) is None:
            msg = f"No module named '{module}' (part of the `{extra}` extra)"
            raise MissingExtraError(msg, name=module)


def import_module_for(module_path: str, what: str) -> ModuleType:
    """Import ``module_path``, re-raising a missing optional extra as :class:`MissingExtraError` naming ``what``.

    Any other import failure (an internal typo, a circular import, a broken
    third-party install) propagates unchanged rather than being blamed on a missing extra.

    Args:
    ----
        module_path: Dotted module to import.
        what: Human-readable owner for the error message, e.g. ``"Node type 'ResizeNode'"``.

    Raises:
    ------
        MissingExtraError: The import failed on a module provided by an optional extra.

    """
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        extra = extra_providing(e.name)
        if extra is None:
            raise
        msg = f"{what} needs the `{extra}` extra ({e}). Install it with: pip install 'neudc[{extra}]'"
        raise MissingExtraError(msg, name=e.name) from e
