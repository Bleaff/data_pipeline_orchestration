"""Tests for `neudc.utils.extras`: telling a missing optional extra apart from a genuine import bug."""

from __future__ import annotations

import pytest

from neudc.utils.extras import MissingExtraError, extra_providing, import_module_for, require_extra


@pytest.mark.parametrize(
    ("module_name", "expected"),
    [("cv2", "cv"), ("sklearn.cluster", "cv"), ("PIL", "cv"), ("neudc.nn.models", None), ("", None), (None, None)],
)
def test_extra_providing(module_name: str | None, expected: str | None) -> None:
    assert extra_providing(module_name) == expected


def test_require_extra_passes_when_installed() -> None:
    require_extra("cv", "json")


def test_require_extra_raises_missing_extra_naming_the_module(without_cv2) -> None:
    without_cv2()

    with pytest.raises(MissingExtraError, match=r"No module named 'cv2' \(part of the `cv` extra\)") as exc:
        require_extra("cv", "cv2")

    assert exc.value.name == "cv2"
    assert isinstance(exc.value, ModuleNotFoundError)


def test_import_module_for_wraps_a_missing_extra(without_cv2) -> None:
    without_cv2("neudc.core.node.readers.image_reader")

    with pytest.raises(MissingExtraError, match=r"^Thing 'x' needs the `cv` extra .*pip install 'neudc\[cv\]'") as exc:
        import_module_for("neudc.core.node.readers.image_reader", "Thing 'x'")

    assert exc.value.name == "cv2"


def test_import_module_for_lets_a_non_extra_failure_through_unchanged() -> None:
    with pytest.raises(ModuleNotFoundError) as exc:
        import_module_for("neudc.no_such_module", "Thing 'x'")

    assert not isinstance(exc.value, MissingExtraError)
    assert exc.value.name == "neudc.no_such_module"
