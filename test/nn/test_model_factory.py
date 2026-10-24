"""Tests for ModelFactory: config dict -> lazily-imported model instance."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from neudc.nn import MODEL_IMPORTS, ModelFactory
from neudc.utils.extras import MissingExtraError


def test_create_rejects_non_dict_config() -> None:
    with pytest.raises(TypeError, match="Config must be a dictionary"):
        ModelFactory.create([])  # type: ignore[arg-type]


def test_create_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unsupported model type"):
        ModelFactory.create({"type": "NotAModel"})


@pytest.mark.cv
def test_known_model_types_map_to_importable_classes() -> None:
    for module_path, class_name in MODEL_IMPORTS.values():
        module = __import__(module_path, fromlist=[class_name])
        assert hasattr(module, class_name)


def test_create_names_the_model_type_and_extra_when_an_extra_is_missing(without_cv2) -> None:
    without_cv2("neudc.nn.models.det.yolo")

    with pytest.raises(MissingExtraError, match=r"Model type 'YOLOv8' needs the `cv` extra"):
        ModelFactory.create({"type": "YOLOv8", "path": "unused"})


def test_create_resolves_module_lazily_through_model_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """`create` must resolve a model class through `MODEL_IMPORTS` at call time, not a pre-imported registry.

    The actual bug this guards against is `neudc.nn` eagerly importing every
    registered model (and therefore cv2/torch) just to build the dict (see the
    subprocess test below for the real regression check).
    """
    fake_module = types.ModuleType("test_fake_model_module")

    class _FakeModel:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_module.FakeModel = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "test_fake_model_module", fake_module)
    monkeypatch.setitem(MODEL_IMPORTS, "FakeModel", ("test_fake_model_module", "FakeModel"))
    # Backend resolution isn't under test here, and the real TorchBackend needs the `cv` extra.
    fake_backend = object()
    monkeypatch.setattr("neudc.nn.get_backend", lambda _name: fake_backend)

    instance = ModelFactory.create({"type": "FakeModel", "backend": "TorchBackend", "path": "unused"})

    assert isinstance(instance, _FakeModel)
    assert instance.kwargs["path"] == "unused"
    assert instance.kwargs["backend"] is fake_backend


def test_importing_neudc_nn_does_not_pull_in_cv2() -> None:
    """Regression test for `neudc.nn.__init__` eagerly importing every model class at module scope.

    `YOLOv8`/`SAHIDetector`/etc. in turn import cv2 — that forced the `cv` extra onto
    every audio-only install, since a package's `__init__.py` always runs before any
    of its submodules (e.g. `neudc.nn.backends.llm.config`, which
    `AsrNode`/`LlmNode`/`TtsNode` depend on). `torch` itself is deliberately not
    asserted here: `neudc.nn.backends` still *optionally* imports it (guarded by
    `TORCH_AVAILABLE`) when it's actually installed, which is correct — only cv2 has
    no reason to load.

    Runs in a subprocess so `sys.modules` reflects only this one import, not whatever
    the rest of the test session already happened to load.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import neudc.nn; print('cv2' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    # LOGGER (CUDA/TensorRT/ONNX "not installed" notices from neudc.utils) also writes to
    # stdout ahead of our print(), so only the last line is the actual check's result.
    assert result.stdout.strip().splitlines()[-1] == "False"
