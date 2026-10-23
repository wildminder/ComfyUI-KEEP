"""P1.4 - the headless bootstrap helper.

The helper is proven on a **dummy package** (the real package import lands in
P3.6):

* a tiny package is registered under an alias and imported - alias resolution
  and the submodule lookup must work,
* a missing environment variable must produce an actionable error that names
  the variable (the acceptance gate runs without environment variables),
* the helper must not contain a hardcoded path (portability is a correctness
  property, not a style preference).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

# The helper is loaded by path, not as ``tests._headless``: the embedded
# interpreter has the ComfyUI runtime tree on ``sys.path`` and that tree
# carries its own ``tests`` package, which would shadow ours.
_HELPER_PATH = Path(__file__).resolve().parent / "_headless.py"
_spec = importlib.util.spec_from_file_location("keep_headless_helper", _HELPER_PATH)
assert _spec is not None and _spec.loader is not None
_headless = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_headless)
HeadlessBootstrapError = _headless.HeadlessBootstrapError


def _make_dummy_package(root: Path, name: str = "DummyNode-1") -> Path:
    """A minimal custom-node-like package with a submodule."""
    package = root / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")
    (package / "engine.py").write_text("ENGINE = 'engine'\n", encoding="utf-8")
    return package


# ---------------------------------------------------------------------------
# package alias
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_register_dummy_package_and_import_it(tmp_path):
    """A tiny package is registered under an alias and can be imported."""
    package = _make_dummy_package(tmp_path)
    alias = _headless.alias_for(package)
    assert alias == "DummyNode_1"
    assert alias.isidentifier()

    module = _headless.register_package(package, execute=True)
    try:
        assert sys.modules[alias] is module
        assert list(module.__path__) == [str(package.resolve())]
        assert list(module.__spec__.submodule_search_locations) == [
            str(package.resolve())
        ]

        imported = importlib.import_module(alias)
        assert imported is module
        assert imported.VALUE == 42

        engine = importlib.import_module(f"{alias}.engine")
        assert engine.ENGINE == "engine"
        assert Path(engine.__file__).parent == package.resolve()
    finally:
        sys.modules.pop(alias, None)
        sys.modules.pop(f"{alias}.engine", None)


@pytest.mark.unit
def test_register_package_without_execution_resolves_submodules(tmp_path):
    """Without execution the entry point is not run, but submodules resolve."""
    package = _make_dummy_package(tmp_path)
    alias = "DummyNode_1_lazy"
    try:
        module = _headless.register_package(package, alias=alias)
        assert not hasattr(module, "VALUE"), "the entry point must not have run"

        engine = importlib.import_module(f"{alias}.engine")
        assert engine.ENGINE == "engine"
    finally:
        sys.modules.pop(alias, None)
        sys.modules.pop(f"{alias}.engine", None)


@pytest.mark.unit
def test_register_package_executes_the_entry_point_on_demand(tmp_path):
    """``execute=True`` runs the real ``__init__.py``."""
    package = _make_dummy_package(tmp_path)
    alias = "DummyNode_1_executed"
    try:
        module = _headless.register_package(package, alias=alias, execute=True)
        assert module.VALUE == 42
        assert sys.modules[alias] is module
    finally:
        sys.modules.pop(alias, None)


@pytest.mark.unit
def test_register_package_rejects_a_directory_without_init(tmp_path):
    """A directory that is not a package produces a clear error."""
    empty = tmp_path / "NotAPackage"
    empty.mkdir()
    with pytest.raises(HeadlessBootstrapError) as excinfo:
        _headless.register_package(empty)
    assert "__init__.py" in str(excinfo.value)


# ---------------------------------------------------------------------------
# environment variables
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_missing_comfyui_root_env_produces_an_actionable_error(monkeypatch):
    """Without ``COMFYUI_ROOT`` the helper names the variable to set."""
    monkeypatch.delenv(_headless.COMFYUI_ROOT_ENV, raising=False)

    with pytest.raises(HeadlessBootstrapError) as excinfo:
        _headless.resolve_comfyui_root()

    message = str(excinfo.value)
    assert _headless.COMFYUI_ROOT_ENV in message
    assert "not set" in message


@pytest.mark.unit
def test_missing_model_dir_env_produces_an_actionable_error(monkeypatch):
    """Without ``KEEP_MODEL_DIR`` the helper names the variable to set."""
    monkeypatch.delenv(_headless.MODEL_DIR_ENV, raising=False)

    with pytest.raises(HeadlessBootstrapError) as excinfo:
        _headless.resolve_model_dir()

    message = str(excinfo.value)
    assert _headless.MODEL_DIR_ENV in message
    assert "not set" in message


@pytest.mark.unit
def test_blank_env_var_counts_as_missing(monkeypatch):
    """An empty variable (the CI sets ``COMFYUI_ROOT=""``) counts as missing."""
    monkeypatch.setenv(_headless.COMFYUI_ROOT_ENV, "   ")
    with pytest.raises(HeadlessBootstrapError) as excinfo:
        _headless.resolve_comfyui_root()
    assert _headless.COMFYUI_ROOT_ENV in str(excinfo.value)


@pytest.mark.unit
def test_env_var_is_used_when_it_points_at_a_comfyui_tree(monkeypatch, tmp_path):
    """A valid ``COMFYUI_ROOT`` is resolved and validated."""
    root = tmp_path / "ComfyUI"
    root.mkdir()
    (root / "folder_paths.py").write_text("models_dir = ''\n", encoding="utf-8")
    monkeypatch.setenv(_headless.COMFYUI_ROOT_ENV, str(root))

    assert _headless.resolve_comfyui_root() == root.resolve()


@pytest.mark.unit
def test_wrong_comfyui_root_is_rejected(monkeypatch, tmp_path):
    """A directory that is not a ComfyUI tree is rejected by name."""
    root = tmp_path / "NotComfyUI"
    root.mkdir()
    monkeypatch.setenv(_headless.COMFYUI_ROOT_ENV, str(root))

    with pytest.raises(HeadlessBootstrapError) as excinfo:
        _headless.resolve_comfyui_root()
    assert "folder_paths.py" in str(excinfo.value)


@pytest.mark.unit
def test_model_dir_env_var_is_used(monkeypatch, tmp_path):
    """``KEEP_MODEL_DIR`` resolves the weight directory."""
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    monkeypatch.setenv(_headless.MODEL_DIR_ENV, str(model_dir))

    assert _headless.resolve_model_dir() == model_dir.resolve()


# ---------------------------------------------------------------------------
# extra model paths
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_load_extra_model_paths_uses_the_server_mechanism(monkeypatch, tmp_path):
    """The extra paths are loaded through ``utils.extra_config`` like the server."""
    root = tmp_path / "ComfyUI"
    (root / "utils").mkdir(parents=True)
    (root / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (root / "utils" / "extra_config.py").write_text(
        "LOADED = []\n\n\ndef load_extra_path_config(yaml_path):\n    LOADED.append(yaml_path)\n",
        encoding="utf-8",
    )
    (root / "folder_paths.py").write_text("models_dir = ''\n", encoding="utf-8")
    config = root / "extra_model_paths.yaml"
    config.write_text("keep_models:\n  base_path: models\n", encoding="utf-8")

    # isolate sys.path / sys.modules from the rest of the suite
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "utils", raising=False)
    monkeypatch.delitem(sys.modules, "utils.extra_config", raising=False)

    assert _headless.load_extra_model_paths(root) is True

    extra_config = sys.modules["utils.extra_config"]
    assert extra_config.LOADED == [str(config)]
    assert str(root.resolve()) in sys.path


@pytest.mark.unit
def test_load_extra_model_paths_without_configuration_is_a_noop(monkeypatch, tmp_path):
    """A missing ``extra_model_paths.yaml`` is not an error."""
    root = tmp_path / "ComfyUI"
    root.mkdir()
    (root / "folder_paths.py").write_text("models_dir = ''\n", encoding="utf-8")

    monkeypatch.setattr(sys, "path", list(sys.path))
    assert _headless.load_extra_model_paths(root) is False


# ---------------------------------------------------------------------------
# ui stubbing
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_stub_ui_replaces_only_the_preview_helpers():
    """The preview helpers become no-ops; everything else stays untouched."""
    ui = types.ModuleType("comfy_api.latest.ui")
    ui.PreviewImage = object
    ui.PreviewAudio = object
    ui.SomethingElse = object

    stubbed = _headless.stub_ui(ui)
    assert stubbed == ["PreviewImage", "PreviewAudio"]
    assert ui.PreviewImage() is None
    assert ui.PreviewAudio("audio") is None
    assert ui.SomethingElse is object, "only the preview helpers may be stubbed"


# ---------------------------------------------------------------------------
# portability
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_helper_contains_no_hardcoded_paths():
    """The helper must not contain a machine-specific path."""
    source = Path(_headless.__file__).read_text(encoding="utf-8")

    assert "C:\\" not in source
    assert "C:/" not in source
    assert "_Dev" not in source
    assert "ComfyUI_t211c130p313" not in source
