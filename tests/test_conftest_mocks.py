"""P1.2 - the root conftest's ComfyUI mocks.

The plan requires two things from the harness:

* "a dummy V3 node runs end-to-end through the mock ``io``" - the mock must be
  usable the way a real V3 node uses it (schema, socket types, node output),
* "``folder_paths`` mock isolates dirs per test" - no test may see another
  test's model directories.

On top of that this file carries the *mock-fidelity tripwires* of plan
section 6: the symbols the mocks imitate must exist with the expected names.
The tripwire against the **real** ComfyUI source is environment dependent and
skips cleanly when ``COMFYUI_ROOT`` is not set (the acceptance gate runs
without environment variables).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFTEST = REPO_ROOT / "conftest.py"

#: The production modules whose ComfyUI imports the mocks have to satisfy.
PRODUCTION_FILES = (
    REPO_ROOT / "nodes.py",
    REPO_ROOT / "modules" / "utils.py",
    REPO_ROOT / "modules" / "keep_model_loader.py",
    REPO_ROOT / "modules" / "keep_processor.py",
)

#: Symbols the mocks imitate (plan section 6, mock-fidelity tripwires).
MOCKED_SYMBOLS = {
    "comfy_api.latest": ("io", "ui", "ComfyExtension", "ComfyAPI", "ComfyAPISync"),
    "comfy.model_management": (
        "load_models_gpu",
        "unload_all_models",
        "soft_empty_cache",
        "get_torch_device",
    ),
    "comfy.model_patcher": ("ModelPatcher",),
    "comfy.utils": ("tiled_scale", "ProgressBar"),
    "folder_paths": (
        "add_model_folder_path",
        "get_full_path_or_raise",
        "get_filename_list",
    ),
}


def _conftest_module():
    """The root conftest module (loaded by pytest, or by path as a fallback)."""
    module = sys.modules.get("conftest")
    if module is None:
        spec = importlib.util.spec_from_file_location("keep_root_conftest", CONFTEST)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _imported_comfy_symbols(path: Path) -> list[tuple[str, str]]:
    """``(module, attribute)`` pairs imported by ``path`` from ComfyUI.

    Only absolute imports of ComfyUI modules are returned; relative imports
    (``from .. import logger``) belong to this package, not to ComfyUI.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:  # relative import -> this package
                continue
            module = node.module or ""
            if module.split(".")[0] not in {"comfy", "comfy_api", "folder_paths"}:
                continue
            for alias in node.names:
                found.append((module, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in {"comfy", "comfy_api", "folder_paths"}:
                    found.append((alias.name, ""))
    return found


# ---------------------------------------------------------------------------
# mock shape
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_comfy_api_latest_exports_io_and_ui_as_siblings():
    """``io`` and ``ui`` are siblings - ``io.ui`` does not exist.

    Verified against the real ComfyUI v0.42.0 source
    (``docs/audit/2026-10-23-comfy-api-surface.md`` section 1): a mock that
    hides ``ui`` inside ``io`` would let the mistake "from .. import io, ui"
    slip through.
    """
    latest = sys.modules["comfy_api.latest"]

    assert hasattr(latest, "io"), "comfy_api.latest must export io"
    assert hasattr(latest, "ui"), "comfy_api.latest must export ui"
    assert latest.io is not latest.ui
    assert not hasattr(latest.io, "ui"), (
        "io.ui must not exist: the real API exports ui as a sibling module"
    )
    for name in ("Schema", "ComfyNode", "NodeOutput", "Image", "Float", "Boolean",
                 "Combo", "Int", "Custom"):
        assert hasattr(latest.io, name), f"io.{name} is missing from the mock"


@pytest.mark.unit
def test_mock_exposes_every_symbol_the_production_code_imports():
    """Every ComfyUI symbol the package imports exists in the mocks.

    The production sources are parsed (not imported - importing them would
    pull the vendored tree into the test process), and each ComfyUI import is
    resolved against the installed mocks.
    """
    for path in PRODUCTION_FILES:
        assert path.is_file(), f"production file missing: {path}"
        for module_name, attribute in _imported_comfy_symbols(path):
            module = importlib.import_module(module_name)
            if not attribute:
                continue
            assert hasattr(module, attribute), (
                f"{path.name} imports {attribute} from {module_name}, but the "
                f"mock does not provide it"
            )


@pytest.mark.unit
def test_comfy_utils_mock_matches_the_production_usage():
    """``tiled_scale`` is a passthrough and ``ProgressBar`` counts."""
    import torch

    from comfy.utils import ProgressBar, tiled_scale

    scaled = tiled_scale(
        torch.ones((1, 3, 4, 4)),
        lambda samples: samples * 2,
        tile_x=2,
        tile_y=2,
        overlap=1,
        upscale_amount=2,
    )
    assert torch.equal(scaled, torch.full((1, 3, 4, 4), 2.0))

    bar = ProgressBar(10)
    bar.update(2)
    assert bar.current == 2
    bar.update_absolute(7)
    assert bar.current == 7
    bar.close()


@pytest.mark.unit
def test_progress_recorder_is_reachable_through_an_instance_only():
    """``execution`` is an instance attribute (class access raises).

    The real API creates ``execution`` in ``__init__``; class-level access
    raises ``AttributeError`` (P4.2 audit, section 2).  ``ComfyAPISync`` is the
    synchronous form, ``ComfyAPI`` the asynchronous one.
    """
    latest = sys.modules["comfy_api.latest"]

    with pytest.raises(AttributeError):
        latest.ComfyAPISync.execution  # noqa: B018 - the access is the assertion
    with pytest.raises(AttributeError):
        latest.ComfyAPI.execution  # noqa: B018 - the access is the assertion

    sync_api = latest.ComfyAPISync()
    sync_api.execution.set_progress(1, 10)
    sync_api.execution.set_progress(5, 10)
    assert sync_api.execution.calls == [(1, 10), (5, 10)]
    assert not inspect.iscoroutinefunction(sync_api.execution.set_progress)

    async_api = latest.ComfyAPI()
    assert inspect.iscoroutinefunction(async_api.execution.set_progress)


# ---------------------------------------------------------------------------
# the dummy V3 node
# ---------------------------------------------------------------------------
@pytest.mark.mock_integration
def test_dummy_v3_node_runs_end_to_end_through_mock_io():
    """A V3 node can be written and executed against the mocked ``io``."""
    from comfy_api.latest import io

    class DummyNode(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="KEEP_Dummy",
                display_name="KEEP Dummy",
                category="ComfyUI-KEEP",
                inputs=[
                    io.Int.Input(id="count", default=3, min=1, max=10, step=1),
                    io.Combo.Input(id="mode", options=["fast", "slow"], default="fast"),
                    io.Image.Input(id="image", optional=True),
                    io.Custom("KEEP_MODEL_PACK").Input(id="keep_model", optional=True),
                ],
                outputs=[io.Image.Output()],
                description="dummy node for the mock harness",
            )

        @classmethod
        def execute(cls, count, mode, image=None, keep_model=None):
            assert isinstance(count, int)
            assert mode in {"fast", "slow"}
            return io.NodeOutput(image, count)

    schema = DummyNode.GET_SCHEMA()
    assert schema.node_id == "KEEP_Dummy"
    assert schema.display_name == "KEEP Dummy"
    assert schema.category == "ComfyUI-KEEP"
    assert [item.id for item in schema.inputs] == ["count", "mode", "image", "keep_model"]
    assert schema.inputs[0].default == 3
    assert schema.inputs[0].min == 1
    assert schema.inputs[0].max == 10
    assert schema.inputs[0].step == 1
    assert schema.inputs[1].options == ["fast", "slow"]
    assert schema.inputs[2].optional is True
    assert schema.inputs[3].optional is True

    output = DummyNode.execute(count=3, mode="fast", image="IMG", keep_model=None)
    assert isinstance(output, io.NodeOutput)
    assert output.result == ("IMG", 3)
    assert output[0] == "IMG"

    # The schema is cached per class, like the real API does.
    assert DummyNode.GET_SCHEMA() is schema


@pytest.mark.mock_integration
def test_dummy_node_can_validate_inputs_and_report_progress():
    """``validate_inputs`` and the progress recorder work together."""
    from comfy_api.latest import ComfyAPISync, io

    class ValidatingNode(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="KEEP_Validating", inputs=[], outputs=[])

        @classmethod
        def validate_inputs(cls, **kwargs):
            return True if kwargs.get("ok") else "ok must be True"

        @classmethod
        def execute(cls, ok=True):
            api = ComfyAPISync()
            api.execution.set_progress(1, 2)
            assert api.execution.calls == [(1, 2)]
            return io.NodeOutput("done")

    assert ValidatingNode.validate_inputs(ok=True) is True
    assert ValidatingNode.validate_inputs(ok=False) == "ok must be True"
    assert ValidatingNode.execute(ok=True).result == ("done",)


# ---------------------------------------------------------------------------
# folder_paths isolation
# ---------------------------------------------------------------------------
_ISOLATION_PROBE: list[str] = []


@pytest.mark.mock_integration
def test_folder_paths_mock_points_at_this_tests_own_directory(mock_folder_paths):
    """The mock's ``models_dir`` lives inside this test's ``tmp_path``."""
    models_dir = Path(mock_folder_paths.models_dir)
    assert models_dir.is_dir(), "the mock models_dir must exist"
    assert models_dir.parent.name == "tmp" or "pytest" in str(models_dir), (
        f"models_dir {models_dir} does not look like a per-test tmp directory"
    )
    _ISOLATION_PROBE.append(str(models_dir))


@pytest.mark.mock_integration
def test_folder_paths_mock_is_isolated_from_the_previous_test(mock_folder_paths):
    """A second test gets a *different* models_dir and an empty registry."""
    assert _ISOLATION_PROBE, "the isolation probe test must run first"
    assert str(mock_folder_paths.models_dir) not in _ISOLATION_PROBE
    assert mock_folder_paths.folder_names_and_paths == {}

    mock_folder_paths.add_model_folder_path("keep_models", str(mock_folder_paths.models_dir))
    assert mock_folder_paths.get_folder_paths("keep_models") == [
        str(mock_folder_paths.models_dir)
    ]


@pytest.mark.mock_integration
def test_folder_paths_mock_resolves_a_file_and_raises_for_a_missing_one(mock_folder_paths):
    """``get_full_path_or_raise`` resolves inside the mocked registry only."""
    weights = Path(mock_folder_paths.models_dir) / "keep_models"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "KEEP-b76feb75.pth").write_bytes(b"stub")

    mock_folder_paths.add_model_folder_path("keep_models", str(weights))
    resolved = mock_folder_paths.get_full_path_or_raise(
        "keep_models", "KEEP-b76feb75.pth"
    )
    assert Path(resolved) == weights / "KEEP-b76feb75.pth"
    assert mock_folder_paths.get_filename_list("keep_models") == ["KEEP-b76feb75.pth"]

    with pytest.raises(FileNotFoundError):
        mock_folder_paths.get_full_path_or_raise("keep_models", "does-not-exist.pth")


# ---------------------------------------------------------------------------
# fixtures + alias
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sample_image_fixture_is_deterministic():
    """``sample_image`` is a seeded 64x64 RGB tensor - same values every time."""
    conftest = _conftest_module()
    first = conftest._make_sample_image()
    second = conftest._make_sample_image()

    assert first.shape == (1, 64, 64, 3)
    assert first.dtype.is_floating_point
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0
    assert bool((first == second).all()), "sample_image must be deterministic"


@pytest.mark.unit
def test_sample_image_fixture_is_usable_by_a_test(sample_image):
    """The fixture itself hands out the documented tensor."""
    assert tuple(sample_image.shape) == (1, 64, 64, 3)
    assert sample_image.dtype.is_floating_point


@pytest.mark.unit
def test_package_alias_resolves_submodules_through_its_path():
    """The alias carries ``__path__`` and resolves submodules inside the repo."""
    alias = sys.modules["ComfyUI_KEEP"]
    assert list(alias.__path__) == [str(REPO_ROOT)]
    assert list(alias.__spec__.submodule_search_locations) == [str(REPO_ROOT)]

    spec = importlib.util.find_spec("ComfyUI_KEEP.modules")
    assert spec is not None
    assert list(spec.submodule_search_locations) == [str(REPO_ROOT / "modules")]


@pytest.mark.unit
def test_alias_provides_the_logger_the_modules_import():
    """``from .. import logger`` works in the register-without-execute mode."""
    alias = sys.modules["ComfyUI_KEEP"]
    assert hasattr(alias, "logger"), (
        "the alias must provide the logger the modules/* code imports"
    )
    assert alias.logger.name == "ComfyUI_KEEP"


# ---------------------------------------------------------------------------
# mock fidelity against the real ComfyUI source (environment dependent)
# ---------------------------------------------------------------------------
@pytest.mark.headless
def test_mocked_symbols_exist_in_the_real_comfyui_source():
    """The mocked names exist in the real ComfyUI source (text only, no import).

    Environment dependent: it needs ``COMFYUI_ROOT`` and therefore skips
    cleanly when the variable is absent (the acceptance gate runs without
    environment variables).
    """
    comfyui_root = os.environ.get("COMFYUI_ROOT", "").strip()
    if not comfyui_root:
        pytest.skip("COMFYUI_ROOT is not set - cannot check the real ComfyUI source")
    root = Path(comfyui_root)
    if not (root / "folder_paths.py").is_file():
        pytest.skip(f"COMFYUI_ROOT={comfyui_root!r} is not a ComfyUI source tree")

    sources = {
        "comfy_api.latest": root / "comfy_api" / "latest" / "__init__.py",
        "comfy.model_management": root / "comfy" / "model_management.py",
        "comfy.model_patcher": root / "comfy" / "model_patcher.py",
        "comfy.utils": root / "comfy" / "utils.py",
        "folder_paths": root / "folder_paths.py",
    }
    for module_name, symbols in MOCKED_SYMBOLS.items():
        source = sources[module_name]
        assert source.is_file(), f"expected source file missing: {source}"
        text = source.read_text(encoding="utf-8", errors="replace")
        if module_name == "comfy_api.latest":
            # io and ui are re-exported by the package __init__.
            latest_text = text
            assert "as io" in latest_text and "as ui" in latest_text, (
                "the real comfy_api.latest must export io and ui as siblings"
            )
            continue
        for symbol in symbols:
            assert symbol in text, (
                f"{module_name}.{symbol} does not exist in the real ComfyUI source"
            )
