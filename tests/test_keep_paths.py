"""P3.5 - the model folders and the path resolution.

The pack registers three model folders with ComfyUI's ``folder_paths`` and
resolves every weight path through them:

* ``keep_models`` (with the ``KEEP`` **sub-directory**),
* ``facedetection``,
* ``gmflow``.

The on-disk layout must not change: the KEEP checkpoints stay in
``models/keep_models/KEEP`` and the facelib weights in ``models/facedetection``.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

COMFYUI_ROOT_ENV = "COMFYUI_ROOT"

#: The probe: the pack is imported against the **real** ComfyUI source tree
#: and the registered folders are reported.
_REAL_FOLDER_PATHS_PROBE = """
import json, os, sys

comfyui_root, repo_root = sys.argv[1], sys.argv[2]
sys.path.insert(0, comfyui_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "tests"))

import _headless

module = _headless.register_package(
    repo_root, alias="ComfyUI_KEEP_paths", execute=True
)
import folder_paths

print(
    "KEEP_FOLDER_PATHS:"
    + json.dumps(
        {
            name: folder_paths.get_folder_paths(name)
            for name in ("keep_models", "facedetection", "gmflow")
        }
    )
)
"""


@pytest.fixture(scope="module")
def keep_paths():
    return importlib.import_module("ComfyUI_KEEP.src.keep_paths")


@pytest.mark.unit
def test_the_three_model_folders_are_registered(keep_paths, mock_folder_paths):
    """The registration adds the three folders with the expected paths."""
    assert keep_paths.register_model_folders() is True

    registered = mock_folder_paths.folder_names_and_paths
    for name in ("keep_models", "facedetection", "gmflow"):
        assert name in registered, sorted(registered)
        expected = os.path.join(mock_folder_paths.models_dir, name)
        assert registered[name][0][0] == expected, (name, registered[name])


@pytest.mark.unit
def test_the_registration_is_idempotent(keep_paths, mock_folder_paths):
    """A second registration does not add the folder twice."""
    keep_paths.register_model_folders()
    keep_paths.register_model_folders()
    paths = mock_folder_paths.folder_names_and_paths["keep_models"]
    assert len(paths) == 1, paths


@pytest.mark.unit
def test_the_keep_subdir_is_preserved(keep_paths, mock_folder_paths):
    """The KEEP checkpoints resolve to ``models/keep_models/KEEP/<file>``."""
    keep_paths.register_model_folders()
    target = keep_paths.resolve_model_target(
        "keep_models", "KEEP-b76feb75.pth", subdir="KEEP"
    )
    expected = os.path.join(
        mock_folder_paths.models_dir, "keep_models", "KEEP", "KEEP-b76feb75.pth"
    )
    assert target == expected, target
    assert os.path.isdir(os.path.dirname(target))


@pytest.mark.unit
def test_an_existing_weight_is_resolved(keep_paths, mock_folder_paths):
    """An existing file is resolved through the registered folders."""
    keep_paths.register_model_folders()
    folder = os.path.join(mock_folder_paths.models_dir, "keep_models", "KEEP")
    os.makedirs(folder, exist_ok=True)
    weight = os.path.join(folder, "KEEP-b76feb75.pth")
    with open(weight, "wb") as handle:
        handle.write(b"not a real checkpoint")

    resolved = keep_paths.resolve_model_file(
        "keep_models", "KEEP-b76feb75.pth", subdir="KEEP"
    )
    assert resolved == weight, resolved


@pytest.mark.unit
def test_a_missing_weight_names_the_file(keep_paths, mock_folder_paths):
    """A missing weight raises an error that names the file."""
    keep_paths.register_model_folders()
    with pytest.raises(FileNotFoundError) as excinfo:
        keep_paths.resolve_model_file(
            "keep_models", "does-not-exist.pth", subdir="KEEP"
        )
    assert "does-not-exist.pth" in str(excinfo.value)


@pytest.mark.unit
def test_the_facedetection_folder_has_no_subdir(keep_paths, mock_folder_paths):
    """The facelib weights stay directly in ``models/facedetection``."""
    keep_paths.register_model_folders()
    target = keep_paths.resolve_model_target(
        "facedetection", "parsing_parsenet.pth"
    )
    expected = os.path.join(
        mock_folder_paths.models_dir, "facedetection", "parsing_parsenet.pth"
    )
    assert target == expected, target


@pytest.mark.unit
def test_no_engine_concatenates_the_models_dir():
    """The grep guard: only ``src/keep_paths.py`` knows ``models_dir``.

    The registration needs the primary root, but no engine may build a
    weight path from ``folder_paths.models_dir`` itself.
    """
    offenders: list[str] = []
    for path in sorted(SRC_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "models_dir"
                and path.name != "keep_paths.py"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], (
        f"only src/keep_paths.py may use folder_paths.models_dir: {offenders}"
    )


@pytest.mark.unit
def test_the_loader_resolves_the_facelib_root_through_keep_paths():
    """The loader does not build the facelib path itself any more."""
    path = SRC_DIR / "keep_model_loader.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "models_dir"
    ]
    assert offenders == [], (
        f"the loader must not use folder_paths.models_dir (lines {offenders})"
    )
    assert "keep_paths.get_primary_folder" in source


@pytest.mark.headless
def test_the_real_folder_paths_know_the_keep_folders():
    """Against the real ComfyUI: the folders are registered and resolvable."""
    value = (os.environ.get(COMFYUI_ROOT_ENV) or "").strip()
    if not value:
        pytest.skip(
            f"{COMFYUI_ROOT_ENV} is not set - the headless folder_paths test "
            "needs the ComfyUI source tree"
        )
    root = Path(value).expanduser()
    if not (root / "folder_paths.py").is_file():
        pytest.skip(
            f"{COMFYUI_ROOT_ENV}={value!r} is not a ComfyUI source tree "
            "(folder_paths.py is missing)"
        )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _REAL_FOLDER_PATHS_PROBE,
            str(root.resolve()),
            str(REPO_ROOT),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"the package did not import\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    sentinel = [
        line for line in result.stdout.splitlines() if line.startswith("KEEP_FOLDER_PATHS:")
    ]
    assert sentinel, (
        f"the probe produced no report\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    report = json.loads(sentinel[-1].split(":", 1)[1])
    for name in ("keep_models", "facedetection", "gmflow"):
        assert report[name], f"{name} is not registered: {report}"
        assert report[name][0].endswith(name), report
