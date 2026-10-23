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


#: A URL that cannot be downloaded - a test that expects **no** download
#: fails fast (and loudly) when the lookup does not find the weight.
UNREACHABLE_URL = "http://127.0.0.1:1/never-downloaded.pth"


def _extra_model_root(mock_folder_paths) -> str:
    """A second model root next to the (temporary) primary one."""
    root = str(Path(mock_folder_paths.models_dir).parent / "extra-models")
    os.makedirs(root, exist_ok=True)
    return root


def _write_weight(path: str, content: bytes = b"not a real checkpoint") -> str:
    """A weight file at ``path`` (the parent directories are created)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)
    return path


@pytest.mark.unit
def test_the_loader_uses_a_weight_from_an_extra_model_root(keep_paths, mock_folder_paths):
    """The **production** path finds a weight in an extra model root.

    The plan (P3.5) wants the multi-root handling in the load path, not
    only in a helper the product never calls: a weight that lives in an
    extra model root is used where it is instead of being downloaded into
    the primary root again.
    """
    keep_paths.register_model_folders()
    extra = _extra_model_root(mock_folder_paths)
    mock_folder_paths.add_model_folder_path("keep_models", extra)
    weight = _write_weight(os.path.join(extra, "KEEP", "KEEP-b76feb75.pth"))

    resolved = keep_paths.load_file_from_url_comfy(
        url=UNREACHABLE_URL,
        folder_name="keep_models",
        file_name="KEEP-b76feb75.pth",
        subdir="KEEP",
    )
    assert resolved == weight, resolved
    # nothing was downloaded into the primary root
    primary = os.path.join(
        mock_folder_paths.models_dir, "keep_models", "KEEP", "KEEP-b76feb75.pth"
    )
    assert not os.path.exists(primary), primary


@pytest.mark.unit
def test_the_primary_root_wins_over_an_extra_model_root(keep_paths, mock_folder_paths):
    """The primary root is searched first."""
    keep_paths.register_model_folders()
    extra = _extra_model_root(mock_folder_paths)
    mock_folder_paths.add_model_folder_path("keep_models", extra)
    primary_weight = _write_weight(
        os.path.join(
            mock_folder_paths.models_dir, "keep_models", "KEEP", "KEEP-b76feb75.pth"
        )
    )
    _write_weight(os.path.join(extra, "KEEP", "KEEP-b76feb75.pth"))

    resolved = keep_paths.load_file_from_url_comfy(
        url=UNREACHABLE_URL,
        folder_name="keep_models",
        file_name="KEEP-b76feb75.pth",
        subdir="KEEP",
    )
    assert resolved == primary_weight, resolved


@pytest.mark.unit
def test_the_model_root_follows_the_weights(keep_paths, mock_folder_paths):
    """The root of a folder is the one the weights really live in.

    The facelib helper builds the weight paths from a **single** root, so
    the root has to be the one that holds the weights - an extra model root
    included.
    """
    keep_paths.register_model_folders()
    names = ["parsing_parsenet.pth", "detection_Resnet50_Final.pth"]
    extra = _extra_model_root(mock_folder_paths)
    mock_folder_paths.add_model_folder_path("facedetection", extra)
    for name in names:
        _write_weight(os.path.join(extra, name))

    assert keep_paths.resolve_model_root("facedetection", names) == extra
    # a weight that is nowhere is downloaded to the primary root
    assert keep_paths.resolve_model_root("facedetection", ["missing.pth"]) == os.path.join(
        mock_folder_paths.models_dir, "facedetection"
    )


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
    """The grep guard: only the registration knows ``models_dir``.

    The registration needs the primary root, but no engine may build a
    weight path from ``folder_paths.models_dir`` itself.  Even inside
    ``src/keep_paths.py`` the attribute may only be used by the
    registration - never to build the path of a weight file.
    """
    offenders: list[str] = []
    for path in sorted(SRC_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        registration = _function_line_range(tree, "register_model_folders")
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and node.attr == "models_dir"):
                continue
            if path.name != "keep_paths.py":
                offenders.append(f"{path.name}:{node.lineno}")
                continue
            if registration is None or not (
                registration[0] <= node.lineno <= registration[1]
            ):
                offenders.append(
                    f"{path.name}:{node.lineno} (outside the registration)"
                )
    assert offenders == [], (
        f"only the registration in src/keep_paths.py may use "
        f"folder_paths.models_dir: {offenders}"
    )


def _function_line_range(tree: ast.Module, name: str) -> tuple[int, int] | None:
    """The line range of a function of the module (``None`` when absent)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node.lineno, node.end_lineno or node.lineno
    return None


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
    # the root comes from keep_paths - and it is the root the weights are in
    assert "keep_paths.resolve_model_root" in source


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


@pytest.mark.unit
def test_the_primary_root_is_registered_first(keep_paths, mock_folder_paths):
    """The primary-root-first rule (P3.5 / P5.2).

    An extra model root that was registered **before** the pack must not
    shadow the primary root: the lookup (``get_full_path_or_raise``) and the
    download target both follow the primary root first.
    """
    extra = _extra_model_root(mock_folder_paths)
    mock_folder_paths.add_model_folder_path("keep_models", extra)

    keep_paths.register_model_folders()

    roots = keep_paths.get_model_folder_paths("keep_models")
    primary = os.path.join(mock_folder_paths.models_dir, "keep_models")
    assert roots[0] == primary, roots
    assert keep_paths.get_primary_folder("keep_models") == primary
    assert keep_paths.resolve_model_target("keep_models", "x.pth").startswith(primary)
