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
import types
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


#: The probe: against the **real** ComfyUI the yaml of the user is read and
#: the registered roots (and the untouched standard folders) are reported.
_EXTRA_MODEL_PATHS_PROBE = """
import json, os, sys

comfyui_root, repo_root, yaml_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, comfyui_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "tests"))

import _headless

_headless.register_package(repo_root, alias="ComfyUI_KEEP_paths", execute=True)
import folder_paths
from ComfyUI_KEEP_paths.src import keep_paths

registry = folder_paths.folder_names_and_paths
standard = ("checkpoints", "loras", "vae", "custom_nodes")
names_before = sorted(registry)
standard_before = {name: list(folder_paths.get_folder_paths(name)) for name in standard}

keep_paths.register_model_folders()
applied = keep_paths.apply_extra_model_paths(yaml_path=yaml_path)

print(
    "KEEP_EXTRA_MODEL_PATHS:"
    + json.dumps(
        {
            "applied": list(applied),
            "names_before": names_before,
            "names_after": sorted(registry),
            "standard_before": standard_before,
            "standard_after": {
                name: list(folder_paths.get_folder_paths(name)) for name in standard
            },
            "roots": {
                name: list(folder_paths.get_folder_paths(name))
                for name in ("keep_models", "facedetection", "gmflow")
            },
        }
    )
)
"""


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
    the primary root again.  The lookup that the pack really uses is
    :func:`src.keep_paths.find_model_file` (the downloader of P5.2 calls
    it), so the test calls that.
    """
    keep_paths.register_model_folders()
    extra = _extra_model_root(mock_folder_paths)
    mock_folder_paths.add_model_folder_path("keep_models", extra)
    weight = _write_weight(os.path.join(extra, "KEEP", "KEEP-b76feb75.pth"))

    resolved = keep_paths.find_model_file(
        "keep_models", "KEEP-b76feb75.pth", subdir="KEEP"
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

    resolved = keep_paths.find_model_file(
        "keep_models", "KEEP-b76feb75.pth", subdir="KEEP"
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


# ---------------------------------------------------------------------------
# the extra model paths of the host (the yaml of the user)
# ---------------------------------------------------------------------------
def _write_yaml(path: Path, text: str) -> str:
    """A yaml file with the given content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _owner_yaml(base_path: str) -> str:
    """A yaml in the shape of a user's ``extra_model_paths.yaml``.

    The shape is the one of the server: a section with a ``base_path`` and
    the standard folders, a section without one.
    """
    return (
        "comfyui:\n"
        f"  base_path: {base_path}\n"
        "  checkpoints: models/checkpoints/\n"
        "  loras: models/loras/\n"
        "other_ui:\n"
        "  custom_nodes: custom_nodes/\n"
    )


def _install_fake_extra_config(monkeypatch, folder_paths, calls, extra_paths) -> None:
    """A stand-in for the loader of ComfyUI (``utils.extra_config``).

    The pack calls the loader of the host, so the test replaces it with a
    double that registers the roots the yaml would describe.
    """
    package = types.ModuleType("utils")
    package.__path__ = []  # type: ignore[attr-defined]
    module = types.ModuleType("utils.extra_config")

    def load_extra_path_config(yaml_path):
        calls.append(yaml_path)
        for name, path in extra_paths.items():
            folder_paths.add_model_folder_path(name, path)

    module.load_extra_path_config = load_extra_path_config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils", package)
    monkeypatch.setitem(sys.modules, "utils.extra_config", module)


@pytest.mark.unit
def test_the_layout_convention_registers_the_extra_roots(
    keep_paths, mock_folder_paths, tmp_path
):
    """A yaml with a ``base_path`` points at ``<base>/models/<pack folder>``.

    The standard layout of ComfyUI is ``<base_path>/models/<folder>``, so a
    yaml that only describes the standard folders still points at the place
    where the weights of the pack live.
    """
    base = tmp_path / "owner"
    base.mkdir()
    yaml_path = _write_yaml(
        tmp_path / "extra_model_paths.yaml", _owner_yaml(str(base))
    )

    roots = keep_paths.apply_extra_model_paths(yaml_path)

    for name in ("keep_models", "facedetection", "gmflow"):
        expected = os.path.join(str(base), "models", name)
        registered = keep_paths.get_model_folder_paths(name)
        assert expected in registered, (name, registered)
        # the primary root stays the first one
        primary = os.path.join(mock_folder_paths.models_dir, name)
        assert registered[0] == primary, (name, registered)
    assert os.path.join(str(base), "models", "keep_models") in roots, roots


@pytest.mark.unit
def test_the_explicit_keys_are_read_by_the_loader_of_the_host(
    keep_paths, mock_folder_paths, tmp_path, monkeypatch
):
    """The pack asks ComfyUI's own loader - and leaves the other folders alone.

    The loader of the host is not scoped to a folder name: it touches every
    key of every section.  The pack may only add a root to its **own** three
    folders - the standard folders of ComfyUI stay untouched (AC-3).
    """
    calls: list[str] = []
    explicit_root = str(tmp_path / "explicit-models")
    _install_fake_extra_config(
        monkeypatch,
        mock_folder_paths,
        calls,
        {
            "keep_models": explicit_root,
            "checkpoints": str(tmp_path / "checkpoints"),
        },
    )
    yaml_path = _write_yaml(
        tmp_path / "extra_model_paths.yaml", _owner_yaml(str(tmp_path))
    )

    keep_paths.apply_extra_model_paths(yaml_path)

    assert calls == [yaml_path], calls
    assert explicit_root in keep_paths.get_model_folder_paths("keep_models")
    # the standard folders of the host are restored byte for byte
    assert "checkpoints" not in mock_folder_paths.folder_names_and_paths


@pytest.mark.unit
def test_the_loader_of_the_host_does_not_repeat_its_own_block(
    keep_paths, mock_folder_paths, tmp_path, monkeypatch, caplog
):
    """The host logged the block of the yaml at its startup.

    The pack asks the host to read the file again (it needs the explicit
    keys), but it must not print the block of the host a second time.
    """
    import logging

    calls: list[str] = []
    package = types.ModuleType("utils")
    package.__path__ = []  # type: ignore[attr-defined]
    module = types.ModuleType("utils.extra_config")

    def load_extra_path_config(yaml_path):
        calls.append(yaml_path)
        logging.info(f"Adding extra search path keep_models {yaml_path}")

    module.load_extra_path_config = load_extra_path_config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils", package)
    monkeypatch.setitem(sys.modules, "utils.extra_config", module)
    yaml_path = _write_yaml(
        tmp_path / "extra_model_paths.yaml", _owner_yaml(str(tmp_path))
    )

    with caplog.at_level(logging.INFO):
        keep_paths.apply_extra_model_paths(yaml_path)

    assert calls == [yaml_path], calls
    assert not [
        record
        for record in caplog.records
        if "Adding extra search path" in record.getMessage()
    ], [record.getMessage() for record in caplog.records]


@pytest.mark.unit
def test_the_yaml_of_the_command_line_is_read_too(
    keep_paths, mock_folder_paths, tmp_path, monkeypatch
):
    """The yaml of ``--extra-model-paths-config`` is read like the default one.

    The host reads the files of the command line the same way (``main.py``:
    the list of lists is flattened before the files are loaded), so the pack
    follows: a user who has **no** default yaml still gets the extra roots
    of the pack.
    """
    base = tmp_path / "cli-owner"
    base.mkdir()
    yaml_path = _write_yaml(tmp_path / "cli_models.yaml", _owner_yaml(str(base)))

    # the stand-in for the command line of ComfyUI
    package = types.ModuleType("comfy")
    package.__path__ = []  # type: ignore[attr-defined]
    cli_args = types.ModuleType("comfy.cli_args")
    cli_args.args = types.SimpleNamespace(  # type: ignore[attr-defined]
        extra_model_paths_config=[[yaml_path]]
    )
    package.cli_args = cli_args  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", package)
    monkeypatch.setitem(sys.modules, "comfy.cli_args", cli_args)

    keep_paths.apply_extra_model_paths()

    for name in ("keep_models", "facedetection", "gmflow"):
        expected = os.path.join(str(base), "models", name)
        assert expected in keep_paths.get_model_folder_paths(name), name


@pytest.mark.unit
def test_the_yaml_is_applied_once_per_change(
    keep_paths, mock_folder_paths, tmp_path, monkeypatch
):
    """The yaml is read **once per change** - ``force`` re-applies it."""
    calls: list[str] = []
    _install_fake_extra_config(
        monkeypatch,
        mock_folder_paths,
        calls,
        {"keep_models": str(tmp_path / "explicit-models")},
    )
    yaml_path = _write_yaml(
        tmp_path / "extra_model_paths.yaml", _owner_yaml(str(tmp_path))
    )

    keep_paths.apply_extra_model_paths(yaml_path)
    keep_paths.apply_extra_model_paths(yaml_path)
    assert calls == [yaml_path], calls

    keep_paths.apply_extra_model_paths(yaml_path, force=True)
    assert calls == [yaml_path, yaml_path], calls
    # the roots are not registered twice
    registered = keep_paths.get_model_folder_paths("keep_models")
    assert len(registered) == len(set(registered)), registered


@pytest.mark.unit
def test_a_missing_yaml_is_not_an_error(keep_paths, tmp_path):
    """No yaml: the pack works with its primary root."""
    roots = keep_paths.apply_extra_model_paths(
        str(tmp_path / "does-not-exist.yaml")
    )
    assert roots == []


@pytest.mark.unit
def test_a_broken_yaml_is_not_an_error(keep_paths, tmp_path):
    """A yaml that cannot be parsed: the pack works with its primary root."""
    yaml_path = _write_yaml(
        tmp_path / "extra_model_paths.yaml", "comfyui: [unclosed\n"
    )
    roots = keep_paths.apply_extra_model_paths(yaml_path)
    assert roots == []


# ---------------------------------------------------------------------------
# the download target: the root that holds the pack
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_download_root_is_the_root_that_holds_the_pack(
    keep_paths, mock_folder_paths
):
    """The root with the most of the pack's weights is the download target."""
    keep_paths.register_model_folders()
    extra = _extra_model_root(mock_folder_paths)
    mock_folder_paths.add_model_folder_path("keep_models", extra)
    _write_weight(os.path.join(extra, "KEEP", "KEEP-b76feb75.pth"))

    siblings = ["KEEP/KEEP-b76feb75.pth", "KEEP/KEEP_Asian-4765ebe0.pth"]
    assert keep_paths.resolve_download_root("keep_models", siblings) == extra


@pytest.mark.unit
def test_the_download_root_is_the_primary_root_on_a_tie(
    keep_paths, mock_folder_paths
):
    """A tie goes to the earliest root - the primary one."""
    keep_paths.register_model_folders()
    extra = _extra_model_root(mock_folder_paths)
    mock_folder_paths.add_model_folder_path("keep_models", extra)
    siblings = ["KEEP/KEEP-b76feb75.pth"]
    _write_weight(os.path.join(extra, "KEEP", "KEEP-b76feb75.pth"))
    _write_weight(
        os.path.join(
            mock_folder_paths.models_dir, "keep_models", "KEEP", "KEEP-b76feb75.pth"
        )
    )

    assert keep_paths.resolve_download_root("keep_models", siblings) == os.path.join(
        mock_folder_paths.models_dir, "keep_models"
    )


@pytest.mark.unit
def test_the_download_root_is_the_primary_root_without_siblings(
    keep_paths, mock_folder_paths
):
    """Without siblings the primary root is the target (the old behaviour)."""
    keep_paths.register_model_folders()
    primary = os.path.join(mock_folder_paths.models_dir, "keep_models")
    assert keep_paths.resolve_download_root("keep_models") == primary


@pytest.mark.unit
def test_the_download_target_follows_the_pack_home(keep_paths, mock_folder_paths):
    """``resolve_model_target`` puts a missing weight next to the pack."""
    keep_paths.register_model_folders()
    extra = _extra_model_root(mock_folder_paths)
    mock_folder_paths.add_model_folder_path("keep_models", extra)
    _write_weight(os.path.join(extra, "KEEP", "KEEP-b76feb75.pth"))

    target = keep_paths.resolve_model_target(
        "keep_models",
        "KEEP_Asian-4765ebe0.pth",
        subdir="KEEP",
        sibling_paths=["KEEP/KEEP-b76feb75.pth"],
    )
    assert target == os.path.join(extra, "KEEP", "KEEP_Asian-4765ebe0.pth"), target


@pytest.mark.headless
def test_the_real_extra_model_paths_are_discovered(tmp_path):
    """Against the real ComfyUI: the yaml of the user is discovered.

    The yaml is a **temporary** file in the shape of a user's
    ``extra_model_paths.yaml``: the tree of the owner is never read (and
    never written).  Both discovery paths are checked - the explicit key and
    the layout convention - and the standard folders of the host stay
    untouched.
    """
    value = (os.environ.get(COMFYUI_ROOT_ENV) or "").strip()
    if not value:
        pytest.skip(
            f"{COMFYUI_ROOT_ENV} is not set - the headless extra model paths "
            "test needs the ComfyUI source tree"
        )
    root = Path(value).expanduser()
    if not (root / "folder_paths.py").is_file():
        pytest.skip(
            f"{COMFYUI_ROOT_ENV}={value!r} is not a ComfyUI source tree "
            "(folder_paths.py is missing)"
        )

    base = tmp_path / "owner"
    (base / "models").mkdir(parents=True)
    explicit = tmp_path / "explicit-models"
    yaml_path = tmp_path / "extra_model_paths.yaml"
    yaml_path.write_text(
        "comfyui:\n"
        f"  base_path: {base}\n"
        "  checkpoints: models/checkpoints/\n"
        "  loras: models/loras/\n"
        "pack:\n"
        f"  keep_models: {explicit}\n"
        f"  some_new_folder: {tmp_path / 'some-new-folder'}\n"
        "other_ui:\n"
        "  custom_nodes: custom_nodes/\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _EXTRA_MODEL_PATHS_PROBE,
            str(root.resolve()),
            str(REPO_ROOT),
            str(yaml_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"the probe did not run\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    sentinel = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("KEEP_EXTRA_MODEL_PATHS:")
    ]
    assert sentinel, (
        f"the probe produced no report\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    report = json.loads(sentinel[-1].split(":", 1)[1])

    roots = report["roots"]["keep_models"]
    # the explicit key of the yaml
    assert str(explicit) in roots, roots
    # the layout convention of the section with the base_path
    assert os.path.join(str(base), "models", "keep_models") in roots, roots
    # the primary root stays the first one
    primary = os.path.join(str(root.resolve()), "models", "keep_models")
    assert os.path.normcase(roots[0]) == os.path.normcase(primary), roots
    # the standard folders of the host are untouched (AC-3)
    assert report["standard_after"] == report["standard_before"], report
    # the loader of the host may create a folder name - the pack removes it
    # again (it may only add roots to its own three folders)
    assert "some_new_folder" not in report["names_after"], report
    assert set(report["names_after"]) == set(report["names_before"]), report
