"""The regression suite of the five defects - the tripwires of the pack.

The five defects of the owner report (2026-10-25) are guarded **twice**:

* by the **test of the batch** that fixed the defect (the behaviour), and
* by the **guards of this file** (the shape of the code) - a repo-wide grep
  over the pack's own sources.

The guards are the second line of defence.  They catch the defect coming
back in a **different** place: a new call site with a ``"KEEP: "`` literal, a
new ``print()``, a new ``tqdm`` import, a new loop over the whole weight
manifest, a second module that touches the model registry of the host.

Every guard names the defect it protects:

======== ==========================================================
D1       the model roots and the download target
D2 / D3  ComfyUI's logger and the single ``[KEEP]`` tag
D4       every console write reaches the logger
D5       only the selected face detector is downloaded
======== ==========================================================

The file also carries the two **behavioural** regressions that close the
loops of the batches (the yaml of the owner is honoured end to end; the log
line of the pack carries the tag) and the invariants of the repository (the
vendored tree is read-only; the runtime dependencies stay the slimmed
host-guarantee set - both files declare ``opencv-python-headless`` only).

The **tripwires of the GPU redesign** live here as well (the guards G1-G6 of
the section "The GPU video pipeline redesign"): the batched conversions, the
read-once contract, the single affine, the chunk contract of the KEEP
inference, the reachable vendored paste fallback and the progress contract of
the sequence path.

The guards scan the **sources of the pack** (``src/``, ``nodes/`` and the
package entry point).  The vendored tree (``modules/deps``) is upstream code
and is never touched - it is guarded by
:func:`test_the_vendored_tree_is_untouched` instead.
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

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
NODES_DIR = REPO_ROOT / "nodes"
INIT_FILE = REPO_ROOT / "__init__.py"

#: The sources of the pack (the vendored tree is **not** part of it).
PACK_SOURCES = (
    tuple(sorted(SRC_DIR.glob("*.py")))
    + tuple(sorted(NODES_DIR.glob("*.py")))
    + (INIT_FILE,)
)

#: The module that owns the model roots of the pack.
PATHS_MODULE = "keep_paths.py"

#: The module that owns the logger of the pack.
LOGGER_MODULE = "keep_logger.py"

#: The three model folders of the pack (the only folders it may touch).
MODEL_FOLDERS = ("keep_models", "facedetection", "gmflow")

#: The file names of the facelib weights.  The manifest names them and the
#: vendored helper maps a detector to its file - the loader and the
#: downloader must not spell them out.
FACELIB_WEIGHT_NAMES = (
    "detection_Resnet50_Final.pth",
    "detection_mobilenet0.25_Final.pth",
    "yolov5n-face.pth",
    "yolov5l-face.pth",
    "parsing_parsenet.pth",
)

#: The detectors of the widget (the order is a compatibility guarantee).
DETECTOR_NAMES = (
    "retinaface_resnet50",
    "retinaface_mobile0.25",
    "YOLOv5l",
    "YOLOv5n",
)

#: The environment variable that points at the ComfyUI source tree (the
#: headless tier).
COMFYUI_ROOT_ENV = "COMFYUI_ROOT"


# ---------------------------------------------------------------------------
# the helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def keep_paths_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_paths")


@pytest.fixture(scope="module")
def download_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_download")


@pytest.fixture(scope="module")
def logger_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_logger")


def _tree(path: Path) -> ast.Module:
    """The parsed source of a module."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _code_strings(path: Path) -> list[str]:
    """The string literals of a module - the docstrings are prose, not code.

    The docstrings describe the **old** behaviour in several places (they
    show the line the pack used to print), so they must not be mistaken
    for a live literal.
    """
    tree = _tree(path)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _imported_modules(path: Path) -> set[str]:
    """The top level modules a source file imports."""
    imported: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def _calls_of(tree: ast.Module, name: str) -> list[ast.Call]:
    """The calls of a plain function name (``print``, ...)."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _attribute_calls(tree: ast.Module, attribute: str) -> list[ast.Call]:
    """The calls of a method (``download_url_to_file``, ...)."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]


class _Collector:
    """A handler that collects the records of a logger."""

    def __init__(self) -> None:
        import logging

        self._handler = logging.Handler()
        self._handler.emit = self._emit  # type: ignore[method-assign]
        self.records: list = []

    def _emit(self, record) -> None:
        self.records.append(record)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    @property
    def handler(self):
        return self._handler

    @property
    def messages(self) -> list[str]:
        return [record.getMessage() for record in self.records]


# ---------------------------------------------------------------------------
# D1 - the model roots and the download target
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_only_the_path_module_touches_the_folder_registry():
    """The model registry of ComfyUI is touched in **one** module.

    ``folder_paths`` is process-global state: a second module that registers
    a root - or that reads the registry - is a second source of truth, and
    the place where the "the weight is downloaded into the dev instance
    again" defect comes back.
    """
    offenders = []
    for path in PACK_SOURCES:
        if path.name == PATHS_MODULE:
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "folder_paths" for name in names):
                offenders.append((path.name, node.lineno))
    assert offenders == [], offenders


@pytest.mark.unit
def test_the_pack_registers_roots_only_for_its_own_folders():
    """``add_model_folder_path`` is called for the three pack folders only.

    The registry of the host is process-global: a root that is added to a
    folder of ComfyUI (``checkpoints``, ``loras``, ...) would change the
    behaviour of the whole installation (AC-3).
    """
    path = SRC_DIR / PATHS_MODULE
    tree = _tree(path)

    # the names that are iterated over ``MODEL_FOLDERS``
    loop_names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and ast.unparse(node.iter) == "MODEL_FOLDERS"
        ):
            loop_names.add(node.target.id)

    offenders = []
    for node in _attribute_calls(tree, "add_model_folder_path"):
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            name = first.value
        elif isinstance(first, ast.Name):
            name = first.id
        else:
            name = None
        if name is None or (name not in MODEL_FOLDERS and name not in loop_names):
            offenders.append((node.lineno, name))
    assert offenders == [], offenders


@pytest.mark.unit
def test_the_weight_of_the_owner_is_found_and_nothing_is_downloaded(
    keep_paths_module, download_module, mock_folder_paths, tmp_path, monkeypatch
):
    """D1 - the yaml of the owner decides where the weights live.

    The defect in one test: a user keeps the checkpoint in the model root
    of the yaml; the pack ignored the yaml and downloaded the checkpoint
    **again** into the dev instance.

    With the production code the weight is found where the user keeps it,
    the transport is never called and a missing sibling of the same folder
    is downloaded **next to** it.
    """
    owner = tmp_path / "owner"
    checkpoint = owner / "models" / "keep_models" / "KEEP" / "KEEP-b76feb75.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a real weight")

    yaml_path = tmp_path / "extra_model_paths.yaml"
    yaml_path.write_text(
        "comfyui:\n"
        f"  base_path: {owner}\n"
        "  checkpoints: models/checkpoints/\n"
        "  loras: models/loras/\n"
        "other_ui:\n"
        "  custom_nodes: custom_nodes/\n",
        encoding="utf-8",
    )

    keep_paths_module.apply_extra_model_paths(str(yaml_path))

    entry = download_module.WeightEntry(
        name="KEEP-b76feb75.pth",
        url="https://example.invalid/KEEP-b76feb75.pth",
        sha256=None,
        dest_folder="keep_models",
        subdir="KEEP",
        required=True,
        available=True,
        unverified=True,
    )

    transported: list[str] = []

    def _transport(url, dst, hash_prefix=None, progress=True):
        transported.append(url)

    monkeypatch.setattr(download_module, "download_url_to_file", _transport)

    found = download_module.find_weight(entry)
    assert found == str(checkpoint), found

    # the file is where the user keeps it - the transport is never called
    path = download_module.download_weight(entry)
    assert path == str(checkpoint), path
    assert transported == [], transported

    # a missing weight of the same folder is downloaded next to it
    missing = download_module.WeightEntry(
        name="KEEP_Asian-4765ebe0.pth",
        url="https://example.invalid/KEEP_Asian-4765ebe0.pth",
        sha256=None,
        dest_folder="keep_models",
        subdir="KEEP",
        required=False,
        available=True,
        unverified=True,
    )
    target = download_module.target_path(missing)
    assert target == str(
        owner / "models" / "keep_models" / "KEEP" / "KEEP_Asian-4765ebe0.pth"
    ), target


# ---------------------------------------------------------------------------
# D2 / D3 - the logger of ComfyUI and the single ``[KEEP]`` tag
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_old_tag_shape_is_gone_from_the_pack():
    """No call site carries the literal ``"KEEP: "`` any more.

    The old line was ``"[INFO] KEEP: downloading ..."``: the old shape of
    the tag (D3) and a private handler that bypassed the console of
    ComfyUI (D2).
    """
    offenders = [
        (path.name, literal)
        for path in PACK_SOURCES
        for literal in _code_strings(path)
        if "KEEP: " in literal
    ]
    assert offenders == [], offenders


@pytest.mark.unit
def test_the_tag_literal_lives_only_in_the_logger_module():
    """``"[KEEP]"`` is written down in **one** module.

    The tag is added by the filter of the logger; a second literal would be
    a second place where the shape of the console line is decided.
    """
    offenders = [
        (path.name, literal)
        for path in PACK_SOURCES
        if path.name != LOGGER_MODULE
        for literal in _code_strings(path)
        if "[KEEP]" in literal
    ]
    assert offenders == [], offenders


@pytest.mark.unit
def test_the_private_handler_lives_only_in_the_logger_module():
    """Only the logger module builds a handler of its own.

    A handler is what makes the pack print the line **itself**: in a live
    server that is a second console line next to the one of ComfyUI.
    """
    offenders = []
    for path in PACK_SOURCES:
        if path.name == LOGGER_MODULE:
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"StreamHandler", "FileHandler", "NullHandler"}:
                    offenders.append((path.name, node.lineno, node.func.attr))
    assert offenders == [], offenders


@pytest.mark.unit
def test_the_pack_does_not_borrow_the_logger_of_comfyui():
    """The pack keeps its **own** logger name.

    ``logging.getLogger("comfy")`` would put the tag filter of the pack on
    the records of ComfyUI itself - every line of the host would carry the
    ``[KEEP]`` tag.
    """
    offenders = []
    for path in PACK_SOURCES:
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "getLogger"):
                continue
            if node.args and isinstance(node.args[0], ast.Constant):
                if node.args[0].value in {"comfy", "comfyui"}:
                    offenders.append((path.name, node.lineno, node.args[0].value))
    assert offenders == [], offenders


@pytest.mark.unit
def test_every_log_line_of_the_pack_carries_the_tag(
    download_module, logger_module, tmp_path, monkeypatch
):
    """D2 / D3 - the line of the pack is ``"[KEEP] ..."``.

    The old line was ``"[INFO] KEEP: downloading ..."``: the tag was
    written into the message of the call site.  Now the filter of the
    logger adds it - once, to every record.
    """
    entry = download_module.WeightEntry(
        name="KEEP-regression-weight.pth",
        url="https://example.invalid/KEEP-regression-weight.pth",
        sha256=None,
        dest_folder="keep_models",
        subdir="KEEP",
        required=False,
        available=True,
        unverified=True,
    )

    def _transport(url, dst, hash_prefix=None, progress=True):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as handle:
            handle.write(b"not a real weight")

    monkeypatch.setattr(download_module, "download_url_to_file", _transport)

    collector = _Collector()
    logger_module.logger.addHandler(collector.handler)
    try:
        download_module.download_weight(entry)
    finally:
        logger_module.logger.removeHandler(collector.handler)

    assert collector.messages, "the download did not log anything"
    for message in collector.messages:
        assert message.startswith("[KEEP] "), message
        assert "KEEP: " not in message, message


# ---------------------------------------------------------------------------
# D4 - every console write reaches the logger
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_pack_does_not_print_to_the_console():
    """No module of the pack calls ``print()``.

    A print bypasses the logger of ComfyUI: it is untagged and it never
    reaches the log file of the host.  The console writes of the **vendored**
    tree are routed by ``src/keep_console.py`` - the pack itself does not
    print.
    """
    offenders = []
    for path in PACK_SOURCES:
        for node in _calls_of(_tree(path), "print"):
            offenders.append((path.name, node.lineno))
    assert offenders == [], offenders


@pytest.mark.unit
def test_the_pack_does_not_import_tqdm():
    """The progress bars of ``tqdm`` are gone from the pack.

    ``tqdm`` writes to stderr, untagged, and it never reaches the logger
    of the host.  The progress of the node is reported through the ComfyUI
    execution API (``src/keep_progress.py``).
    """
    offenders = [path.name for path in PACK_SOURCES if "tqdm" in _imported_modules(path)]
    assert offenders == [], offenders


@pytest.mark.unit
def test_the_transport_is_never_called_with_a_progress_bar():
    """``download_url_to_file`` is called with ``progress=False``.

    The bar of ``torch.hub`` is written to stderr, untagged - it would
    bypass the logger of the pack (D4).
    """
    offenders = []
    for path in PACK_SOURCES:
        for node in ast.walk(_tree(path)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "download_url_to_file"
            ):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            value = keywords.get("progress")
            if not (isinstance(value, ast.Constant) and value.value is False):
                offenders.append((path.name, node.lineno))
    assert offenders == [], offenders


@pytest.mark.unit
def test_the_calls_into_the_vendored_tree_are_wrapped_in_the_capture():
    """The facelib calls run inside ``capture_console_to_logger()``.

    The vendored helper prints while it is built and while it detects -
    those writes carry the tag only when they run inside the capture.
    """
    for name in ("keep_processor.py", "keep_model_loader.py"):
        tree = _tree(SRC_DIR / name)
        wrapped = any(
            isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == "capture_console_to_logger"
                for item in node.items
            )
            for node in ast.walk(tree)
        )
        assert wrapped, (
            f"{name} does not capture the console writes of the vendored tree"
        )


#: The V1 progress bar of the host is allowed in exactly one surface: the
#: guarded download-progress factory of the downloader (its lazy import) and
#: the one-step report of a finished download.  Every other use - in
#: particular the processing loop - must go through the V3 execution API
#: (``src/keep_progress.py``).
DOWNLOAD_BAR_FUNCTIONS = ("_host_progress_bar", "_report_download_done")


def _progressbar_offenders(path: Path) -> list[tuple[str, int]]:
    """The ProgressBar references outside the one allowed download surface."""
    tree = _tree(path)
    allowed: list[tuple[int, int]] = []
    if path.name == "keep_download.py":
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name in DOWNLOAD_BAR_FUNCTIONS
            ):
                allowed.append((node.lineno, node.end_lineno))
    offenders = []
    for node in ast.walk(tree):
        referenced = (
            isinstance(node, ast.Attribute) and node.attr == "ProgressBar"
        ) or (
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "ProgressBar" for alias in node.names)
        )
        if referenced and not any(
            low <= node.lineno <= high for low, high in allowed
        ):
            offenders.append((path.name, node.lineno))
    return offenders


@pytest.mark.unit
def test_the_progress_of_the_pack_goes_through_the_comfy_api():
    """The pack reports progress through the V3 execution API.

    The V1 progress bar of ``comfy.utils`` writes to the console of the host
    without the tag of the pack - the processing loop must not touch it.  The
    one allowed surface is the guarded download bar of the downloader
    (``DOWNLOAD_BAR_FUNCTIONS``): one step per weight file, on the bar the
    host supplies during prompt execution.
    """
    offenders = [
        offender for path in PACK_SOURCES for offender in _progressbar_offenders(path)
    ]
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# D5 - only the selected face detector is downloaded
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_loader_never_resolves_the_whole_manifest():
    """The loader does not loop over the manifest of the facelib folder.

    The old code downloaded **every** detector of the manifest (D5) - three
    of them are not used at all.  The selection decides the weight; the
    manifest is only a lookup table.
    """
    tree = _tree(SRC_DIR / "keep_model_loader.py")
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.comprehension)):
            if isinstance(node.iter, ast.Name) and node.iter.id in {
                "WEIGHT_MANIFEST",
                "FACELIB_MODEL_URLS",
            }:
                offenders.append((node.lineno, node.iter.id))
    assert offenders == [], offenders


@pytest.mark.unit
def test_the_loader_does_not_spell_out_a_weight_file_name():
    """The weight files are named by the manifest and by the vendored helper.

    A literal file name in the loader or in the downloader would be a
    second table - and it would drift from the selection.
    """
    for name in ("keep_model_loader.py", "keep_download.py"):
        offenders = [
            literal
            for literal in _code_strings(SRC_DIR / name)
            if literal in FACELIB_WEIGHT_NAMES
        ]
        assert offenders == [], (name, offenders)


@pytest.mark.unit
def test_the_widget_options_come_from_the_config_module():
    """The detector list of the widget has **one** source.

    ``DETECTION_MODELS`` lives in ``src/keep_model_configs.py`` and the node
    imports it.  A second list in the node would drift from the mapping of
    the vendored helper (and from the order the widget promises).
    """
    tree = _tree(NODES_DIR / "keep_model_loader.py")
    option_lists = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "options":
                option_lists.append((node.lineno, ast.unparse(keyword.value)))

    assert option_lists, "the node does not define any option list"

    detector_options = [text for _, text in option_lists if "DETECTION_MODELS" in text]
    assert detector_options == ["list(DETECTION_MODELS)"], option_lists

    for _, text in option_lists:
        if not text.startswith("["):
            continue
        for detector in DETECTOR_NAMES:
            assert detector not in text, (detector, text)


# ---------------------------------------------------------------------------
# the invariants of the repository
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_vendored_tree_is_untouched():
    """``modules/deps`` is upstream code - this pack does not patch it.

    The vendored tree is read-only (deviation D7): the console writes of the
    facelib are routed around it, never edited in it.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", "modules/deps"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"git is not usable in this checkout: {exc}")
    assert result.stdout.strip() == "", (
        "the vendored tree must not be modified:\n" + result.stdout
    )


@pytest.mark.unit
def test_the_runtime_dependencies_agree_on_the_slimmed_set():
    """``requirements.txt`` and ``pyproject.toml`` agree - on the slimmed set.

    The two files are the declared runtime set of the pack and must not
    drift apart.  The host-guarantee policy declares **one** dependency -
    ``opencv-python-headless``: the ComfyUI host ships torch, torchvision,
    numpy, einops, pyyaml, Pillow, scipy, tqdm and requests itself (its own
    requirements.txt), and ``tqdm`` in particular is gone from the pack's
    own code (the progress is reported through the ComfyUI execution API).
    """
    requirements = set()
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        requirements.add(name.split(";")[0].strip())

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            pytest.skip("no TOML parser available - install ``tomli`` on Python 3.10")
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)

    declared = set()
    for requirement in pyproject["project"]["dependencies"]:
        declared.add(requirement.split(";")[0].split("[")[0].strip())

    assert requirements == declared, (sorted(requirements), sorted(declared))
    assert requirements == {"opencv-python-headless"}, sorted(requirements)


# ---------------------------------------------------------------------------
# The GPU video pipeline redesign - the tripwires of the batching
# ---------------------------------------------------------------------------
# The redesign of the sequence path (the design document
# ``.dev/docs/2026-10-25-keep-gpu-video-pipeline.md``, sections 7.1-7.8)
# replaced the per-frame loops of ``process_image_sequence`` with batched
# torch ops.  The **behavioural** pins live with the code that wrote them
# (``tests/test_gpu_batch_equivalence.py``, ``tests/test_gpu_paste_equivalence.py``)
# and in the processor tests (``tests/test_keep_processor.py``); the guards
# below are the second line of defence - they catch the defect coming back
# in a **different** place, the way the guards of the five defects do.
#
# ========== ================================================================
# G1         the per-frame host<->device transfers of the conversions
# G2         the read-once contract (one ``read_image`` per frame)
# G3         the affine is estimated once (no second ``align_warp_face``)
# G4         the KEEP chunk size is the quality contract (``max_clip_length``)
# G5         the vendored paste fallback stays reachable
# G6         the progress contract: ``num_frames * 4`` steps, four families
# ========== ================================================================

#: The module that owns the sequence path.
PROCESSOR_MODULE = "keep_processor.py"

#: The function of the sequence path that the redesign rewired.
SEQUENCE_PATH = "process_image_sequence"


def _sequence_node() -> ast.FunctionDef:
    """The AST of ``process_image_sequence``."""
    for node in ast.walk(_tree(SRC_DIR / PROCESSOR_MODULE)):
        if isinstance(node, ast.FunctionDef) and node.name == SEQUENCE_PATH:
            return node
    raise AssertionError(f"{SEQUENCE_PATH} is not in {PROCESSOR_MODULE}")


def _attribute_calls_in(node: ast.AST, attribute: str) -> list[ast.Call]:
    """The attribute calls inside a node (``helper.read_image(...)``)."""
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == attribute
    ]


@pytest.mark.unit
def test_the_sequence_path_does_not_go_back_to_the_per_frame_converters():
    """G1 - the conversions of the sequence path stay batched.

    The base path converted the frames one by one: ``comfy_image_to_cv2``
    per frame, ``tensor2img`` per frame and ``cv2_to_comfy_image`` per
    frame - one host<->device transfer per frame and per conversion.  That
    is the cost the redesign removed; the batched forms must stay the ones
    the sequence path calls.
    """
    node = _sequence_node()
    for name in ("comfy_image_to_cv2", "cv2_to_comfy_image", "tensor2img"):
        assert _calls_of(node, name) == [], (
            f"the sequence path calls the per-frame {name} again"
        )
    for name in (
        "comfy_batch_to_bgr_u8",
        "tensor2img_batch",
        "bgr_u8_batch_to_comfy",
    ):
        assert _calls_of(node, name), (
            f"the sequence path does not call the batched {name}"
        )


@pytest.mark.unit
def test_the_sequence_path_reads_every_frame_once():
    """G2 - one ``read_image`` per frame.

    The base path read every frame three times: the detection, the crop and
    the paste.  The redesign reads it once (the detection) and the crop
    warps the image the detector saw.  A second call site in the sequence
    path would bring the per-frame copies back.
    """
    assert len(_attribute_calls_in(_sequence_node(), "read_image")) == 1, (
        "the sequence path must read every frame exactly once"
    )


@pytest.mark.unit
def test_the_sequence_path_estimates_the_affine_once():
    """G3 - the crop owns the affine, the paste reuses it.

    The base path ran ``align_warp_face`` twice per frame (the crop and the
    paste) - two ``cv2.estimateAffinePartial2D`` calls for the same
    landmarks.  The redesign estimates the matrices once (``estimate_affines``
    - the same vendored call) and the paste loop reuses them.
    """
    node = _sequence_node()
    assert _attribute_calls_in(node, "align_warp_face") == [], (
        "the sequence path runs align_warp_face again"
    )
    assert _calls_of(node, "estimate_affines"), (
        "the sequence path does not estimate the affines"
    )


@pytest.mark.unit
def test_the_keep_chunks_follow_the_quality_contract():
    """G4 - the chunk size of the KEEP inference is ``max_clip_length``.

    ``max_clip_length`` is a **quality contract**: the temporal propagation
    of KEEP changes the output pixels with the chunk boundaries, so the
    boundaries must not move.  A VRAM-derived chunk size (the pattern of the
    parse batching, which is a performance knob) would silently change the
    output of the node.
    """
    chunk_loops = [
        call
        for call in _calls_of(_sequence_node(), "range")
        if len(call.args) == 3
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 0
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "num_input_frames"
    ]
    assert len(chunk_loops) == 1, chunk_loops
    step = chunk_loops[0].args[2]
    assert isinstance(step, ast.Name) and step.id == "max_clip_length", (
        f"the KEEP chunks step by {ast.unparse(step)} instead of max_clip_length"
    )


@pytest.mark.unit
def test_the_vendored_paste_fallback_stays_reachable():
    """G5 - the vendored paste is the fallback of the non-default branches.

    The batched paste takes the default branch (no box, no face upsampler,
    a color frame, a parse net).  Every other frame keeps the vendored
    ``paste_faces_to_input_image`` - the call path must not be deleted, it
    **is** the behaviour of the non-default branches.
    """
    tree = _tree(SRC_DIR / PROCESSOR_MODULE)
    assert _attribute_calls(tree, "paste_faces_to_input_image"), (
        "the vendored paste fallback is gone from the processor"
    )


#: The vendored 5-point 512 template (``face_restoration_helper.py:52-53``).
FACE_TEMPLATE_512 = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]]
)


class _ProgressKeepNet:
    """A KEEP net that returns zeros - the progress is the subject here."""

    def __call__(self, frames, need_upscale=False):
        return torch.zeros_like(frames)


class _ProgressFaceHelper:
    """The facelib helper, reduced to the calls the sequence path makes.

    It detects **one** face on every frame (at the template landmarks, so
    the affine of the crop is the identity and the batched crop really
    runs) and it has **no** parse net: the batched paste does not take the
    frame and the vendored fallback of the processor runs.  Both branches
    report the same progress families.
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.upscale_factor = 1.0
        self.is_gray = False
        self.cropped_faces: list = []
        self.restored_faces: list = []
        self.all_landmarks_5: list = []
        self.affine_matrices: list = []
        self.input_img = None
        self.face_template = FACE_TEMPLATE_512
        self.calls: list[str] = []

    def clean_all(self):
        self.calls.append("clean_all")
        self.cropped_faces = []
        self.restored_faces = []

    def read_image(self, image):
        self.calls.append("read_image")
        self.input_img = image

    def get_face_landmarks_5(self, **kwargs):
        self.calls.append("get_face_landmarks_5")
        self.all_landmarks_5 = [FACE_TEMPLATE_512.copy()]
        return 1

    def add_restored_face(self, face, **kwargs):
        self.calls.append("add_restored_face")
        self.restored_faces = [face]

    def get_inverse_affine(self, *args, **kwargs):
        self.calls.append("get_inverse_affine")

    def paste_faces_to_input_image(self, *args, **kwargs):
        self.calls.append("paste_faces_to_input_image")
        return np.zeros((16, 16, 3), dtype=np.uint8)


def _progress_pack(keep_net, face_helper):
    """A model pack the processor can consume."""
    return types.SimpleNamespace(
        keep_net=keep_net,
        face_helper=face_helper,
        bg_upscale_model=None,
        face_upscale_model=None,
        device=torch.device("cpu"),
        model_type_str="KEEP",
    )


@pytest.mark.unit
def test_the_progress_reaches_the_total_of_the_frame_count(monkeypatch):
    """G6 - the progress contract of the batched sequence path.

    The reporter is created with ``num_frames * 4`` steps and the four
    families of the updates stay: the conversion (one batched update of
    ``N``), the detection (one update of ``N``), the KEEP chunks (one
    update per chunk - they sum up to ``N``) and the paste (one update per
    frame).  The pin drives the **real** reporter (``src/keep_progress.py``)
    with five frames and two KEEP chunks, so the chunk family is measured
    with more than one chunk - the case the single-chunk pin of the
    processor tests does not reach.
    """
    progress_module = importlib.import_module("ComfyUI_KEEP.src.keep_progress")
    processor_module = importlib.import_module("ComfyUI_KEEP.src.keep_processor")

    reported: list[tuple[float, float]] = []

    def _record(value, max_value):
        reported.append((value, max_value))

    monkeypatch.setattr(progress_module, "report_progress", _record)

    processor = processor_module.KEEPFaceProcessor(
        _progress_pack(_ProgressKeepNet(), _ProgressFaceHelper())
    )
    frames = torch.rand((5, 16, 16, 3), dtype=torch.float32)
    processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=2,
    )

    # the total is the frame count times the five families (round 4
    # D-R2.3.1 added the paste stage's second convert-in: single *5)
    assert {max_value for _, max_value in reported} == {25}, reported
    # 5 (convert) + 5 (detection) + 2 + 2 + 1 (the KEEP chunks)
    # + 5 (the second convert-in) + 5 x 1 (paste)
    assert [value for value, _ in reported] == [
        5, 10, 12, 14, 15, 20, 21, 22, 23, 24, 25
    ], reported


# ---------------------------------------------------------------------------
# D1 - the end to end proof against the **real** ComfyUI (the headless tier)
# ---------------------------------------------------------------------------
#: The probe: the pack is imported against the real ComfyUI source tree, the
#: yaml of the owner is applied and the weight of the owner is looked up.
#: The transport is replaced by a double that records the call - the probe
#: fails when the weight is downloaded again.
_DOWNLOADER_EXTRA_ROOTS_PROBE = """
import json, os, sys

comfyui_root, repo_root, workdir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, comfyui_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "tests"))

import _headless

_headless.register_package(repo_root, alias="ComfyUI_KEEP_regressions", execute=False)
import folder_paths

# The primary root of the probe is a **temporary** tree: the probe must
# never write into the ComfyUI source tree.
models_dir = os.path.join(workdir, "models")
os.makedirs(models_dir, exist_ok=True)
folder_paths.models_dir = models_dir

from ComfyUI_KEEP_regressions.src import keep_download, keep_paths

# the probe reads **only** its own yaml - never the one of the real tree
keep_paths._default_extra_model_paths = lambda: None

owner = os.path.join(workdir, "owner")
extra_root = os.path.join(owner, "models", "keep_models")
checkpoint = os.path.join(extra_root, "KEEP", "KEEP-b76feb75.pth")
os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
with open(checkpoint, "wb") as handle:
    handle.write(b"not a real weight")

yaml_path = os.path.join(workdir, "extra_model_paths.yaml")
with open(yaml_path, "w", encoding="utf-8") as handle:
    handle.write(
        "comfyui:\\n"
        "  base_path: " + owner + "\\n"
        "  checkpoints: models/checkpoints/\\n"
        "  loras: models/loras/\\n"
        "pack:\\n"
        "  keep_models: " + extra_root + "\\n"
        "other_ui:\\n"
        "  custom_nodes: custom_nodes/\\n"
    )

keep_paths.apply_extra_model_paths(yaml_path)

transported = []


def _transport(url, dst, hash_prefix=None, progress=True):
    transported.append(url)


keep_download.download_url_to_file = _transport

entry = keep_download.WeightEntry(
    name="KEEP-b76feb75.pth",
    url="https://example.invalid/KEEP-b76feb75.pth",
    sha256=None,
    dest_folder="keep_models",
    subdir="KEEP",
    required=True,
    available=True,
    unverified=True,
)

found = keep_download.find_weight(entry)
path = keep_download.download_weight(entry)

print(
    "KEEP_DOWNLOADER_EXTRA_ROOTS:"
    + json.dumps(
        {
            "found": found,
            "path": path,
            "transported": transported,
            "roots": keep_paths.get_model_folder_paths("keep_models"),
            "primary": os.path.join(models_dir, "keep_models"),
        }
    )
)
"""


@pytest.mark.headless
def test_the_real_comfyui_finds_the_weight_of_the_owner(tmp_path):
    """D1 - against the real ComfyUI: the weight is **not** downloaded again.

    The yaml of the owner is a **temporary** file in the shape of a user's
    ``extra_model_paths.yaml``: the tree of the owner is never read and
    never written.  The real ``folder_paths`` and the real
    ``utils.extra_config`` of the host do the work.

    The defect was exactly this: the weight existed in the model root of
    the yaml and was downloaded into the dev instance **anyway**.
    """
    value = (os.environ.get(COMFYUI_ROOT_ENV) or "").strip()
    if not value:
        pytest.skip(
            f"{COMFYUI_ROOT_ENV} is not set - the headless downloader test "
            "needs the ComfyUI source tree"
        )
    root = Path(value).expanduser()
    if not (root / "folder_paths.py").is_file():
        pytest.skip(
            f"{COMFYUI_ROOT_ENV}={value!r} is not a ComfyUI source tree "
            "(folder_paths.py is missing)"
        )

    workdir = tmp_path / "probe"
    workdir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _DOWNLOADER_EXTRA_ROOTS_PROBE,
            str(root.resolve()),
            str(REPO_ROOT),
            str(workdir),
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
        if line.startswith("KEEP_DOWNLOADER_EXTRA_ROOTS:")
    ]
    assert sentinel, (
        f"the probe produced no report\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    report = json.loads(sentinel[-1].split(":", 1)[1])

    extra_root = str(workdir / "owner" / "models" / "keep_models")
    checkpoint = os.path.join(extra_root, "KEEP", "KEEP-b76feb75.pth")

    # the weight is found where the owner keeps it
    assert report["found"] == checkpoint, report
    assert report["path"] == checkpoint, report
    # and it is **not** downloaded again
    assert report["transported"] == [], report
    # the extra root is registered, the primary root stays the first one
    assert extra_root in report["roots"], report
    assert os.path.normcase(report["roots"][0]) == os.path.normcase(
        report["primary"]
    ), report
    # the probe did not write into the ComfyUI source tree
    assert str(report["primary"]).startswith(str(workdir)), report
