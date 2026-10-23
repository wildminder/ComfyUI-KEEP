"""P0.4 - dependency decision tests.

Two guards, both named by the plan:

* a **grep guard**: the package must contain exactly one diffusers import, and
  it must import exactly ``FeedForward`` and ``AdaLayerNorm`` (the two symbols
  that P2.2 vendors from diffusers 0.11.x);
* an **xfail guard** "flownet path is wired": the architecture config that is
  handed to the KEEP arch must resolve a non-``None`` ``flownet_path``.
  It is marked ``xfail(strict=False)`` because it is expected to fail until
  **P5.5** wires the GMFlow resolution (the flip owner recorded in the
  reason string and in the audit document).
"""

from __future__ import annotations

import ast
import logging
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "modules"
SRC_DIR = REPO_ROOT / "src"
NODES_DIR = REPO_ROOT / "nodes"
UTILS_FILE = SRC_DIR / "keep_model_configs.py"
AUDIT_DOC = REPO_ROOT / "docs" / "audit" / "2026-10-22-dependency-decisions.md"

EXPECTED_DIFFUSERS_SYMBOLS = {"FeedForward", "AdaLayerNorm"}


def _package_files() -> list[Path]:
    """Every production python file of the pack (P3.2 layout)."""
    files: list[Path] = []
    for directory in (PACKAGE_DIR, SRC_DIR, NODES_DIR):
        files.extend(sorted(directory.rglob("*.py")))
    return files


def _diffusers_imports() -> list[tuple[str, int, set[str]]]:
    """Every diffusers import in the package: (file, line, imported names)."""
    found = []
    for path in _package_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == "diffusers":
                    names = {alias.name for alias in node.names}
                    found.append((path.relative_to(REPO_ROOT).as_posix(), node.lineno, names))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "diffusers":
                        found.append(
                            (path.relative_to(REPO_ROOT).as_posix(), node.lineno, {alias.name})
                        )
    return found


@pytest.fixture(scope="module")
def utils_module():
    """The model configs of the package, imported the way the package does.

    Since P3.2 the configs live in ``src/keep_model_configs.py``; the module
    is imported through the package alias, so the *real* module is observed
    (the ComfyUI seams - ``folder_paths`` and ``comfy.utils`` - are the ones
    of the root conftest's mocks).
    """
    import importlib

    return importlib.import_module("ComfyUI_KEEP.src.keep_model_configs")


@pytest.mark.unit
def test_no_diffusers_import_remains_in_the_vendored_tree():
    """P2.2 flipped this guard: **no** diffusers import may remain.

    The two symbols are vendored byte-for-byte in
    ``modules/deps/basicsr/archs/_diffusers_compat.py``; the vendored tree
    must not depend on diffusers any more.
    """
    imports = _diffusers_imports()
    assert imports == [], f"no diffusers import may remain, got {imports}"


@pytest.mark.unit
def test_the_vendored_compat_module_provides_the_expected_symbols():
    """The vendored module must define exactly the two planned symbols."""
    path = (
        REPO_ROOT
        / "modules"
        / "deps"
        / "basicsr"
        / "archs"
        / "_diffusers_compat.py"
    )
    assert path.is_file(), f"the vendored compat module is missing: {path}"
    source = path.read_text(encoding="utf-8")
    for symbol in sorted(EXPECTED_DIFFUSERS_SYMBOLS):
        assert f"class {symbol}(nn.Module):" in source, symbol
    # the attribution header must name the upstream project and version
    assert "diffusers 0.11.0" in source, (
        "the attribution header must name diffusers 0.11.0"
    )
    assert "Apache License" in source, (
        "the attribution header must name the Apache-2.0 license"
    )

    # and the architecture must use the vendored copy
    keep_arch = (
        REPO_ROOT / "modules" / "deps" / "basicsr" / "archs" / "keep_arch.py"
    ).read_text(encoding="utf-8")
    assert "from ._diffusers_compat import FeedForward, AdaLayerNorm" in keep_arch, (
        "keep_arch.py must import the vendored classes"
    )


@pytest.mark.unit
def test_audit_doc_records_the_diffusers_decision():
    text = AUDIT_DOC.read_text(encoding="utf-8")
    assert "FeedForward" in text and "AdaLayerNorm" in text
    assert "0.11" in text, "the diffusers version must be recorded"


@pytest.mark.unit
def test_flownet_path_is_wired(monkeypatch, tmp_path):
    """P5.5 flipped this guard: the flownet path is resolved before the arch.

    The P0.4 guard asserted that the **config table** carries a non-``None``
    ``flownet_path``.  That premise was disproved by P0.4 section 2: the
    shipped KEEP checkpoint carries the trained flownet weights (123
    ``flownet.*`` tensors), so the GMFlow file is only needed when a
    checkpoint does **not** provide them - the resolution is conditional.

    The guard now asserts the guarantee that actually matters: a checkpoint
    **without** flownet weights gets a **resolved** (non-``None``)
    ``flownet_path`` in the config that is handed to the arch - the flow net
    must never keep its random weights.
    """
    import importlib

    import torch

    loader_module = importlib.import_module("ComfyUI_KEEP.src.keep_model_loader")
    built: list[dict] = []

    class _StubArch:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def to(self, device):
            return self

        def load_state_dict(self, state_dict, strict=True):
            return self

        def eval(self):
            return self

    class _StubHelper:
        def __init__(self, **kwargs):
            pass

    gmflow = str(tmp_path / "gmflow_sintel-0c07dcb3.pth")
    monkeypatch.setattr(loader_module, "ARCH_REGISTRY", {"KEEP": _StubArch})
    monkeypatch.setattr(
        loader_module.torch, "load", lambda *a, **k: {"params_ema": {"w": torch.zeros(1)}}
    )
    monkeypatch.setattr(loader_module, "_face_restore_helper", lambda: _StubHelper)
    monkeypatch.setattr(
        loader_module.keep_paths, "resolve_model_root", lambda *a, **k: str(tmp_path)
    )
    monkeypatch.setattr(
        loader_module.keep_download,
        "ensure_weight",
        lambda entry, progress=True: str(tmp_path / "checkpoint.pth"),
    )
    monkeypatch.setattr(
        loader_module.keep_download,
        "ensure_required",
        lambda entry, progress=True: str(tmp_path / "weight.pth"),
    )
    monkeypatch.setattr(
        loader_module.keep_download,
        "ensure_optional",
        lambda entry, progress=True: gmflow,
    )

    loader_module.KEEPModelLoader().load_keep_model_pack(
        "KEEP", "retinaface_resnet50"
    )

    assert built, "the architecture was never built"
    assert built[0]["flownet_path"] is not None, (
        "flownet_path is None - the GMFlow net would keep its random weights"
    )
    assert built[0]["flownet_path"] == gmflow


@pytest.mark.unit
def test_flownet_is_built_by_the_arch(utils_module):
    """The vendored arch builds the flownet from the config (code read check)."""
    arch = REPO_ROOT / "modules" / "deps" / "basicsr" / "archs" / "keep_arch.py"
    source = arch.read_text(encoding="utf-8")
    assert "self.flownet = FlowGenerator(path=flownet_path)" in source
    assert "flows = self.get_flow(x)" in source
