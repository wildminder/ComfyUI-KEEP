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
UTILS_FILE = PACKAGE_DIR / "utils.py"
AUDIT_DOC = REPO_ROOT / "docs" / "audit" / "2026-10-22-dependency-decisions.md"

EXPECTED_DIFFUSERS_SYMBOLS = {"FeedForward", "AdaLayerNorm"}


def _diffusers_imports() -> list[tuple[str, int, set[str]]]:
    """Every diffusers import in the package: (file, line, imported names)."""
    found = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
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
    """Execute ``modules/utils.py`` the way the package does.

    The module needs two ComfyUI symbols (``folder_paths`` and
    ``comfy.utils.ProgressBar``) and the parent package's ``logger``.  Only
    those *ComfyUI* seams are stubbed - the vendored dependencies are **not**,
    so the real ``from re import T`` failure still happens and the real
    fallback behaviour is observed.
    """
    saved_path = list(sys.path)
    saved_modules = dict(sys.modules)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(REPO_ROOT / "models")
    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")

    class ProgressBar:  # minimal stand-in for comfy.utils.ProgressBar
        def __init__(self, total=None):
            self.total = total

        def update(self, value=1):
            return None

    comfy_utils.ProgressBar = ProgressBar
    comfy.utils = comfy_utils

    parent_name = "keep_p0_probe"
    parent = types.ModuleType(parent_name)
    parent.__path__ = [str(REPO_ROOT)]  # type: ignore[attr-defined]
    parent.logger = logging.getLogger(parent_name)

    package_name = f"{parent_name}.modules"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PACKAGE_DIR)]  # type: ignore[attr-defined]

    child_name = f"{package_name}.utils"
    child = types.ModuleType(child_name)
    child.__file__ = str(UTILS_FILE)
    child.__package__ = package_name
    child.__path__ = [str(PACKAGE_DIR)]  # type: ignore[attr-defined]

    sys.modules["folder_paths"] = folder_paths
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = comfy_utils
    sys.modules[parent_name] = parent
    sys.modules[package_name] = package
    sys.modules[child_name] = child

    try:
        source = UTILS_FILE.read_text(encoding="utf-8")
        exec(compile(source, str(UTILS_FILE), "exec"), child.__dict__)
        yield child
    finally:
        sys.path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved_modules)


@pytest.mark.unit
def test_exactly_one_diffusers_import():
    imports = _diffusers_imports()
    assert len(imports) == 1, f"expected exactly one diffusers import, got {imports}"
    path, line, names = imports[0]
    assert path == "modules/deps/basicsr/archs/keep_arch.py", (path, line)
    assert names == EXPECTED_DIFFUSERS_SYMBOLS, (
        f"the vendored diffusers import must request exactly "
        f"{sorted(EXPECTED_DIFFUSERS_SYMBOLS)}, got {sorted(names)}"
    )


@pytest.mark.unit
def test_diffusers_import_targets_the_expected_symbols():
    """The import must name the two symbols P2.2 vendors byte-for-byte."""
    path = REPO_ROOT / "modules" / "deps" / "basicsr" / "archs" / "keep_arch.py"
    source = path.read_text(encoding="utf-8")
    assert (
        "from diffusers.models.attention import FeedForward, AdaLayerNorm" in source
    ), "the diffusers import line changed - P2.2 vendors these two symbols"


@pytest.mark.unit
def test_audit_doc_records_the_diffusers_decision():
    text = AUDIT_DOC.read_text(encoding="utf-8")
    assert "FeedForward" in text and "AdaLayerNorm" in text
    assert "0.11" in text, "the diffusers version must be recorded"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "flownet_path is not wired yet (modules/utils.py sets it to None) - "
        "flip owner: P5.5, which resolves the GMFlow manifest entry and sets "
        "architecture['flownet_path'] before the arch is built"
    ),
)
@pytest.mark.unit
def test_flownet_path_is_wired(utils_module):
    """The arch config must resolve a real flownet path (flip owner: P5.5)."""
    architecture = utils_module.KEEP_MODEL_CONFIGS["KEEP"]["architecture"]
    assert architecture["flownet_path"] is not None, (
        "flownet_path is None - the GMFlow net would keep its random weights"
    )


@pytest.mark.unit
def test_flownet_is_built_by_the_arch(utils_module):
    """The vendored arch builds the flownet from the config (code read check)."""
    arch = REPO_ROOT / "modules" / "deps" / "basicsr" / "archs" / "keep_arch.py"
    source = arch.read_text(encoding="utf-8")
    assert "self.flownet = FlowGenerator(path=flownet_path)" in source
    assert "flows = self.get_flow(x)" in source
