"""P1.3 - structure guards.

The guards are **source-text based**: they never import the package, they read
the files.  That keeps them honest while the package is being moved around
(the very thing they guard) and lets them run on a throw-away copy of the
repository.

Five guards, each with the step that flips it from ``xfail`` to strict:

===== ================================================ ==================
guard  what it checks                                  flip owner
===== ================================================ ==================
(a)    every class under ``nodes/*.py`` is exported     **P3.1**
       from ``nodes/__init__.py``
(b)    ``src/**`` contains no ``io.Schema(`` /           **P3.2**
       ``define_schema``
(c)    the root ``__init__.py`` exposes exactly          **P4.4**
       ``comfy_entrypoint`` (no V1 mapping names)
(d)    no ``sys.path.insert`` / ``sys.path.append``      **P2.5**
       anywhere in the package (the vendored tree
       included - it was exempt only until P2.5)
(e)    no ``pdb`` / ``torch.profiler`` / ``import pip``  **P2.1**
       / ``pip.main`` in the vendored runtime closure
===== ================================================ ==================

Every guard is ``xfail(strict=False)`` until its owner step lands; the reason
of each xfail names that step, so an xfail can never be orphaned.  The
self-tests at the bottom prove that each guard really fails on a deliberately
violating copy - a guard that cannot fail is worthless.
"""

from __future__ import annotations

import ast
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The production sources the ``sys.path`` guard (d) looks at.  Test code
#: (``tests/``, ``tools/``, ``conftest.py``) is not part of the package and is
#: therefore out of scope - the guard is about the shipped package.
PACKAGE_SOURCE_ROOTS = ("__init__.py", "nodes.py", "nodes", "src", "modules")

#: The patterns of guard (e).  They are the ``pdb``/``torch_profiler``/
#: ``import_pip``/``pip_main`` patterns of the P0.3 scanner, so the guard and
#: the scanner never drift apart.
BREAKAGE_PATTERNS = ("pdb", "torch_profiler", "import_pip", "pip_main")

_SYS_PATH_PATTERN = re.compile(r"sys\.path\s*\.\s*(insert|append)\s*\(")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _package_files(root: Path) -> list[Path]:
    """Every production python file of the package (the vendored tree included).

    The vendored tree used to be exempt from guard (d): it was a plain folder
    on ``sys.path`` until P2.5 package-ized it.  Since P2.5 it is a real part
    of the package, so the plan's §3 rule ("no ``sys.path`` mutation anywhere
    in the package") applies to it as well - the last vendored mutation was
    removed in the P2 review fix.
    """
    files: list[Path] = []
    for entry in PACKAGE_SOURCE_ROOTS:
        target = root / entry
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
    return files


def _closure_files(root: Path) -> list[Path]:
    """The static runtime import closure of the package (P0.3 machinery).

    ``tools/scan_compat.py`` resolves the vendored imports against its own
    ``DEPS_DIR``; the guard runs against arbitrary roots (the self-test uses a
    copy), so the directory is pointed at the root in question.
    """
    from tools import scan_compat  # noqa: PLC0415 - local import keeps the module light

    original_deps_dir = scan_compat.DEPS_DIR
    scan_compat.DEPS_DIR = root / "modules" / "deps"
    try:
        entry_points = [root / rel for rel in scan_compat.ENTRY_POINTS]
        return scan_compat.find_import_closure(entry_points)
    finally:
        scan_compat.DEPS_DIR = original_deps_dir


# ---------------------------------------------------------------------------
# guard (a) - the node classes are exported
# ---------------------------------------------------------------------------
def nodes_export_violations(root: Path) -> list[str]:
    """Every class defined in ``nodes/*.py`` must be exported by ``nodes/__init__.py``."""
    nodes_dir = root / "nodes"
    if not nodes_dir.is_dir():
        return ["nodes/ does not exist yet (P3.1 creates it)"]

    init_file = nodes_dir / "__init__.py"
    if not init_file.is_file():
        return ["nodes/__init__.py does not exist"]

    exported: set[str] = set()
    tree = ast.parse(_read(init_file), filename=str(init_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                exported.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                exported.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    for element in ast.walk(node.value):
                        if isinstance(element, ast.Constant) and isinstance(element.value, str):
                            exported.add(element.value)

    violations: list[str] = []
    for path in sorted(nodes_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        module_tree = ast.parse(_read(path), filename=str(path))
        for node in module_tree.body:
            if isinstance(node, ast.ClassDef) and node.name not in exported:
                violations.append(
                    f"{_relative(path, root)}: class {node.name} is not exported "
                    f"by nodes/__init__.py"
                )
    return violations


# ---------------------------------------------------------------------------
# guard (b) - src/ holds no schema
# ---------------------------------------------------------------------------
def src_schema_violations(root: Path) -> list[str]:
    """``src/**`` must not define a node schema (schemas belong to ``nodes/``)."""
    src_dir = root / "src"
    if not src_dir.is_dir():
        return ["src/ does not exist yet (P3.2 creates it)"]

    violations: list[str] = []
    for path in sorted(src_dir.rglob("*.py")):
        text = _read(path)
        for number, line in enumerate(text.splitlines(), 1):
            if "io.Schema(" in line or "define_schema" in line:
                violations.append(
                    f"{_relative(path, root)}:{number}: schema definition in src/ "
                    f"({line.strip()})"
                )
    return violations


# ---------------------------------------------------------------------------
# guard (c) - exactly one entry point
# ---------------------------------------------------------------------------
def entrypoint_violations(root: Path) -> list[str]:
    """The root ``__init__.py`` must expose exactly ``comfy_entrypoint``."""
    init_file = root / "__init__.py"
    if not init_file.is_file():
        return ["__init__.py does not exist"]

    text = _read(init_file)
    violations: list[str] = []
    if "comfy_entrypoint" not in text:
        violations.append("__init__.py does not define comfy_entrypoint")
    for v1_name in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
        if v1_name in text:
            violations.append(f"__init__.py still exports the V1 name {v1_name}")
    return violations


# ---------------------------------------------------------------------------
# guard (d) - no sys.path mutation
# ---------------------------------------------------------------------------
def sys_path_violations(root: Path) -> list[str]:
    """No ``sys.path.insert``/``append`` anywhere in the package."""
    violations: list[str] = []
    for path in _package_files(root):
        for number, line in enumerate(_read(path).splitlines(), 1):
            if _SYS_PATH_PATTERN.search(line):
                violations.append(f"{_relative(path, root)}:{number}: {line.strip()}")
    return violations


# ---------------------------------------------------------------------------
# guard (e) - no debug/pip leftovers in the vendored closure
# ---------------------------------------------------------------------------
def vendored_breakage_violations(root: Path) -> list[str]:
    """No ``pdb``/``torch.profiler``/``import pip``/``pip.main`` in the closure."""
    from tools import scan_compat  # noqa: PLC0415 - local import keeps the module light

    violations: list[str] = []
    for path in _closure_files(root):
        if not path.is_file():
            continue
        text = _read(path)
        for finding in scan_compat.scan_source(text, _relative(path, root)):
            if finding["pattern"] in BREAKAGE_PATTERNS:
                violations.append(
                    f"{finding['path']}:{finding['line']}: {finding['pattern']}"
                )
    return violations


# ---------------------------------------------------------------------------
# the guards (xfail until their owner step lands)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.xfail(
    strict=False,
    reason="flip owner: P3.1 - nodes/ classes must be exported by nodes/__init__.py",
)
def test_guard_nodes_classes_are_exported():
    assert nodes_export_violations(REPO_ROOT) == []


@pytest.mark.unit
@pytest.mark.xfail(
    strict=False,
    reason="flip owner: P3.2 - src/ must not contain node schemas",
)
def test_guard_src_has_no_schema():
    assert src_schema_violations(REPO_ROOT) == []


@pytest.mark.unit
@pytest.mark.xfail(
    strict=False,
    reason="flip owner: P4.4 - __init__.py must expose exactly comfy_entrypoint",
)
def test_guard_entrypoint_is_v3_only():
    assert entrypoint_violations(REPO_ROOT) == []


@pytest.mark.unit
def test_guard_no_sys_path_mutation():
    """Guard (d) - flipped to strict in P2.5 (its flip owner)."""
    assert sys_path_violations(REPO_ROOT) == []


@pytest.mark.unit
def test_guard_vendored_tree_has_no_debug_or_pip_leftovers():
    """Guard (e) - flipped to strict in P2.1 (its flip owner)."""
    assert vendored_breakage_violations(REPO_ROOT) == []


# ---------------------------------------------------------------------------
# self-tests: every guard must fail on a deliberately violating copy
# ---------------------------------------------------------------------------
@pytest.fixture
def repo_copy(tmp_path):
    """A throw-away copy of the repository the guards can be run against."""
    destination = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            ".dev",
            ".codegraph",
            ".pytest_cache",
            ".zcode",
            "__pycache__",
            "*.pyc",
        ),
    )
    return destination


@pytest.mark.unit
def test_self_test_guard_a_detects_an_unexported_class(repo_copy):
    nodes_dir = repo_copy / "nodes"
    nodes_dir.mkdir(exist_ok=True)
    (nodes_dir / "__init__.py").write_text(
        "from .keep_model_loader import KEEP_ModelLoaderNode\n"
        "__all__ = ['KEEP_ModelLoaderNode']\n",
        encoding="utf-8",
    )
    (nodes_dir / "keep_model_loader.py").write_text(
        "class KEEP_ModelLoaderNode:\n    pass\n", encoding="utf-8"
    )
    assert nodes_export_violations(repo_copy) == [], (
        "the exported class must not be reported"
    )

    (nodes_dir / "orphan.py").write_text("class OrphanNode:\n    pass\n", encoding="utf-8")
    violations = nodes_export_violations(repo_copy)
    assert any("OrphanNode" in violation for violation in violations), (
        f"the guard did not catch the unexported class: {violations}"
    )


@pytest.mark.unit
def test_self_test_guard_b_detects_a_schema_in_src(repo_copy):
    src_dir = repo_copy / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "keep_image_utils.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert src_schema_violations(repo_copy) == [], "a clean src/ must not be reported"

    (src_dir / "keep_nodes.py").write_text(
        "from comfy_api.latest import io\n\n"
        "SCHEMA = io.Schema(node_id='KEEP_Bad')\n",
        encoding="utf-8",
    )
    violations = src_schema_violations(repo_copy)
    assert any("keep_nodes.py" in violation for violation in violations), (
        f"the guard did not catch the schema in src/: {violations}"
    )


@pytest.mark.unit
def test_self_test_guard_c_detects_a_v1_entrypoint(repo_copy):
    init_file = repo_copy / "__init__.py"

    init_file.write_text(
        "async def comfy_entrypoint():\n    return None\n", encoding="utf-8"
    )
    assert entrypoint_violations(repo_copy) == [], (
        "a pure V3 entry point must not be reported"
    )

    init_file.write_text(
        "NODE_CLASS_MAPPINGS = {}\n"
        "async def comfy_entrypoint():\n    return None\n",
        encoding="utf-8",
    )
    violations = entrypoint_violations(repo_copy)
    assert any("NODE_CLASS_MAPPINGS" in violation for violation in violations), (
        f"the guard did not catch the V1 mapping: {violations}"
    )


@pytest.mark.unit
def test_self_test_guard_d_detects_a_sys_path_mutation(repo_copy):
    utils_file = repo_copy / "modules" / "utils.py"
    text = utils_file.read_text(encoding="utf-8")
    cleaned = "\n".join(
        line for line in text.splitlines() if "sys.path.insert" not in line
    )
    utils_file.write_text(cleaned + "\n", encoding="utf-8")
    assert sys_path_violations(repo_copy) == [], (
        "a package without sys.path mutation must not be reported"
    )

    src_dir = repo_copy / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "evil.py").write_text(
        "import sys\nsys.path.append('/tmp/evil')\n", encoding="utf-8"
    )
    violations = sys_path_violations(repo_copy)
    assert any("evil.py" in violation for violation in violations), (
        f"the guard did not catch the sys.path mutation: {violations}"
    )


@pytest.mark.unit
def test_self_test_guard_e_detects_a_debug_import_in_the_closure(repo_copy):
    """The guard reports a ``pdb`` import in a file that joins the closure.

    The self-test does **not** depend on the current tree still containing the
    known breakage: after P2.1 the baseline is empty and the self-test must
    keep working (it proves the guard, not the state of the tree).
    """
    before = vendored_breakage_violations(repo_copy)

    # A new ``*_arch.py`` file joins the closure through the explicit import
    # list of ``basicsr/archs/__init__.py`` (P2.4 replaced the eager scan).
    arch_file = repo_copy / "modules" / "deps" / "basicsr" / "archs" / "zz_guard_arch.py"
    arch_file.write_text("import pdb\n\n\ndef f():\n    pdb.set_trace()\n", encoding="utf-8")
    archs_init = repo_copy / "modules" / "deps" / "basicsr" / "archs" / "__init__.py"
    init_text = archs_init.read_text(encoding="utf-8")
    assert "from . import (" in init_text, init_text[:200]
    archs_init.write_text(
        init_text.replace("from . import (\n", "from . import (\n    zz_guard_arch,\n", 1),
        encoding="utf-8",
    )

    after = vendored_breakage_violations(repo_copy)
    assert any("zz_guard_arch.py" in violation for violation in after), (
        f"the guard did not catch the new closure member: {after}"
    )
    # The pre-existing findings are unchanged: the guard reports exactly the
    # baseline plus the injected file.
    assert [violation for violation in after if "zz_guard_arch.py" not in violation] == before


@pytest.mark.unit
def test_guard_e_flip_owner_covers_every_current_finding():
    """P2.1 must be able to clear every finding the guard reports.

    Guard (e) is ``xfail(strict=False)``: nothing forces it to flip.  If the
    flip owner's work list misses a file, the guard stays red forever and
    nobody notices.  This test compares the P2.1 section of the plan with the
    live scanner output - the plan's work list and the tree cannot drift.
    """
    plan = REPO_ROOT / "docs" / "plans" / "2026-10-22-comfyui-keep-v3-modernization.md"
    if not plan.is_file():
        pytest.skip("the modernization plan is not part of this checkout")

    text = plan.read_text(encoding="utf-8")
    section = re.search(r"\*\*P2\.1\b.*?(?=\n\*\*P2\.2)", text, re.DOTALL)
    assert section, "the plan must have a P2.1 section"

    for violation in vendored_breakage_violations(REPO_ROOT):
        path = violation.split(":", 1)[0]
        assert Path(path).name in section.group(0), (
            f"{path} is not on the P2.1 work list: guard (e) cannot flip green "
            f"while the file is not cleaned up"
        )
