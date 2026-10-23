"""P1.3 - structure guards.

The guards are **source-text based**: they never import the package, they read
the files.  That keeps them honest while the package is being moved around
(the very thing they guard) and lets them run on a throw-away copy of the
repository.

Five guards, each with the step that flips it from ``xfail`` to strict:

===== ================================================ ==================
guard  what it checks                                  flip owner
===== ================================================ ==================
(a)    every class under ``nodes/*.py`` is exported     **P3.1** (flipped)
       from ``nodes/__init__.py``
(b)    ``src/**`` contains no ``io.Schema(`` /           **P3.2** (flipped)
       ``define_schema``
(c)    the root ``__init__.py`` exposes exactly          **P4.4**
       ``comfy_entrypoint`` (no V1 mapping names)
(d)    no ``sys.path.insert`` / ``sys.path.append``      **P2.5**
       anywhere in the package (the vendored tree
       included - it was exempt only until P2.5)
(e)    no ``pdb`` / ``torch.profiler`` / ``import pip``  **P2.1**
       / ``pip.main`` in the vendored runtime closure
(f)    no module-level ``comfy`` / ``comfy_api`` import  **P3** (flipped)
       in ``src/**`` (the plan's §3 rule: the pack
       imports without a ComfyUI server)
(g)    no ``torch.cuda.empty_cache()`` in the            **P4.6**
       runtime closure (the pack's code - entry point,
       node layer and ``src/`` - **and** the
       vendored tree - the known upstream occurrences
       are allowlisted by file and count)
(h)    the dead video path (``process_video`` /          **P4.7**
       ``VideoReader`` / ``VideoWriter``) is gone
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

#: The production sources of the pack **itself** - the vendored tree
#: (``modules/``) is not part of it.  These are the files ComfyUI executes:
#: the root entry point, the node layer (``nodes/``) and ``src/``.
PACKAGE_OWN_SOURCE_ROOTS = ("__init__.py", "nodes.py", "nodes", "src")

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


def _package_files(root: Path, roots: tuple[str, ...] = PACKAGE_SOURCE_ROOTS) -> list[Path]:
    """Every production python file of the package (the vendored tree included).

    The vendored tree used to be exempt from guard (d): it was a plain folder
    on ``sys.path`` until P2.5 package-ized it.  Since P2.5 it is a real part
    of the package, so the plan's §3 rule ("no ``sys.path`` mutation anywhere
    in the package") applies to it as well - the last vendored mutation was
    removed in the P2 review fix.

    ``roots`` selects the trees to walk: the whole package by default, the
    pack's own code without the vendored tree for the guards that scan the
    vendored part through the import closure instead.
    """
    files: list[Path] = []
    for entry in roots:
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
    """The root ``__init__.py`` must expose exactly ``comfy_entrypoint``.

    The check is **AST based**: a mention of ``comfy_entrypoint`` in a
    docstring is not a definition, and the V1 names are violations whether
    they are assigned or imported.
    """
    init_file = root / "__init__.py"
    if not init_file.is_file():
        return ["__init__.py does not exist"]

    tree = ast.parse(_read(init_file), filename=str(init_file))
    violations: list[str] = []

    has_entrypoint = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "comfy_entrypoint"
        for node in tree.body
    )
    if not has_entrypoint:
        violations.append("__init__.py does not define comfy_entrypoint")

    v1_names = ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS")
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in v1_names:
                    violations.append(f"__init__.py still exports the V1 name {target.id}")
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in v1_names:
                violations.append(f"__init__.py still exports the V1 name {node.target.id}")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in v1_names:
                    violations.append(f"__init__.py still exports the V1 name {alias.name}")
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
# guard (f) - no module-level comfy import in src/
# ---------------------------------------------------------------------------
#: The module roots ComfyUI provides.  Inside ``src/`` they must only be
#: imported **lazily** (function level), so the pack imports without a
#: ComfyUI server (the plan's §3 import rule).
COMFY_MODULE_ROOTS = ("comfy", "comfy_api")


def _module_level_imports(tree: ast.Module):
    """Every import that runs while the module is imported.

    Function and class bodies are **not** module level: an import there is
    lazy by definition.  The body of a module level ``try`` / ``if`` is.
    """
    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # a function / class body runs later
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                yield child
            yield from walk(child)

    yield from walk(tree)


def _imported_roots(node) -> set[str]:
    """The top level module names an import statement pulls in."""
    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        if node.level:  # a relative import - never a comfy import
            return set()
        module = node.module or ""
        return {module.split(".")[0]} if module else set()
    return set()


def comfy_import_violations(root: Path) -> list[str]:
    """``src/**`` must not import ``comfy.*`` at module level."""
    src_dir = root / "src"
    if not src_dir.is_dir():
        return ["src/ does not exist yet (P3.2 creates it)"]

    violations: list[str] = []
    for path in sorted(src_dir.rglob("*.py")):
        tree = ast.parse(_read(path), filename=str(path))
        for node in _module_level_imports(tree):
            roots = _imported_roots(node) & set(COMFY_MODULE_ROOTS)
            if roots:
                violations.append(
                    f"{_relative(path, root)}:{node.lineno}: module-level "
                    f"{sorted(roots)} import"
                )
    return violations


# ---------------------------------------------------------------------------
# guard (g) - no unconditional torch.cuda.empty_cache in the runtime closure
# ---------------------------------------------------------------------------
#: The pattern of guard (g).  ``torch.cuda.empty_cache()`` is unconditional: it
#: releases the cached blocks of **every** device, it fights with the VRAM
#: arbitration of ComfyUI and it crashes without a GPU (the B7 breakage of
#: the plan).  The pack releases the VRAM through
#: ``model_management.soft_empty_cache()``.
EMPTY_CACHE_PATTERN = re.compile(r"torch\.cuda\.empty_cache\s*\(")

#: The known occurrences in the **vendored** BasicSR tree.  They are upstream
#: code in training-only paths (``SRModel.test`` and the ``cpu_cache`` option
#: of the BasicVSR++ arch) that the pack never executes - the pack calls the
#: KEEP arch directly.  The allowlist is **exact** (file *and* count): a file
#: that gains or loses an occurrence is reported, so the guard cannot go
#: stale and a new occurrence is never silently tolerated.
VENDORED_EMPTY_CACHE_ALLOWLIST = {
    "modules/deps/basicsr/archs/basicvsrpp_arch.py": 4,
    "modules/deps/basicsr/models/hifacegan_model.py": 1,
    "modules/deps/basicsr/models/sr_model.py": 1,
    "modules/deps/basicsr/models/video_base_model.py": 1,
    "modules/deps/basicsr/models/video_recurrent_model.py": 1,
}


def _code_text(path: Path) -> str:
    """The source of a file **without** the comments and the string literals.

    The guard greps the **code**: the pack documents the breakage it fixed
    (the pattern is named in the docstrings), and a docstring must not be
    reported as a call.
    """
    import io
    import tokenize

    source = _read(path)
    pieces: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    return "".join(pieces)


def empty_cache_violations(root: Path) -> list[str]:
    """No ``torch.cuda.empty_cache()`` in the surviving runtime closure.

    The scan covers the pack's **own** code - the root entry point, the node
    layer (``nodes/**``) and ``src/**`` - **and** the vendored ``modules/deps``
    closure (not just ``src/``).  The node layer is the code ComfyUI
    actually executes, so it is guarded like ``src/`` (P4 review fix).  The
    vendored occurrences are allowlisted by file and count (see
    ``VENDORED_EMPTY_CACHE_ALLOWLIST``); every other occurrence is a
    violation.
    """
    files: dict[str, Path] = {}
    for path in _closure_files(root):
        if path.is_file():
            files[_relative(path, root)] = path
    for path in _package_files(root, PACKAGE_OWN_SOURCE_ROOTS):
        files[_relative(path, root)] = path

    counts: dict[str, int] = {}
    for relative, path in files.items():
        found = len(EMPTY_CACHE_PATTERN.findall(_code_text(path)))
        if found:
            counts[relative] = found

    violations: list[str] = []
    for relative, found in sorted(counts.items()):
        allowed = VENDORED_EMPTY_CACHE_ALLOWLIST.get(relative, 0)
        if found != allowed:
            violations.append(
                f"{relative}: {found} occurrence(s) of torch.cuda.empty_cache() "
                f"(allowlisted: {allowed})"
            )
    for relative in sorted(set(VENDORED_EMPTY_CACHE_ALLOWLIST) - set(counts)):
        violations.append(
            f"{relative}: the allowlist expects "
            f"{VENDORED_EMPTY_CACHE_ALLOWLIST[relative]} occurrence(s) of "
            f"torch.cuda.empty_cache() but the file has none"
        )
    return violations


# ---------------------------------------------------------------------------
# guard (h) - the dead video path is gone
# ---------------------------------------------------------------------------
#: The symbols of the deleted video path (P4.7).  The vendored ffmpeg shim
#: (``basicsr.utils.video_util``) was deleted in P2.1, so the video path of
#: the processor was dead code - it is gone with P4.7.
DEAD_VIDEO_SYMBOLS = ("process_video", "VideoReader", "VideoWriter")


def dead_video_violations(root: Path) -> list[str]:
    """The dead video path must not appear in the package any more."""
    violations: list[str] = []
    for path in _package_files(root):
        for number, line in enumerate(_read(path).splitlines(), 1):
            for symbol in DEAD_VIDEO_SYMBOLS:
                if symbol in line:
                    violations.append(
                        f"{_relative(path, root)}:{number}: {symbol} ({line.strip()})"
                    )
    return violations


# ---------------------------------------------------------------------------
# the guards (xfail until their owner step lands)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_guard_nodes_classes_are_exported():
    """Guard (a) - flipped to strict in P3.1 (its flip owner)."""
    assert nodes_export_violations(REPO_ROOT) == []


@pytest.mark.unit
def test_guard_src_has_no_schema():
    """Guard (b) - flipped to strict in P3.2 (its flip owner)."""
    assert src_schema_violations(REPO_ROOT) == []


@pytest.mark.unit
def test_guard_entrypoint_is_v3_only():
    """Guard (c) - flipped to strict in P4.4 (its flip owner)."""
    assert entrypoint_violations(REPO_ROOT) == []


@pytest.mark.unit
def test_guard_no_sys_path_mutation():
    """Guard (d) - flipped to strict in P2.5 (its flip owner)."""
    assert sys_path_violations(REPO_ROOT) == []


@pytest.mark.unit
def test_guard_vendored_tree_has_no_debug_or_pip_leftovers():
    """Guard (e) - flipped to strict in P2.1 (its flip owner)."""
    assert vendored_breakage_violations(REPO_ROOT) == []


@pytest.mark.unit
def test_guard_src_has_no_module_level_comfy_import():
    """Guard (f) - flipped to strict in P3 (the plan's §3 import rule)."""
    assert comfy_import_violations(REPO_ROOT) == []


@pytest.mark.unit
def test_guard_no_unconditional_empty_cache_in_the_closure():
    """Guard (g) - flipped to strict in P4.6 (its flip owner)."""
    assert empty_cache_violations(REPO_ROOT) == []


@pytest.mark.unit
def test_guard_the_dead_video_path_is_gone():
    """Guard (h) - flipped to strict in P4.7 (its flip owner)."""
    assert dead_video_violations(REPO_ROOT) == []


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
    # The self-test owns the whole ``nodes/`` folder of the copy: the other
    # node modules of the repository would be reported as unexported (they
    # are not part of this fixture) and the guard would fail for the wrong
    # reason.
    for stale in nodes_dir.glob("*.py"):
        stale.unlink()
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
    utils_file = repo_copy / "src" / "keep_model_configs.py"
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
def test_self_test_guard_f_detects_a_module_level_comfy_import(repo_copy):
    """Guard (f) reports a ``comfy`` import that runs at import time."""
    src_dir = repo_copy / "src"
    src_dir.mkdir(exist_ok=True)
    target = src_dir / "keep_lazy_probe.py"
    target.write_text(
        "from comfy.utils import ProgressBar\n\n"
        "def build():\n"
        "    from comfy import model_management\n"
        "    return ProgressBar, model_management\n",
        encoding="utf-8",
    )
    violations = comfy_import_violations(repo_copy)
    assert any("keep_lazy_probe.py" in violation for violation in violations), (
        f"the guard did not catch the module-level comfy import: {violations}"
    )
    # the function-level import of the same module is **not** a violation
    target.write_text(
        "def build():\n"
        "    from comfy.utils import ProgressBar\n"
        "    return ProgressBar\n",
        encoding="utf-8",
    )
    assert comfy_import_violations(repo_copy) == [], (
        "a lazy comfy import must not be reported"
    )


@pytest.mark.unit
def test_self_test_guard_g_detects_an_empty_cache_call(repo_copy):
    """The guard reports a ``torch.cuda.empty_cache()`` in the pack's code."""
    target = repo_copy / "src" / "keep_processor.py"
    assert empty_cache_violations(repo_copy) == [], (
        "a clean closure must not be reported"
    )

    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _probe():\n    torch.cuda.empty_cache()\n",
        encoding="utf-8",
    )
    violations = empty_cache_violations(repo_copy)
    assert any("keep_processor.py" in violation for violation in violations), (
        f"the guard did not catch the empty_cache call: {violations}"
    )


@pytest.mark.unit
def test_self_test_guard_g_detects_an_empty_cache_call_in_the_node_layer(repo_copy):
    """The guard covers the node layer - the code ComfyUI executes."""
    target = repo_copy / "nodes" / "keep_process_image_sequence.py"
    assert empty_cache_violations(repo_copy) == [], (
        "a clean package must not be reported"
    )

    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _probe():\n    torch.cuda.empty_cache()\n",
        encoding="utf-8",
    )
    violations = empty_cache_violations(repo_copy)
    assert any("keep_process_image_sequence.py" in violation for violation in violations), (
        f"the guard did not catch the empty_cache call in the node layer: {violations}"
    )


@pytest.mark.unit
def test_self_test_guard_g_detects_an_empty_cache_call_in_the_entry_point(repo_copy):
    """The guard covers the root ``__init__.py`` - the entry point."""
    target = repo_copy / "__init__.py"
    assert empty_cache_violations(repo_copy) == [], (
        "a clean package must not be reported"
    )

    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _probe():\n    torch.cuda.empty_cache()\n",
        encoding="utf-8",
    )
    violations = empty_cache_violations(repo_copy)
    assert any("__init__.py" in violation for violation in violations), (
        f"the guard did not catch the empty_cache call in the entry point: {violations}"
    )


@pytest.mark.unit
def test_self_test_guard_g_detects_a_new_vendored_empty_cache_call(repo_copy):
    """The guard covers the vendored closure too - not only ``src/``."""
    vendored = repo_copy / "modules" / "deps" / "basicsr" / "models" / "sr_model.py"
    vendored.write_text(
        vendored.read_text(encoding="utf-8")
        + "\n\ndef _probe():\n    torch.cuda.empty_cache()\n",
        encoding="utf-8",
    )
    violations = empty_cache_violations(repo_copy)
    assert any("sr_model.py" in violation for violation in violations), (
        f"the guard did not catch the new vendored occurrence: {violations}"
    )


@pytest.mark.unit
def test_self_test_guard_g_detects_a_stale_allowlist_entry(repo_copy):
    """The allowlist is exact: a **removed** occurrence is reported, too."""
    vendored = (
        repo_copy / "modules" / "deps" / "basicsr" / "archs" / "basicvsrpp_arch.py"
    )
    vendored.write_text(
        vendored.read_text(encoding="utf-8").replace(
            "torch.cuda.empty_cache()", "pass", 1
        ),
        encoding="utf-8",
    )
    violations = empty_cache_violations(repo_copy)
    assert any("basicvsrpp_arch.py" in violation for violation in violations), (
        f"the guard did not catch the stale allowlist entry: {violations}"
    )


@pytest.mark.unit
def test_self_test_guard_h_detects_the_dead_video_path(repo_copy):
    """The guard reports a reference to the deleted video path."""
    target = repo_copy / "src" / "keep_processor.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _probe():\n    return VideoReader\n",
        encoding="utf-8",
    )
    violations = dead_video_violations(repo_copy)
    assert any("VideoReader" in violation for violation in violations), (
        f"the guard did not catch the dead video symbol: {violations}"
    )


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
