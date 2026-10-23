"""P0.3 - vendored-dependency compatibility scanner.

Walks the **runtime import closure** of the current package and greps every
file in it for Python 3.13 / numpy 2 incompatibilities.

Entry points (plan P0.3; moved to ``src/`` by P3.2):

    src/keep_model_configs.py
    src/keep_model_loader.py
    src/keep_processor.py

The closure is computed with a static import tracer:

* absolute imports of ``basicsr`` / ``facelib`` resolve into ``modules/deps``
  (the vendored tree lives there since P2.5);
* relative imports resolve inside the current package;
* importing a package also imports its parent packages' ``__init__.py``;
* the **eager folder scans** of the vendored ``basicsr`` packages are dynamic
  ``importlib.import_module`` loops - they are modelled explicitly (see
  :data:`EAGER_SCAN_PACKAGES`), so every matching module of those folders
  belongs to the closure.

Usage (embedded python):

    C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe tools/scan_compat.py
    C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe tools/scan_compat.py --json out.json

Exit codes:

* ``0`` - the scan ran and found **no** incompatible pattern in the closure;
* ``1`` - the scan ran and found **at least one** incompatible pattern
  (``finding_count > 0``).  This is a *result*, not a tool crash: the fix list
  is in ``docs/audit/2026-10-22-compat-scan.md``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPS_DIR = REPO_ROOT / "modules" / "deps"

ENTRY_POINTS = (
    "src/keep_model_configs.py",
    "src/keep_model_loader.py",
    "src/keep_processor.py",
)

#: Top-level packages that are vendored and therefore resolved inside
#: ``modules/deps`` (the vendored tree is part of the node package since
#: P2.5 - it is never put on ``sys.path``).
VENDORED_ROOTS = ("basicsr", "facelib")

#: The vendored ``basicsr`` packages whose ``__init__.py`` scans its own folder
#: at import time and imports every matching module **eagerly** through
#: ``importlib.import_module``.  Each of those modules is therefore part of the
#: runtime closure, even though no static import statement mentions it.
#:
#: package folder -> (file suffix, source of the scan in the vendored tree)
#:
#: * ``archs``  - ``basicsr/archs/__init__.py:13-15``  (``*_arch.py``)
#: * ``data``   - ``basicsr/data/__init__.py:20-22``   (``*_dataset.py``)
#: * ``losses`` - ``basicsr/losses/__init__.py:14-16`` (``*_loss.py``)
#: * ``models`` - ``basicsr/models/__init__.py:13-15`` (``*_model.py``)
#:
#: Omitting one of these scans understates the closure: ``basicsr/__init__.py``
#: imports every one of those packages, so the whole ``models/`` / ``data/`` /
#: ``losses/`` folders are loaded by ``import basicsr``.
EAGER_SCAN_PACKAGES: dict[str, str] = {
    "archs": "_arch.py",
    "data": "_dataset.py",
    "losses": "_loss.py",
    "models": "_model.py",
}

#: (pattern id, human description, regex)
PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "re_import",
        "`from re import ...` - `re.T` etc. were removed in Python 3.11",
        re.compile(r"^\s*from\s+re\s+import\s+\w+", re.MULTILINE),
    ),
    (
        "np_float",
        "np.float - removed in numpy 1.24",
        re.compile(r"\bnp\.float\b"),
    ),
    (
        "np_int",
        "np.int - removed in numpy 1.24",
        re.compile(r"\bnp\.int\b"),
    ),
    (
        "np_bool",
        "np.bool - removed in numpy 1.24",
        re.compile(r"\bnp\.bool\b"),
    ),
    (
        "np_object",
        "np.object - removed in numpy 1.24",
        re.compile(r"\bnp\.object\b"),
    ),
    (
        "np_str",
        "np.str - removed in numpy 1.24",
        re.compile(r"\bnp\.str\b"),
    ),
    (
        "np_nan",
        "np.NaN - removed in numpy 2.0",
        re.compile(r"\bnp\.NaN\b"),
    ),
    (
        "distutils",
        "distutils - removed from the stdlib in Python 3.12",
        re.compile(r"\bdistutils\b"),
    ),
    (
        "import_imp",
        "`import imp` - removed in Python 3.12",
        re.compile(r"^\s*import\s+imp\b", re.MULTILINE),
    ),
    (
        "torch_cuda_amp",
        "torch.cuda.amp - deprecated in torch 2.4+",
        re.compile(r"\btorch\.cuda\.amp\b"),
    ),
    (
        "pdb",
        "pdb - debug break left in the runtime closure",
        re.compile(r"^\s*(import\s+pdb\b|pdb\.\w+)", re.MULTILINE),
    ),
    (
        "torch_profiler",
        "torch.profiler - heavyweight debug import",
        re.compile(r"\btorch\.profiler\b"),
    ),
    (
        "collections_mapping",
        "`from collections import ...Mapping` - moved to collections.abc in 3.10",
        re.compile(r"^\s*from\s+collections\s+import\s+.*\bMapping\b", re.MULTILINE),
    ),
    (
        "import_pip",
        "`import pip` - no pip execution inside the package",
        re.compile(r"^\s*import\s+pip\b", re.MULTILINE),
    ),
    (
        "pip_main",
        "`pip.main(...)` - pip execution inside the package",
        re.compile(r"\bpip\.main\s*\("),
    ),
)


def scan_source(text: str, path: str = "<memory>") -> list[dict]:
    """Return every finding of the known patterns in ``text``."""
    findings: list[dict] = []
    lines = text.splitlines()
    for pattern_id, description, regex in PATTERNS:
        for match in regex.finditer(text):
            line_number = text.count("\n", 0, match.start()) + 1
            snippet = lines[line_number - 1].strip() if line_number <= len(lines) else ""
            findings.append(
                {
                    "path": path,
                    "line": line_number,
                    "pattern": pattern_id,
                    "description": description,
                    "snippet": snippet[:160],
                }
            )
    findings.sort(key=lambda f: (f["path"], f["line"], f["pattern"]))
    return findings


def _module_candidates(base: Path, parts: list[str]) -> list[Path]:
    """Candidate files for the module ``parts`` below ``base``."""
    if not parts:
        return []
    module_path = base.joinpath(*parts)
    candidates = [module_path.with_suffix(".py")]
    package_init = module_path / "__init__.py"
    if package_init.is_file():
        candidates.append(package_init)
    return candidates


def _package_inits(base: Path, parts: list[str]) -> list[Path]:
    """``__init__.py`` of every parent package of ``parts`` below ``base``."""
    inits = []
    for depth in range(1, len(parts)):
        parent = base.joinpath(*parts[:depth]) / "__init__.py"
        if parent.is_file():
            inits.append(parent)
    return inits


def _imported_module_names(tree: ast.AST, current_file: Path) -> list[tuple[int, list[str]]]:
    """(level, module parts) for every import statement of ``tree``."""
    imports: list[tuple[int, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((0, alias.name.split(".")))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            base = module.split(".") if module else []
            imports.append((node.level or 0, base))
            # ``from <package> import <submodule>`` loads the submodule file as
            # well (e.g. ``from basicsr.models import lr_scheduler`` in
            # ``basicsr/models/base_model.py:8``).  The resolver drops the
            # candidates that are not files, so plain symbol imports cost
            # nothing.
            for alias in node.names:
                if alias.name != "*":
                    imports.append((node.level or 0, base + [alias.name]))
    return imports


def _resolve(level: int, parts: list[str], current_file: Path) -> list[Path]:
    """Resolve one import statement to the files it pulls in."""
    if level:
        # relative import: walk up from the current package directory
        package_dir = current_file.parent
        for _ in range(level - 1):
            package_dir = package_dir.parent
        return [
            candidate
            for candidate in _module_candidates(package_dir, parts)
            if candidate.is_file()
        ] + [
            parent
            for parent in _package_inits(package_dir, parts + ["__dummy__"])
            if parent.is_file()
        ]
    if not parts:
        return []
    if parts[0] not in VENDORED_ROOTS:
        return []  # third-party / ComfyUI: not our closure
    return [
        candidate
        for candidate in _module_candidates(DEPS_DIR, parts)
        if candidate.is_file()
    ] + [parent for parent in _package_inits(DEPS_DIR, parts) if parent.is_file()]


def _has_dynamic_folder_scan(init_file: Path, suffix: str) -> bool:
    """True if the package ``__init__.py`` really scans its own folder.

    The vendored ``basicsr`` packages used to import every matching module of
    their folder at import time (a dynamic ``importlib.import_module`` loop).
    P2.4 replaced those loops with explicit import lists, so the closure
    computation must only model the scan when it is still there.
    """
    try:
        text = init_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return "importlib.import_module" in text and suffix in text


def find_import_closure(entry_points: list[Path]) -> list[Path]:
    """Static runtime import closure of ``entry_points`` (sorted, de-duplicated)."""
    closure: set[Path] = set()
    queue: list[Path] = [p.resolve() for p in entry_points]
    while queue:
        current = queue.pop()
        if current in closure or not current.is_file():
            continue
        closure.add(current)
        try:
            tree = ast.parse(current.read_text(encoding="utf-8"), filename=str(current))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for level, parts in _imported_module_names(tree, current):
            for dependency in _resolve(level, parts, current):
                if dependency.resolve() not in closure:
                    queue.append(dependency.resolve())
        # the eager folder scans of basicsr/archs|data|losses|models/__init__.py
        # (P2.4: only modelled when the package init really contains such a
        # scan - the explicit import lists have none)
        if current.name == "__init__.py":
            suffix = EAGER_SCAN_PACKAGES.get(current.parent.name)
            if suffix and _has_dynamic_folder_scan(current, suffix):
                for module in sorted(current.parent.glob(f"*{suffix}")):
                    if module.resolve() not in closure:
                        queue.append(module.resolve())
    return sorted(closure, key=lambda p: str(p).lower())


def scan_paths(paths: list[Path], repo_root: Path = REPO_ROOT) -> list[dict]:
    findings: list[dict] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            relative = str(path)
        findings.extend(scan_source(text, relative))
    findings.sort(key=lambda f: (f["path"], f["line"], f["pattern"]))
    return findings


def scan_repo(repo_root: Path = REPO_ROOT) -> dict:
    entry_points = [repo_root / rel for rel in ENTRY_POINTS]
    closure = find_import_closure(entry_points)
    findings = scan_paths(closure, repo_root)
    by_pattern: dict[str, int] = {}
    for finding in findings:
        by_pattern[finding["pattern"]] = by_pattern.get(finding["pattern"], 0) + 1
    return {
        "entry_points": list(ENTRY_POINTS),
        "closure_file_count": len(closure),
        "closure_files": [
            p.resolve().relative_to(repo_root.resolve()).as_posix() for p in closure
        ],
        "finding_count": len(findings),
        "findings_by_pattern": by_pattern,
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = scan_repo()
    text = json.dumps(report, indent=2)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")
    if args.quiet:
        print(json.dumps({k: v for k, v in report.items() if k != "findings"}, indent=2))
    else:
        print(text)
    return 1 if report["finding_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
