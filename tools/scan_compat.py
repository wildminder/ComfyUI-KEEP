"""P0.3 - vendored-dependency compatibility scanner.

Walks the **runtime import closure** of the current package and greps every
file in it for Python 3.13 / numpy 2 incompatibilities.

Entry points (plan P0.3):

    modules/utils.py
    modules/keep_model_loader.py
    modules/keep_processor.py

The closure is computed with a static import tracer:

* absolute imports of ``basicsr`` / ``facelib`` resolve into ``modules/deps``
  (that is where ``modules/utils.py:16-19`` puts them on ``sys.path``);
* relative imports resolve inside the current package;
* importing a package also imports its parent packages' ``__init__.py``;
* the **eager arch scan** of ``basicsr/archs/__init__.py:13-15`` is a dynamic
  ``importlib.import_module`` loop - it is modelled explicitly, so every
  ``*_arch.py`` in that folder belongs to the closure.

Usage (embedded python):

    C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe tools/scan_compat.py
    C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe tools/scan_compat.py --json out.json
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
    "modules/utils.py",
    "modules/keep_model_loader.py",
    "modules/keep_processor.py",
)

#: Top-level packages that are vendored and therefore resolved inside
#: ``modules/deps`` (they are put on ``sys.path`` by ``modules/utils.py``).
VENDORED_ROOTS = ("basicsr", "facelib")

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
            imports.append((node.level or 0, module.split(".") if module else []))
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
        # the eager arch scan of basicsr/archs/__init__.py
        if current.name == "__init__.py" and current.parent.name == "archs":
            for arch in sorted(current.parent.glob("*_arch.py")):
                if arch.resolve() not in closure:
                    queue.append(arch.resolve())
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
