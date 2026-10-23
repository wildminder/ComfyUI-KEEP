"""P2 - vendored dependency repair (plan sections P2.1 - P2.7).

The tests in this file are **execution based**: they import the vendored tree
and assert on what really happens, not on a recorded string.

Tiers:

* ``unit``        - the vendored tree / the package wiring;
* ``mock_integration`` - the package under the root conftest's ComfyUI mocks.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPS_DIR = REPO_ROOT / "modules" / "deps"
BASICSR_UTILS = DEPS_DIR / "basicsr" / "utils"

#: The vendored tree must not contain these packages any more (P2.6): they
#: were phantom entries of ``requirements.txt``.
PHANTOM_PACKAGES = ("addict", "lmdb", "yapf")


def _run_python(code: str, *args: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter (the embedded one)."""
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )


# ---------------------------------------------------------------------------
# P2.1 - no pip, no debug leftovers
# ---------------------------------------------------------------------------
_PIP_BLOCKED_PROBE = """
import importlib, json, sys
from pathlib import Path

class _BlockPip:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pip" or fullname.startswith("pip."):
            raise AssertionError("pip was imported: " + fullname)
        return None

sys.meta_path.insert(0, _BlockPip())

root = Path(sys.argv[1])
utils_dir = root / "modules" / "deps" / "basicsr" / "utils"
sys.path.insert(0, str(root))
import conftest  # registers the package alias and the ComfyUI mocks

result = {"pip_imported": False, "modules": {}, "errors": []}
for path in sorted(utils_dir.glob("*.py")):
    if path.name == "__init__.py":
        continue
    name = "ComfyUI_KEEP.modules.deps.basicsr.utils." + path.stem
    try:
        importlib.import_module(name)
        result["modules"][name] = "ok"
    except BaseException as exc:  # noqa: BLE001 - the probe records everything
        result["modules"][name] = type(exc).__name__ + ": " + str(exc)[:200]
        result["errors"].append(name)
print(json.dumps(result))
"""


@pytest.mark.unit
def test_importing_the_vendored_utils_never_calls_pip():
    """P2.1 (b): the vendored ``basicsr/utils`` walk performs **no** pip call.

    The probe blocks the ``pip`` module outright: any attempt to import it
    raises ``AssertionError`` and fails the test.  The vendored
    ``video_util.py`` (the import-time pip shim) is gone, so the walk either
    imports cleanly or fails for the *known* reason only - the diffusers
    import of the eager arch scan (P2.2 removes that one as well).
    """
    result = _run_python(_PIP_BLOCKED_PROBE, str(REPO_ROOT))
    assert result.returncode == 0, (
        f"the probe itself failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["pip_imported"] is False, report
    missing = set()
    for name, outcome in report["modules"].items():
        if outcome == "ok":
            continue
        # the only tolerated failures are
        #   * the diffusers import of the eager arch scan (B2, removed in P2.2)
        #   * the phantom packages of P2.6 (removed from the tree in P2.4)
        if "diffusers" in outcome or "cached_download" in outcome:
            continue
        if outcome.startswith("ModuleNotFoundError: No module named "):
            match = re.search(r"No module named '([^']+)'", outcome)
            assert match, outcome
            missing.add(match.group(1).split(".")[0])
            continue
        raise AssertionError(f"{name} failed unexpectedly: {outcome}")
    assert missing <= set(PHANTOM_PACKAGES), (
        f"the vendored tree needs packages outside the trimmed set: {sorted(missing)}"
    )
    assert not (BASICSR_UTILS / "video_util.py").exists(), (
        "the import-time pip shim module must be deleted (P2.1)"
    )


@pytest.mark.unit
def test_the_next_failure_after_p21_is_the_diffusers_import():
    """P2.1 (c): the eager arch scan now fails **exactly** at the diffusers import.

    The plan's documented progression: B1 (``from re import T``) is gone, so
    B2 (the diffusers import) is the next failure.
    """
    code = (
        "import importlib, sys\n"
        f"sys.path.insert(0, {str(DEPS_DIR)!r})\n"
        "try:\n"
        "    importlib.import_module('basicsr.archs.keep_arch')\n"
        "    print('IMPORTED')\n"
        "except BaseException as exc:\n"
        "    import traceback\n"
        "    print(type(exc).__name__)\n"
        "    print(str(exc)[:300])\n"
        "    print('TB_HAS_DIFFUSERS', 'diffusers' in traceback.format_exc())\n"
        "    print('TB_HAS_RE_T', \"cannot import name 'T' from 're'\" in traceback.format_exc())\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    if "IMPORTED" in out:
        # P2.2 vendored the diffusers symbols: the arch imports cleanly now
        return
    # P2.1 state: the failure is exactly the diffusers import (B2)
    assert "ImportError" in out, out
    assert "cached_download" in out, out
    assert "TB_HAS_DIFFUSERS True" in out, out
    assert "TB_HAS_RE_T False" in out, out


@pytest.mark.unit
def test_no_debug_leftovers_in_the_vendored_closure():
    """P2.1 (a): the grep guard of P1.3-e is green over the whole closure."""
    from tools import scan_compat  # noqa: PLC0415 - local import keeps the module light

    closure = scan_compat.find_import_closure(
        [REPO_ROOT / rel for rel in scan_compat.ENTRY_POINTS]
    )
    assert closure, "the closure must not be empty"
    findings = scan_compat.scan_paths(closure, REPO_ROOT)
    breakage = [f for f in findings if f["pattern"] in ("pdb", "torch_profiler", "import_pip", "pip_main")]
    assert breakage == [], breakage


# ---------------------------------------------------------------------------
# P2.2 - the vendored diffusers classes
# ---------------------------------------------------------------------------
COMPAT_FILE = DEPS_DIR / "basicsr" / "archs" / "_diffusers_compat.py"
EQUIVALENCE_DOC = REPO_ROOT / "docs" / "audit" / "2026-10-22-diffusers-equivalence.md"


def _load_compat_module():
    """Load ``_diffusers_compat.py`` directly (without the basicsr import)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("keep_diffusers_compat", COMPAT_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_the_vendored_feedforward_preserves_the_shape():
    """P2.2 (b): ``FeedForward(8)(zeros)`` preserves the shape."""
    import torch

    compat = _load_compat_module()
    feed_forward = compat.FeedForward(8)
    sample = torch.zeros((2, 8))
    with torch.no_grad():
        output = feed_forward(sample)
    assert tuple(output.shape) == (2, 8), output.shape
    assert output.dtype == torch.float32, output.dtype


@pytest.mark.unit
def test_the_vendored_ada_layer_norm_has_layer_norm_semantics():
    """P2.2 (b): the inner norm is a LayerNorm (mean 0 / std 1).

    The plan's wording ("AdaLayerNorm(8, num_embeds_ada_norm=4) returns
    mean~0/std~1") describes the *LayerNorm* part: the affine output of
    diffusers 0.11.0 is ``norm(x) * (1 + scale) + shift`` and is therefore
    **not** mean 0 / std 1 by construction.  The real 0.11.0 signature is
    ``AdaLayerNorm(embedding_dim, num_embeddings)`` - the keyword
    ``num_embeds_ada_norm`` does not exist in 0.11.0.
    """
    import torch

    compat = _load_compat_module()
    layer_norm = compat.AdaLayerNorm(8, 4)

    # the real 0.11.0 signature
    assert layer_norm.emb.num_embeddings == 4
    assert layer_norm.emb.embedding_dim == 8
    assert layer_norm.linear.in_features == 8
    assert layer_norm.linear.out_features == 16
    assert layer_norm.norm.normalized_shape == (8,)
    assert layer_norm.norm.elementwise_affine is False

    sample = torch.randn((3, 8), generator=torch.Generator().manual_seed(3))
    normalized = layer_norm.norm(sample)
    # LayerNorm normalises every row over the feature dimension
    assert torch.allclose(normalized.mean(dim=1), torch.zeros(3), atol=1e-6), (
        normalized.mean(dim=1)
    )
    assert torch.allclose(
        normalized.var(dim=1, unbiased=False), torch.ones(3), atol=1e-3
    ), normalized.var(dim=1, unbiased=False)

    # the timestep embedding follows the upstream formula exactly
    timestep = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        emb = layer_norm.linear(layer_norm.silu(layer_norm.emb(timestep)))
    assert tuple(emb.shape) == (3, 16), emb.shape

    # the upstream class chunks the embedding on dim 0, so the forward does
    # not broadcast for a (batch, channels) input - the vendored copy
    # reproduces that behaviour byte-for-byte (see the equivalence record).
    with pytest.raises(RuntimeError):
        layer_norm(sample, timestep)


@pytest.mark.unit
def test_no_diffusers_import_remains_in_the_vendored_tree():
    """P2.2 (c): the vendored tree must not import diffusers any more."""
    import ast

    offenders = []
    for path in sorted(DEPS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] == "diffusers":
                    offenders.append(f"{path}:{node.lineno} {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "diffusers":
                        offenders.append(f"{path}:{node.lineno} {alias.name}")
    assert offenders == [], offenders


@pytest.mark.unit
def test_the_diffusers_equivalence_run_is_recorded():
    """P2.2 (a): the one-off equivalence run must be recorded.

    The recorded run is re-executed when the pristine diffusers 0.11.0
    source is available (``KEEP_DIFFUSERS_SRC``); otherwise the recorded
    document is checked and the re-run is skipped with a message that names
    the environment variable.
    """
    assert EQUIVALENCE_DOC.is_file(), f"missing equivalence record: {EQUIVALENCE_DOC}"
    text = EQUIVALENCE_DOC.read_text(encoding="utf-8")
    assert "diffusers 0.11.0" in text, text[:200]
    assert "byte-identical" in text, text[:200]
    assert "overall equivalent: **True**" in text, text[:200]

    source = os.environ.get("KEEP_DIFFUSERS_SRC", "").strip()
    if not source:
        pytest.skip(
            "KEEP_DIFFUSERS_SRC is not set - the pristine diffusers 0.11.0 "
            "source is not available, so the recorded equivalence run cannot "
            "be repeated (the recorded evidence is checked above)"
        )
    result = _run_python(
        "import sys; from tools.diffusers_equivalence import run; "
        "r = run(__import__('pathlib').Path(sys.argv[1])); "
        "print('EQUIVALENT', r['equivalent'])",
        source,
    )
    assert result.returncode == 0, result.stderr
    assert "EQUIVALENT True" in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# P2.4 - the pruned tree and the explicit import lists
# ---------------------------------------------------------------------------

#: Blocks every ``pip`` import - no import of the vendored tree may install
#: anything (the deleted pip shim of P2.1 must not come back).
_PIP_BLOCK = """
import sys

class _BlockPip:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pip" or fullname.startswith("pip."):
            raise AssertionError("pip was imported: " + fullname)
        return None

sys.meta_path.insert(0, _BlockPip())
"""


def _scan_report():
    """The P0.3 scanner report of the repository."""
    tools_dir = str(REPO_ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import scan_compat  # noqa: PLC0415 - the tool is a script, not a package

    return scan_compat.scan_repo(REPO_ROOT)


@pytest.mark.unit
def test_the_closure_equals_the_pruned_file_set():
    """P2.4: the closure re-computation equals the pruned file set.

    Missing-free (every file of the closure exists) and orphan-free (every
    file of the vendored tree is part of the closure).
    """
    report = _scan_report()
    closure = {
        rel for rel in report["closure_files"] if rel.startswith("modules/deps/")
    }
    tree = {path.relative_to(REPO_ROOT).as_posix() for path in DEPS_DIR.rglob("*.py")}
    assert closure, "the runtime closure is empty"
    assert closure == tree, (
        "the closure and the pruned tree must be identical; "
        f"missing: {sorted(tree - closure)}; orphans: {sorted(closure - tree)}"
    )


@pytest.mark.unit
def test_no_dynamic_folder_scan_remains_in_the_vendored_tree():
    """P2.4: the eager ``importlib`` folder scans are gone."""
    offenders = []
    for path in sorted(DEPS_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "importlib.import_module" in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == [], offenders


@pytest.mark.unit
def test_import_basicsr_still_succeeds():
    """P2.4/P2.5: the vendored ``basicsr`` package still imports.

    This is the guard against the silent breakage path: a deleted package
    directory raises ``ModuleNotFoundError`` inside ``import basicsr``, which
    ``modules/utils.py`` swallows into the ``DummyArchRegistry`` - the node
    would keep loading and silently lose the KEEP architecture.

    Since P2.5 the package is imported relatively (it is part of the node
    package), so it is never a top-level ``basicsr`` module.
    """
    import importlib

    module = importlib.import_module("ComfyUI_KEEP.modules.deps.basicsr")
    assert module.__name__ == "ComfyUI_KEEP.modules.deps.basicsr"
    for name in ("archs", "models", "data", "losses", "metrics"):
        assert importlib.import_module(
            f"ComfyUI_KEEP.modules.deps.basicsr.{name}"
        ) is not None
    assert "basicsr" not in sys.modules



@pytest.mark.unit
def test_the_keep_architecture_is_still_registered():
    """P2.4/P2.5: the explicit arch list still registers the KEEP architecture."""
    import importlib

    importlib.import_module("ComfyUI_KEEP.modules.deps.basicsr.archs")
    registry = importlib.import_module(
        "ComfyUI_KEEP.modules.deps.basicsr.utils.registry"
    )
    assert registry.ARCH_REGISTRY.get("KEEP").__name__ == "KEEP"


# ---------------------------------------------------------------------------
# P2.5 - the vendored tree is a package of the node package
# ---------------------------------------------------------------------------

#: The vendored packages inside the node package.
VENDORED = "ComfyUI_KEEP.modules.deps"

#: P2.5 (a): every module of ``modules/deps`` imports - with pip blocked and
#: without touching the network (no import of the vendored tree downloads
#: anything; the downloads happen in explicit calls).
_WALK_PROBE = """
import importlib, json, pkgutil, sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
import conftest  # registers the package alias and the ComfyUI mocks

import ComfyUI_KEEP.modules.deps as deps

failures = []
for info in pkgutil.walk_packages(deps.__path__, deps.__name__ + "."):
    try:
        importlib.import_module(info.name)
    except BaseException as exc:
        failures.append({
            "module": info.name,
            "type": type(exc).__name__,
            "error": str(exc)[:200],
        })
print("KEEP_WALK_FAILURES:" + json.dumps(failures))
"""


@pytest.mark.unit
def test_every_vendored_module_imports_without_pip():
    """P2.5 (a): the ``walk_packages`` import test over ``modules/deps``."""
    result = _run_python(_PIP_BLOCK + _WALK_PROBE, str(REPO_ROOT))
    assert result.returncode == 0, result.stdout + result.stderr
    sentinel = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("KEEP_WALK_FAILURES:")
    ]
    assert sentinel, (
        f"the walk probe produced no result\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    failures = json.loads(sentinel[-1].split(":", 1)[1])
    assert failures == [], f"vendored modules failed to import: {failures}"


@pytest.mark.unit
def test_the_vendored_packages_never_enter_sys_modules():
    """P2.5 (b): ``basicsr`` / ``facelib`` are not top-level modules."""
    import importlib

    importlib.import_module(f"{VENDORED}.basicsr")
    importlib.import_module(f"{VENDORED}.facelib")
    assert "basicsr" not in sys.modules
    assert "basicsr.utils" not in sys.modules
    assert "facelib" not in sys.modules
    assert "facelib.utils" not in sys.modules


@pytest.mark.unit
def test_the_processor_and_loader_use_the_vendored_classes():
    """P2.5 (c): the real vendored symbols - not stubs."""
    import importlib

    processor = importlib.import_module("ComfyUI_KEEP.modules.keep_processor")
    loader = importlib.import_module("ComfyUI_KEEP.modules.keep_model_loader")
    assert processor.img2tensor.__module__.startswith(f"{VENDORED}.basicsr")
    assert processor.tensor2img.__module__.startswith(f"{VENDORED}.basicsr")
    assert processor.is_gray.__module__.startswith(f"{VENDORED}.facelib")
    assert loader.FaceRestoreHelper.__module__.startswith(f"{VENDORED}.facelib")


@pytest.mark.unit
def test_importing_the_package_does_not_change_sys_path():
    """P2.5 (d): ``sys.path`` before/after the import is byte-identical."""
    code = (
        "import sys\n"
        "root = sys.argv[1]\n"
        "sys.path.insert(0, root)\n"
        "import conftest\n"
        "before = list(sys.path)\n"
        "import ComfyUI_KEEP.modules.utils\n"
        "import ComfyUI_KEEP.modules.keep_processor\n"
        "import ComfyUI_KEEP.modules.keep_model_loader\n"
        "print('KEEP_SYS_PATH_SAME', before == sys.path)\n"
    )
    result = _run_python(code, str(REPO_ROOT))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KEEP_SYS_PATH_SAME True" in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# P2.6 - requirements.txt is the true runtime set
# ---------------------------------------------------------------------------

#: The third-party modules the closure imports -> their distribution name.
_DISTRIBUTIONS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "scipy": "scipy",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "einops": "einops",
    "tqdm": "tqdm",
    "yaml": "pyyaml",
    "requests": "requests",
}

#: The dependency set of the pruned closure (plan P2.6 + the two packages the
#: plan's list missed - see ``docs/audit/2026-10-23-p2-vendored-repair.md``).
_EXPECTED_REQUIREMENTS = {
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "Pillow",
    "einops",
    "scipy",
    "tqdm",
    "pyyaml",
    "requests",
}


def _requirement_entries():
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _closure_imports():
    """The third-party module names imported by the runtime closure."""
    import ast

    report = _scan_report()
    names = set()
    for rel in report["closure_files"]:
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


@pytest.mark.unit
def test_the_requirements_are_the_true_runtime_set():
    """P2.6: ``requirements.txt`` lists exactly the runtime dependencies."""
    entries = _requirement_entries()
    assert entries == _EXPECTED_REQUIREMENTS, sorted(entries)
    for phantom in PHANTOM_PACKAGES:
        assert phantom not in entries


@pytest.mark.unit
def test_no_phantom_package_is_imported_in_the_closure():
    """P2.6: ``addict`` / ``lmdb`` / ``yapf`` are imported nowhere."""
    imported = _closure_imports()
    offenders = sorted(name for name in imported if name in PHANTOM_PACKAGES)
    assert offenders == [], offenders


@pytest.mark.unit
def test_every_third_party_import_of_the_closure_is_declared():
    """P2.6: every third-party import of the closure is a declared dependency."""
    imported = _closure_imports()
    declared = {
        _DISTRIBUTIONS[name] for name in imported if name in _DISTRIBUTIONS
    }
    entries = _requirement_entries()
    assert declared == entries, (
        f"the closure imports {sorted(declared)}, requirements.txt declares "
        f"{sorted(entries)}"
    )
