"""P2 - vendored dependency repair (plan sections P2.1 - P2.7).

The tests in this file are **execution based**: they import the vendored tree
and assert on what really happens, not on a recorded string.

Tiers:

* ``unit``        - the vendored tree / the package wiring;
* ``mock_integration`` - the package under the root conftest's ComfyUI mocks.
"""

from __future__ import annotations

import ast
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

from src.keep_vendored import vendor  # the pack-private namespace (Round 21)

result = {"pip_imported": False, "modules": {}, "errors": []}
for path in sorted(utils_dir.glob("*.py")):
    if path.name == "__init__.py":
        continue
    target = "basicsr.utils." + path.stem
    try:
        module = vendor(target)
        result["modules"][module.__name__] = "ok"
    except BaseException as exc:  # noqa: BLE001 - the probe records everything
        result["modules"][target] = type(exc).__name__ + ": " + str(exc)[:200]
        result["errors"].append(target)
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


#: The probe of the P2.1 (c) progression.  The vendored architecture is
#: imported through the **packaged** layout (P2.5): the top-level ``basicsr``
#: of ``modules/deps`` does not exist any more, so a probe that put
#: ``modules/deps`` on ``sys.path`` would exercise a *second* copy of the
#: vendored tree instead of the shipped one.
_ARCH_PROBE = """
import importlib, json, sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
import conftest  # registers the package alias and the ComfyUI mocks

from src.keep_vendored import vendor  # the pack-private namespace (Round 21)

report = {}
try:
    module = vendor("basicsr.archs.keep_arch")
except BaseException as exc:  # noqa: BLE001 - the probe records everything
    import traceback
    report["error"] = type(exc).__name__ + ": " + str(exc)[:300]
    report["traceback"] = traceback.format_exc()
else:
    report["module"] = module.__name__
    report["feedforward"] = module.FeedForward.__module__
    report["ada_layer_norm"] = module.AdaLayerNorm.__module__
report["basicsr_in_sys_modules"] = "basicsr" in sys.modules
print(json.dumps(report))
"""


@pytest.mark.unit
def test_the_next_failure_after_p21_is_resolved_by_the_vendored_diffusers():
    """P2.1 (c): the documented next-failure progression is **complete**.

    The plan's progression: B1 (``from re import T``) is gone, so B2 (the
    diffusers import) had to be the next failure.  P2.2 vendored the two
    diffusers classes, so the architecture imports now - the test has to
    *prove* that instead of returning early (a test that returns before its
    assertions verifies nothing).
    """
    result = _run_python(_ARCH_PROBE, str(REPO_ROOT))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert "error" not in report, report
    # the architecture imports through the pack-private namespace
    assert report["module"] == (
        "_keep_vendor.basicsr.archs.keep_arch"
    ), report
    # B2 is resolved by the vendored copy, not by a hidden fallback
    assert report["feedforward"] == (
        "_keep_vendor.basicsr.archs._diffusers_compat"
    ), report
    assert report["ada_layer_norm"] == (
        "_keep_vendor.basicsr.archs._diffusers_compat"
    ), report
    # no top-level ``basicsr`` shadows an installed one (P2.5)
    assert report["basicsr_in_sys_modules"] is False, report


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
EQUIVALENCE_DOC = REPO_ROOT / ".dev" / "docs" / "audit" / "2026-10-22-diffusers-equivalence.md"


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
@pytest.mark.skipif(
    not EQUIVALENCE_DOC.exists(),
    reason=f"the audit artifact {EQUIVALENCE_DOC.name} is not in the working tree",
)
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

    2026-11-03, keep-fix5-implement1 Phase 4.1: two folders stay on disk
    (byte identical, the plan does not list them for deletion) while the
    regenerated ``basicsr/__init__.py`` no longer imports them - and their
    remaining files import deleted utils (``metrics/``) or were only ever
    imported by the deleted ``stylegan2_arch`` (``ops/``).  They are
    excluded from **both** sides of the equality (and must not be in the
    closure).
    """
    report = _scan_report()
    closure = {
        rel for rel in report["closure_files"] if rel.startswith("modules/deps/")
    }
    tree = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in DEPS_DIR.rglob("*.py")
        if "/basicsr/metrics/" not in path.as_posix()
        and "/basicsr/ops/" not in path.as_posix()
    }
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

    Since P2.5 the package is part of the node package; since Round 21 it
    loads under the pack-private namespace (``src/keep_vendored.py``), so
    the import goes through the loader and the assertion also pins the
    private name itself - it is never a top-level ``basicsr`` module.

    2026-11-03, keep-fix5-implement1 Phase 4.1: ``models``/``data``/
    ``losses`` are deleted; ``metrics`` and ``ops`` stay on disk but off the
    import path (the regenerated package init imports only ``archs``,
    ``utils`` and ``version``), so exactly those two subpackages are
    asserted.
    """
    from src.keep_vendored import vendor

    module = vendor("basicsr")
    assert module.__name__ == "_keep_vendor.basicsr"
    for name in ("archs", "utils"):
        assert vendor(f"basicsr.{name}") is not None
    assert "basicsr" not in sys.modules



@pytest.mark.unit
def test_the_keep_architecture_is_still_registered():
    """P2.4/P2.5: the explicit arch list still registers the KEEP architecture."""
    from src.keep_vendored import vendor

    vendor("basicsr.archs")  # the explicit arch list runs the registrations
    registry = vendor("basicsr.utils.registry")
    assert registry.ARCH_REGISTRY.get("KEEP").__name__ == "KEEP"


# ---------------------------------------------------------------------------
# P2.5 - the vendored tree is a package of the node package
# ---------------------------------------------------------------------------

#: The canonical name of the vendored packages - the pack-private namespace
#: (Round 21).  It does not derive from the name the host registers the pack
#: under, so it is the same in every import mode (the V3 package mode, the
#: pytest alias mode, the headless alias mode) and the tree instantiates
#: exactly once per process.
VENDORED = "_keep_vendor"

#: P2.5 (a): every module of ``modules/deps`` imports - with pip blocked and
#: without touching the network (no import of the vendored tree downloads
#: anything; the downloads happen in explicit calls).
_WALK_PROBE = """
import importlib, json, pkgutil, sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
import conftest  # registers the package alias and the ComfyUI mocks

# Round 21: the walk runs over the pack-private namespace - the loader
# anchors the private root to the pack's own ``modules/deps`` files.
from src.keep_vendored import _ensure_vendor_root

_ensure_vendor_root()
deps = sys.modules["_keep_vendor"]

failures = []
for info in pkgutil.walk_packages(deps.__path__, deps.__name__ + "."):
    # keep-fix5-implement1 Phase 4.1 (2026-11-03): basicsr.metrics is off the
    # import path - the package init no longer imports it and its files
    # import deleted utils (matlab_functions, color_util); the folder stays
    # on disk, so the walk skips it deliberately.
    if info.name.startswith("_keep_vendor.basicsr.metrics"):
        continue
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
    """P2.5 (a): the ``walk_packages`` import test over ``modules/deps``.

    2026-11-03, keep-fix5-implement1 Phase 4.1: ``basicsr.metrics`` is
    skipped (see the probe comment) - the plan deletes the utils its files
    import, and the regenerated ``basicsr/__init__.py`` no longer imports
    the package.
    """
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
    """P2.5 (b): ``basicsr`` / ``facelib`` are not top-level modules.

    Since Round 21 the packages load under the pack-private namespace: the
    import goes through the loader, so the private name (and only it) lands
    in ``sys.modules``.
    """
    from src.keep_vendored import vendor

    assert vendor("basicsr").__name__ == f"{VENDORED}.basicsr"
    assert vendor("facelib").__name__ == f"{VENDORED}.facelib"
    assert "basicsr" not in sys.modules
    assert "basicsr.utils" not in sys.modules
    assert "facelib" not in sys.modules
    assert "facelib.utils" not in sys.modules


@pytest.mark.unit
def test_the_processor_and_loader_use_the_vendored_classes():
    """P2.5 (c): the real vendored symbols - not stubs."""
    import importlib

    processor = importlib.import_module("ComfyUI_KEEP.src.keep_processor")
    loader = importlib.import_module("ComfyUI_KEEP.src.keep_model_loader")
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
        "import ComfyUI_KEEP.src.keep_model_configs\n"
        "import ComfyUI_KEEP.src.keep_processor\n"
        "import ComfyUI_KEEP.src.keep_model_loader\n"
        "print('KEEP_SYS_PATH_SAME', before == sys.path)\n"
    )
    result = _run_python(code, str(REPO_ROOT))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KEEP_SYS_PATH_SAME True" in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# P2.6 -> the host-guarantee policy: requirements.txt is the set the host
# does **not** provide
# ---------------------------------------------------------------------------

#: The third-party module names of the closure -> their **distribution** name.
#: Only the names whose distribution differs from the module name are listed;
#: every other module is its own distribution name (``torch`` -> ``torch``,
#: ``numpy`` -> ``numpy``, ...).
#:
#: This map is a **rename** map, not an allowlist: a module that is not in it
#: is *not* dropped, it is mapped to its own name - and therefore has to be
#: covered by the declared set or by the host (that was the hole of the first
#: version of this guard: it filtered the imports through this map, so every
#: module outside of it was silently ignored).  ``cv2`` maps to the headless
#: wheel: no GUI function is called anywhere in the closure and the pack is
#: server-side.
_DISTRIBUTIONS = {
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "yaml": "pyyaml",
}

#: The distributions the **ComfyUI host** guarantees (the host's own
#: ``requirements.txt``) - the pack declares none of them, but they still
#: cover their share of the closure imports.  The value is the line of the
#: host's requirements.txt that ships the distribution: when the host drops
#: one, the failing test names the line that disappeared and the fix is a
#: one-set edit (move the distribution into ``_EXPECTED_REQUIREMENTS``).
#:
#: 2026-11-03, keep-fix5-implement1 Phase 4.1: ``scipy`` is gone - only the
#: ``basicsr.metrics`` files imported it and they are off the import path
#: now (the folder stays on disk but is not part of the closure).
_HOST_PROVIDED = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy>=1.25.0",
    "Pillow": "Pillow>=10.1.0",
    "einops": "einops",
    "tqdm": "tqdm",
    "pyyaml": "pyyaml",
    "requests": "requests",
}

#: The modules the **ComfyUI host** provides - they are not dependencies of
#: this package and therefore not part of ``requirements.txt``.
#:
#: ``app`` and ``utils`` are the top-level packages of the server: ``app``
#: carries the logger of the host (``app.logger.setup_logger`` - the pack
#: probes it to decide whether it may own a handler) and ``utils`` carries
#: the loader of the extra model paths (``utils.extra_config`` - the pack
#: asks the host to interpret ``extra_model_paths.yaml``).  Both are
#: imported **lazily** and both are absent without a server.
_HOST_MODULES = ("comfy", "comfy_api", "folder_paths", "app", "utils")

#: The vendored packages - imported relatively (P2.5), never a dependency.
_VENDORED_MODULES = ("basicsr", "facelib")

#: The pack's own top-level packages - they are part of this repository, not
#: a distribution.  Since Round 21 ``modules`` is deliberately **absent**
#: from this set: the vendored tree loads under ``_keep_vendor`` and nothing
#: imports the shared top-level ``modules`` name any more.  If the flat
#: fallback ever came back, the closure import would no longer be excluded
#: and the declared-set check below would fail loudly instead of silently
#: tolerating the regression.
_OWN_PACKAGES = ("src", "nodes")

#: The third-party modules the closure imports **lazily** (inside a
#: function).  They are not runtime dependencies of the node, so they are not
#: declared - but every one of them has to be listed here with a reason, so
#: that a new lazy import cannot slip through unnoticed either.
_OPTIONAL_LAZY_IMPORTS = {
    "dlib": (
        "facelib FaceRestoreHelper.init_dlib_from_paths - only reached with "
        "det_model='dlib'; the node offers retinaface / YOLOv5 only"
    ),
    "gdown": (
        "facelib misc.download_pretrained_models - never called by the node"
    ),
    "wandb": (
        "basicsr logger.init_wandb_logger - never called by the node "
        "(the training entry point basicsr/train.py is deleted in "
        "keep-fix5-implement1 Phase 4.1)"
    ),
}

#: The declared set (the host-guarantee policy): the only runtime dependency
#: the ComfyUI host does not already provide.
_EXPECTED_REQUIREMENTS = {"opencv-python-headless"}


def _requirement_entries():
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


class _ImportCollector(ast.NodeVisitor):
    """Collects the imported module names, split by *import time* and *lazy*.

    An import at module level (or in a class body - it runs when the class
    is created) is a dependency of the package: it runs on every import.
    An import inside a function is **lazy**: it only runs when that function
    is called, so it is not a dependency of the package itself.
    """

    def __init__(self) -> None:
        self.module_level: set[str] = set()
        self.lazy: set[str] = set()
        self._function_depth = 0

    # -- the scope handling ------------------------------------------------
    def _visit_defaults(self, arguments) -> None:
        """The default arguments are evaluated at import time."""
        defaults = list(arguments.defaults)
        defaults += [value for value in arguments.kw_defaults if value]
        for default in defaults:
            self.visit(default)

    def _visit_function(self, node) -> None:
        # decorators and default arguments are evaluated at import time
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_defaults(node.args)
        # the body runs when the function is called
        self._function_depth += 1
        for statement in node.body:
            self.visit(statement)
        self._function_depth -= 1

    def visit_FunctionDef(self, node) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node) -> None:
        self._visit_defaults(node.args)
        self._function_depth += 1
        self.visit(node.body)
        self._function_depth -= 1

    # -- the imports -------------------------------------------------------
    def visit_Import(self, node) -> None:
        for alias in node.names:
            self._add(alias.name.split(".")[0])

    def visit_ImportFrom(self, node) -> None:
        # a relative import (level > 0) stays inside the vendored tree
        if node.level == 0 and node.module:
            self._add(node.module.split(".")[0])

    def _add(self, name: str) -> None:
        if self._function_depth:
            self.lazy.add(name)
        else:
            self.module_level.add(name)


def _closure_import_collector() -> _ImportCollector:
    """The import collector of the whole runtime closure."""
    collector = _ImportCollector()
    for rel in _scan_report()["closure_files"]:
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
        collector.visit(tree)
    return collector


def _closure_imports() -> set[str]:
    """Every absolute module name imported by the runtime closure."""
    collector = _closure_import_collector()
    return collector.module_level | collector.lazy


def _closure_third_party_imports() -> tuple[set[str], set[str]]:
    """The third-party imports of the closure: (at import time, lazy only).

    Excluded are the stdlib, the vendored packages (they are imported
    relatively, P2.5) and the modules the ComfyUI host provides.
    """
    collector = _closure_import_collector()
    excluded = (
        set(sys.stdlib_module_names)
        | set(_VENDORED_MODULES)
        | set(_OWN_PACKAGES)
        | set(_HOST_MODULES)
    )
    module_level = collector.module_level - excluded
    lazy = (collector.lazy - excluded) - module_level
    return module_level, lazy


@pytest.mark.unit
def test_the_requirements_are_the_true_runtime_set():
    """``requirements.txt`` lists exactly what the host does not provide.

    The host-guarantee policy: the pack declares only the dependency the
    ComfyUI host does not ship (OpenCV, headless wheel) - everything else
    the closure imports comes from the host (``_HOST_PROVIDED``).
    """
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
    """Every third-party import of the closure is declared or host-provided.

    The check is **exhaustive**: every third-party module the closure
    imports at import time is mapped to a distribution name (identity,
    unless ``_DISTRIBUTIONS`` says otherwise) and that distribution has to
    be covered by the policy - declared in ``requirements.txt`` **or**
    shipped by the ComfyUI host (``_HOST_PROVIDED``).  A module covered by
    neither fails the test, so a new dependency cannot slip through
    unnoticed; a distribution the closure no longer imports shows up as
    stale and forces a one-set edit of the policy.
    """
    module_level, _lazy = _closure_third_party_imports()
    imported = {_DISTRIBUTIONS.get(name, name) for name in module_level}
    covered = _EXPECTED_REQUIREMENTS | set(_HOST_PROVIDED)
    assert imported == covered, (
        f"the closure imports {sorted(imported)} at import time, the policy "
        f"covers {sorted(covered)} (declared + host-provided); "
        f"uncovered: {sorted(imported - covered)}; "
        f"stale (covered but not imported): {sorted(covered - imported)}"
    )


@pytest.mark.unit
def test_every_lazy_third_party_import_is_explicitly_allowed():
    """P2.6: a lazy third-party import is listed with a reason.

    A lazy import (inside a function) is not a dependency of the package,
    so it is not declared in ``requirements.txt``.  It is still tracked:
    an unlisted one fails this test, so a new third-party import cannot
    slip through unnoticed - neither at import time nor inside a function.
    """
    _module_level, lazy = _closure_third_party_imports()
    unlisted = sorted(lazy - set(_OPTIONAL_LAZY_IMPORTS))
    assert unlisted == [], (
        f"the closure imports {unlisted} lazily; either declare them in "
        f"requirements.txt or list them in _OPTIONAL_LAZY_IMPORTS with a reason"
    )
    stale = sorted(set(_OPTIONAL_LAZY_IMPORTS) - lazy)
    assert stale == [], (
        f"_OPTIONAL_LAZY_IMPORTS lists {stale}, which the closure does not "
        f"import lazily any more"
    )


# ---------------------------------------------------------------------------
# P2.7 - no swallow-and-stub fallbacks any more
# ---------------------------------------------------------------------------

_IMPORT_PACKAGE_SCRIPT = """
import sys

root = sys.argv[1]
sys.path.insert(0, root)
import conftest  # registers the package alias and the ComfyUI mocks

import ComfyUI_KEEP.src.keep_model_configs
print("KEEP_IMPORT_OK")
"""


@pytest.mark.unit
def test_the_stub_sites_yield_the_real_vendored_symbols():
    """P2.7 (a): every former stub site resolves to the vendored symbol."""
    import importlib

    utils = importlib.import_module("ComfyUI_KEEP.src.keep_model_configs")
    processor = importlib.import_module("ComfyUI_KEEP.src.keep_processor")
    loader = importlib.import_module("ComfyUI_KEEP.src.keep_model_loader")

    assert utils.ARCH_REGISTRY.__module__.endswith(
        "_keep_vendor.basicsr.utils.registry"
    )
    # P4.7: the ``load_file_from_url`` import of the config module was dead
    # (the downloads run through ``src/keep_paths.py``) - it is gone.
    assert not hasattr(utils, "load_file_from_url")
    assert processor.img2tensor.__module__.endswith(
        "_keep_vendor.basicsr.utils.img_util"
    )
    assert processor.tensor2img.__module__.endswith(
        "_keep_vendor.basicsr.utils.img_util"
    )
    assert processor.is_gray.__module__.endswith("_keep_vendor.facelib.utils.misc")
    assert loader.FaceRestoreHelper.__module__.endswith(
        "_keep_vendor.facelib.utils.face_restoration_helper"
    )
    # the dummy fallbacks are gone
    assert not hasattr(utils, "DummyArchRegistry")
    assert not hasattr(processor, "VideoReader")
    assert not hasattr(processor, "VideoWriter")


@pytest.mark.unit
def test_the_dead_video_path_references_no_deleted_symbol():
    """P2.1/P2.7: the disabled ``process_video`` references no deleted symbol.

    The vendored ``video_util`` module (with ``VideoReader`` /
    ``VideoWriter``) is deleted in P2.1.  The unreachable body of
    ``process_video`` still called them - it is gone now, so the method is
    the raise and nothing else (the AST is checked, so the docstrings that
    *mention* the deleted symbols are not offenders).
    """
    source = (REPO_ROOT / "src" / "keep_processor.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    offenders = [
        f"{node.lineno}: {node.id}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in ("VideoReader", "VideoWriter")
    ]
    assert offenders == [], offenders


@pytest.mark.unit
def test_a_broken_vendored_import_fails_loudly(tmp_path):
    """P2.7 (b): a broken vendored dependency makes the import fail.

    The vendored ``registry`` module is renamed in a **copy** of the
    repository: the import of the package must raise instead of falling back
    to the dummy registry (which silently lost the KEEP architecture).
    """
    import shutil

    copy_root = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT,
        copy_root,
        ignore=shutil.ignore_patterns(
            "__pycache__", ".git", ".dev", ".codegraph", "*.pyc", ".pytest_cache"
        ),
    )
    registry = copy_root / "modules" / "deps" / "basicsr" / "utils" / "registry.py"
    registry.rename(registry.with_suffix(".py.broken"))

    result = _run_python(_IMPORT_PACKAGE_SCRIPT, str(copy_root))
    assert result.returncode != 0, (
        "the broken vendored import must fail loudly: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "KEEP_IMPORT_OK" not in result.stdout, result.stdout
    assert (
        "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr
    ), result.stderr
    assert "registry" in result.stderr, result.stderr
    # the dummy registry must not appear any more
    assert "DummyArchRegistry" not in result.stderr, result.stderr

    # the control: the unmodified repository imports fine
    control = _run_python(_IMPORT_PACKAGE_SCRIPT, str(REPO_ROOT))
    assert control.returncode == 0, control.stdout + control.stderr
    assert "KEEP_IMPORT_OK" in control.stdout, control.stdout


@pytest.mark.unit
def test_no_import_fallback_stub_remains_in_the_vendored_tree():
    """P2.7: the vendored tree has no swallow-and-stub fallback any more.

    A fallback stub is an ``except ImportError`` handler that defines a
    function or a class: the import failure is swallowed and the caller
    gets a local substitute instead of the vendored symbol.  The two sites
    that still had one (``facelib/detection`` and ``facelib/parsing``) were
    removed in the P2 review fix - a broken vendored dependency has to fail
    loudly, not degrade into a (broken) local stub.
    """
    offenders = []
    for path in sorted(DEPS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                caught = handler.type
                if isinstance(caught, ast.Tuple):
                    caught_names = {
                        element.id
                        for element in caught.elts
                        if isinstance(element, ast.Name)
                    }
                elif isinstance(caught, ast.Name):
                    caught_names = {caught.id}
                else:
                    caught_names = set()
                if not caught_names & {"ImportError", "ModuleNotFoundError"}:
                    continue
                for statement in handler.body:
                    if isinstance(
                        statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT).as_posix()}:"
                            f"{statement.lineno}: fallback stub "
                            f"{statement.name}"
                        )
    assert offenders == [], offenders


@pytest.mark.unit
def test_the_facelib_model_helpers_use_the_vendored_download_util():
    """P2.7: the facelib detection / parsing helpers use the vendored symbol."""
    from src.keep_vendored import vendor

    for name in ("detection", "parsing"):
        module = vendor(f"facelib.{name}")
        assert module.load_file_from_url.__module__ == (
            "_keep_vendor.basicsr.utils.download_util"
        ), module.load_file_from_url


@pytest.mark.unit
def test_the_vendored_tree_never_mutates_sys_path():
    """P2.5: the vendored tree must not mutate ``sys.path``.

    The plan's §3 import rule forbids a ``sys.path`` mutation **anywhere**
    in the package.  The last one (``basicsr/utils/file_client.py``, the
    optional ``sys_path`` of the memcached backend) was removed in the P2
    review fix - the structure guard (d) now covers the vendored tree as
    well.
    """
    offenders = []
    for path in sorted(DEPS_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if re.search(r"sys\.path\s*\.\s*(insert|append|extend)\s*\(", line):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()}:{number}: "
                    f"{line.strip()}"
                )
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# Round 21 - the pack-private namespace (``_keep_vendor``)
# ---------------------------------------------------------------------------
#
# Since Round 21 the vendored trees are imported only through
# ``src/keep_vendored.py`` under the pack-private name ``_keep_vendor``
# (pip ``_vendor`` semantics): one canonical name per process, anchored to
# the pack's own files, independent of the name the host registers the pack
# under.  The pins of this section make the property un-rottable:
#
# * the vendored tree itself must stay free of the imports that would
#   defeat the private namespace (absolute self-imports, import machinery);
# * the pack code and the harness must never reach the shared top-level
#   names (``modules`` everywhere, ``basicsr`` / ``facelib`` in the pack
#   runtime);
# * a flat engine import must bind no shared name - even on a ``sys.path``
#   where a foreign ``modules/`` package wins (the decoy probe);
# * the alias mode and the flat mode must share one vendored instantiation
#   (registry identity).

#: The vendored trees' own top-level names: an absolute import of one of
#: these from **inside** ``modules/deps`` would bind the tree under a
#: shared name instead of the private one.
_VENDORED_ROOTS = ("basicsr", "facelib")

#: The import machinery the vendored tree must not use: a dynamic import
#: would re-resolve a module outside the private namespace (and the static
#: closure of ``tools/scan_compat.py`` could not see it either).
_DYNAMIC_IMPORT_CALLS = (
    "import_module",
    "__import__",
    "create_module",
    "exec_module",
    "spec_from_file_location",
)

#: The shared top-level names banned per directory (AC4): the pack runtime
#: may reach none of the three, the harness surface (``tools/``, ``tests/``)
#: not the ``modules`` name - that is the name the old flat fallback bound.
_SHARED_NAME_BAN = {
    "src": ("modules", "basicsr", "facelib"),
    "nodes": ("modules", "basicsr", "facelib"),
    "tools": ("modules",),
    "tests": ("modules",),
}

#: The §2.6 (b) exemption of the namespace ban: the machine-local GPU
#: profiling probe keeps its three function-local flat imports
#: (``from modules.deps... import ...``).  It is a dev-time probe,
#: referenced by no test, ``pyproject.toml`` entry or CI workflow, and the
#: round's ownership partition leaves it unowned - so pin 2 exempts it by
#: explicit path with this reason.  Migrating it to the loader
#: (design §2.6 (a)) lifts the exemption; the stale-exemption check below
#: forces that edit.
_PROF_TOOL_EXEMPTION = {
    "tools/prof_gpu_video_pipeline.py": (
        "machine-local GPU profiling probe: flat-imports the vendored trees "
        "function-locally; not run by the tier-1 gate, not referenced by "
        "tests/ or CI (Round 21 design §2.6 (b))"
    ),
}


def _absolute_import_roots(node: ast.AST):
    """The top-level names an absolute ``Import`` / ``ImportFrom`` reaches.

    Relative imports (``level > 0``) stay inside their own tree and yield
    nothing; string contents (docstrings, probe sources) are not import
    statements and are never seen by the AST.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name.split(".")[0]
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        yield node.module.split(".")[0]


def _absolute_import_offenders(rel_dirs, banned_roots) -> list[str]:
    """Every absolute import under ``rel_dirs`` reaching ``banned_roots``."""
    offenders = []
    for rel_dir in rel_dirs:
        for path in sorted((REPO_ROOT / rel_dir).rglob("*.py")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                for root_name in _absolute_import_roots(node):
                    if root_name in banned_roots:
                        offenders.append(f"{rel}:{node.lineno} {root_name}")
    return offenders


def _dynamic_import_offenders() -> list[str]:
    """Every use of the import machinery inside the vendored tree."""
    offenders = []
    for path in sorted(DEPS_DIR.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.attr if isinstance(func, ast.Attribute)
                    else getattr(func, "id", None)
                )
                if name in _DYNAMIC_IMPORT_CALLS:
                    offenders.append(f"{rel}:{node.lineno} {name}()")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "importlib.util" or (
                        alias.name == "importlib"
                        and any(a.name == "util" for a in node.names)
                    ):
                        offenders.append(f"{rel}:{node.lineno} {alias.name}")
            elif isinstance(node, ast.ImportFrom) and (
                node.module == "importlib.util"
                or (
                    node.module == "importlib"
                    and any(a.name == "util" for a in node.names)
                )
            ):
                offenders.append(f"{rel}:{node.lineno} importlib.util")
    return offenders


@pytest.mark.unit
def test_no_absolute_self_import_or_import_machinery_remains_in_the_vendored_tree():
    """AC2, Round 21: the vendored tree cannot escape the private namespace.

    Under ``_keep_vendor`` the trees' own imports must stay relative (the
    private root's ``__path__`` carries the subpackages) and the trees must
    not use the import machinery: a dynamic import would re-resolve a
    module under a shared name and defeat the one-instantiation property.
    The scan is AST based, so the commented absolute import at
    ``facelib/utils/misc.py:9`` is not an offender.
    """
    absolute = _absolute_import_offenders(["modules/deps"], _VENDORED_ROOTS)
    assert absolute == [], (
        "the vendored tree imports itself under a shared top-level name: "
        f"{absolute}"
    )
    dynamic = _dynamic_import_offenders()
    assert dynamic == [], (
        f"the vendored tree uses the import machinery: {dynamic}"
    )


@pytest.mark.unit
def test_no_pack_code_reaches_the_shared_top_level_names():
    """AC4, Round 21: the pack imports none of the shared names - in any mode.

    The ban covers ``src/``, ``nodes/``, ``tools/`` and ``tests/``: no
    absolute import may reach the top-level ``modules`` name (the name the
    old dual-mode flat fallback bound - first-importer-wins decided which
    tree the pack got), and the pack runtime (``src/``, ``nodes/``) may not
    reach ``basicsr`` / ``facelib`` either.  After the Round 21 migration
    the count is zero; the one exemption is recorded with its reason.
    """
    exempt = set(_PROF_TOOL_EXEMPTION)
    offenders = _absolute_import_offenders(
        ["src", "nodes", "tools", "tests"], {"modules"}
    )
    offenders += _absolute_import_offenders(["src", "nodes"], {"basicsr", "facelib"})
    remaining = [off for off in offenders if off.split(":")[0] not in exempt]
    assert remaining == [], (
        "pack code reaches a shared top-level name; every hit must move to "
        f"the pack-private namespace (src/keep_vendored.py): {remaining}"
    )
    # the exemption is not a permanent hole: when the prof tool migrates,
    # the stale entry forces this one-set edit instead of hiding silently
    stale = sorted(
        path for path in exempt
        if not any(off.split(":")[0] == path for off in offenders)
    )
    assert stale == [], (
        "the namespace-ban exemption is stale (the file no longer imports a "
        f"shared name) - drop it from _PROF_TOOL_EXEMPTION: {stale}"
    )


#: The flat decoy probe (AC3 flat / AC4): a foreign ``modules/`` package on
#: ``sys.path`` (the decoy) must win nothing the pack binds, and the pack's
#: vendored tree must still come from the pack's own files.  The flat
#: engine mode itself stays (the design keeps it - no launcher can be
#: *proven* to never import flat), so the probe proves the safe half: after
#: the engine imports, ``import modules`` reaches the **decoy**.
_FLAT_DECOY_PROBE = """\
import importlib, json, sys, types

decoy_root, repo_root = sys.argv[1], sys.argv[2]

# The decoy root FIRST, the repository second: on this sys.path a flat
# import of every shared name the decoy ships would resolve into the decoy
# - unless the pack really stopped binding them (Round 21).
sys.path.insert(0, decoy_root)
sys.path.append(repo_root)

# The vendored face_restoration_helper imports ``comfy`` at module level;
# the probe has no server, so a minimal stand-in covers the two names it
# imports (the real host provides them in every real import mode).
comfy = types.ModuleType("comfy")
model_management = types.ModuleType("comfy.model_management")
comfy_utils = types.ModuleType("comfy.utils")
comfy_utils.tiled_scale = lambda *args, **kwargs: None
comfy.model_management = model_management
comfy.utils = comfy_utils
sys.modules["comfy"] = comfy
sys.modules["comfy.model_management"] = model_management
sys.modules["comfy.utils"] = comfy_utils

import src.keep_model_configs as configs
import src.keep_processor as processor
import src.keep_model_loader as loader

shadowed = sorted(
    name for name in ("modules", "basicsr", "facelib") if name in sys.modules
)
report = {
    "shadowed": shadowed,
    "registry_module": type(configs.ARCH_REGISTRY).__module__,
    "keep_registered": configs.ARCH_REGISTRY.get("KEEP") is not None,
    "img2tensor_module": processor.img2tensor.__module__,
    "face_helper_module": loader.FaceRestoreHelper.__module__,
    "flat_engine_imported": "src.keep_model_configs" in sys.modules,
}
print("KEEP_FLAT_DECOY:" + json.dumps(report))

# The pack bound no shared name - a flat import NOW must reach the decoy,
# not the pack's own modules/ package.
import modules
print("KEEP_FLAT_DECOY_DECOY:" + str(getattr(modules, "KEEP_DECOY", False)))
"""


@pytest.mark.unit
def test_the_flat_engine_imports_bind_no_shared_name_and_the_decoy_survives(tmp_path):
    """AC3 (flat) / AC4: a foreign ``modules/`` package wins nothing.

    The probe builds a decoy tree (``modules/__init__.py`` and
    ``modules/deps/__init__.py``, each with a marker constant), puts the
    decoy root **first** on ``sys.path`` and the repository second, and
    flat-imports the three engines.  Assertions:

    * no top-level ``modules`` / ``basicsr`` / ``facelib`` is in
      ``sys.modules`` after the engine imports - the pack bound none of
      the shared names (``"src"`` *is* imported: the flat engine mode
      still works, from the repository);
    * the registry and the vendored symbols come from the private
      namespace, and the KEEP architecture is registered - the pack's own
      files won, not the decoy;
    * ``import modules`` afterwards binds the **decoy** (marker found) -
      the pack imported first without shadowing the shared name.
    """
    decoy_root = tmp_path / "decoy-root"
    (decoy_root / "modules" / "deps").mkdir(parents=True)
    (decoy_root / "modules" / "__init__.py").write_text(
        "KEEP_DECOY = True\n", encoding="utf-8"
    )
    (decoy_root / "modules" / "deps" / "__init__.py").write_text(
        "KEEP_DECOY_DEPS = True\n", encoding="utf-8"
    )

    result = _run_python(_FLAT_DECOY_PROBE, str(decoy_root), str(REPO_ROOT))
    assert result.returncode == 0, (
        f"the flat decoy probe failed\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    sentinel = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("KEEP_FLAT_DECOY:")
    ]
    assert sentinel, (
        f"the probe produced no report\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    report = json.loads(sentinel[-1].split(":", 1)[1])
    assert report["shadowed"] == [], report
    assert report["registry_module"] == "_keep_vendor.basicsr.utils.registry", report
    assert report["keep_registered"] is True, report
    assert report["img2tensor_module"].startswith("_keep_vendor.basicsr"), report
    assert report["face_helper_module"].startswith("_keep_vendor.facelib"), report
    assert report["flat_engine_imported"] is True, report
    # the pack did not shadow the shared name: the decoy wins a flat import
    assert "KEEP_FLAT_DECOY_DECOY:True" in result.stdout, result.stdout


#: The registry-identity probe (AC3): the alias mode and the flat mode in
#: **one** interpreter must share the one vendored instantiation.  This is
#: the collision the round closes - two engine copies that each built their
#: own vendored tree would carry two ``ARCH_REGISTRY`` objects, and the
#: second execution of ``keep_arch`` fails the registry's duplicate assert.
_REGISTRY_IDENTITY_PROBE = """\
import importlib, json, sys

root = sys.argv[1]
sys.path.insert(0, root)
import conftest  # registers the package alias and the ComfyUI mocks

alias_configs = importlib.import_module("ComfyUI_KEEP.src.keep_model_configs")
flat_configs = importlib.import_module("src.keep_model_configs")
alias_processor = importlib.import_module("ComfyUI_KEEP.src.keep_processor")
flat_processor = importlib.import_module("src.keep_processor")
alias_loader = importlib.import_module("ComfyUI_KEEP.src.keep_model_loader")
flat_loader = importlib.import_module("src.keep_model_loader")

report = {
    "vendor_roots": [n for n in sys.modules if n == "_keep_vendor"],
    "registry_entries": [
        n for n in sys.modules if n == "_keep_vendor.basicsr.utils.registry"
    ],
    "registry_module": type(alias_configs.ARCH_REGISTRY).__module__,
    "same_registry_object": (
        alias_configs.ARCH_REGISTRY is flat_configs.ARCH_REGISTRY
    ),
    "same_img2tensor": alias_processor.img2tensor is flat_processor.img2tensor,
    "same_face_helper": (
        alias_loader.FaceRestoreHelper is flat_loader.FaceRestoreHelper
    ),
    "shared_names_bound": sorted(
        n for n in sys.modules if n in ("modules", "ComfyUI_KEEP.modules")
    ),
}
print("KEEP_REGISTRY_IDENTITY:" + json.dumps(report))
"""


@pytest.mark.unit
def test_the_alias_mode_and_the_flat_mode_share_one_vendored_instantiation():
    """AC3: two import modes, one vendored tree, one ``ARCH_REGISTRY``.

    The probe imports the engines under the pytest alias **and** flat in a
    single interpreter - the suite's other probes never mix the two modes.
    Both engine copies must hold the same vendored objects: exactly one
    ``_keep_vendor`` root in ``sys.modules``, exactly one registry module
    entry, and object identity across the copies (the registry's duplicate
    assert would have failed the probe on a second instantiation).
    """
    result = _run_python(_REGISTRY_IDENTITY_PROBE, str(REPO_ROOT))
    assert result.returncode == 0, (
        f"the registry-identity probe failed\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    sentinel = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("KEEP_REGISTRY_IDENTITY:")
    ]
    assert sentinel, (
        f"the probe produced no report\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    report = json.loads(sentinel[-1].split(":", 1)[1])
    assert report["vendor_roots"] == ["_keep_vendor"], report
    assert report["registry_entries"] == ["_keep_vendor.basicsr.utils.registry"], (
        report
    )
    assert report["registry_module"] == "_keep_vendor.basicsr.utils.registry", report
    assert report["same_registry_object"] is True, report
    assert report["same_img2tensor"] is True, report
    assert report["same_face_helper"] is True, report
    assert report["shared_names_bound"] == [], report
