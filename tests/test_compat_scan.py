"""P0.3 - compatibility scanner tests.

The scanner is unit-tested against fixtures (known-bad snippets are flagged,
known-good ones are not), including a fixture for the vendored ``pip.main``
shim.  The tests then check the scanner against the **real** vendored tree and
against the audit document that lists the fix list.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCANNER = REPO_ROOT / "tools" / "scan_compat.py"
AUDIT_DOC = REPO_ROOT / ".dev" / "docs" / "audit" / "2026-10-22-compat-scan.md"
DEPS_DIR = REPO_ROOT / "modules" / "deps"

#: The audit artifacts are development documentation: they are not shipped with
#: the pack and may be absent from a working tree.
pytestmark = pytest.mark.skipif(
    not AUDIT_DOC.exists(),
    reason=f"the audit artifact {AUDIT_DOC.name} is not in the working tree",
)

KNOWN_BAD = '''\
import pdb
import imp
import pip
from re import T
from distutils.version import LooseVersion
from collections import Mapping
import numpy as np

x = np.float(1.0)
y = np.int(2)
z = np.bool(True)
o = np.object()
s = np.str("a")
n = np.NaN
amp = torch.cuda.amp
prof = torch.profiler.profile
pip.main(["install", "--user", "ffmpeg-python"])
pdb.set_trace()
'''

KNOWN_GOOD = '''\
import math
import numpy as np
import torch

from torch import nn

def add(a, b):
    """A perfectly fine module."""
    return a + b + float(np.pi) + torch.tensor(1.0).sum()
'''

PIP_SHIM_FIXTURE = '''\
import cv2
import sys
import numpy as np

try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg
'''


def _load_scanner():
    spec = importlib.util.spec_from_file_location("keep_scan_compat", SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scanner():
    return _load_scanner()


@pytest.mark.unit
def test_known_bad_fixture_is_fully_flagged(scanner):
    findings = scanner.scan_source(KNOWN_BAD, "fixture_bad.py")
    flagged = {f["pattern"] for f in findings}
    expected = {
        "re_import",
        "np_float",
        "np_int",
        "np_bool",
        "np_object",
        "np_str",
        "np_nan",
        "distutils",
        "import_imp",
        "torch_cuda_amp",
        "pdb",
        "torch_profiler",
        "collections_mapping",
        "import_pip",
        "pip_main",
    }
    assert expected <= flagged, f"not flagged: {sorted(expected - flagged)}"
    # the pdb pattern fires twice (import + set_trace), the pip pattern twice
    assert len([f for f in findings if f["pattern"] == "pdb"]) == 2
    assert len([f for f in findings if f["pattern"] == "pip_main"]) == 1


@pytest.mark.unit
def test_known_good_fixture_is_not_flagged(scanner):
    assert scanner.scan_source(KNOWN_GOOD, "fixture_good.py") == []


@pytest.mark.unit
def test_pip_main_shim_fixture_is_flagged(scanner):
    findings = scanner.scan_source(PIP_SHIM_FIXTURE, "video_util.py")
    by_pattern = {f["pattern"]: f for f in findings}
    assert "import_pip" in by_pattern, findings
    assert "pip_main" in by_pattern, findings
    assert by_pattern["import_pip"]["line"] == 8
    assert by_pattern["pip_main"]["line"] == 9
    assert "pip.main" in by_pattern["pip_main"]["snippet"]


@pytest.mark.unit
def test_finding_line_numbers_are_exact(scanner):
    findings = scanner.scan_source(KNOWN_BAD, "fixture_bad.py")
    lines = {f["pattern"]: f["line"] for f in findings}
    assert lines["re_import"] == 4
    assert lines["import_pip"] == 3
    assert lines["np_nan"] == 14
    assert lines["pip_main"] == 17


@pytest.mark.unit
def test_real_tree_findings(scanner):
    """The live tree must not contain any finding any more.

    P2.1 removed ``from re import T``, the ``pdb`` imports, the
    ``torch.profiler`` import and the pip shim module; P2.3 removed the
    remaining ``distutils`` import.  The list below is therefore the
    *expected* state of the tree - a new finding fails this test.
    """
    report = scanner.scan_repo(REPO_ROOT)
    findings = report["findings"]
    assert report["closure_file_count"] > 0
    located = {(f["path"], f["line"], f["pattern"]) for f in findings}

    assert located == set(), (
        "the vendored closure must be free of compatibility findings, "
        f"got {sorted(located)}"
    )


@pytest.mark.unit
def test_closure_contains_the_runtime_files(scanner):
    report = scanner.scan_repo(REPO_ROOT)
    closure = set(report["closure_files"])
    for expected in (
        "modules/deps/basicsr/archs/keep_arch.py",
        "modules/deps/basicsr/archs/gmflow_arch.py",
        "modules/deps/basicsr/archs/arch_util.py",
        "modules/deps/basicsr/utils/img_util.py",
        "modules/deps/facelib/utils/face_restoration_helper.py",
    ):
        assert expected in closure, f"{expected} is not in the runtime closure"
    # P2.1 deleted the pip-shim module - it must not be in the closure any more
    assert "modules/deps/basicsr/utils/video_util.py" not in closure


@pytest.mark.unit
def test_closure_contains_the_explicitly_listed_modules(scanner):
    """The modules of the explicit import lists belong to the closure.

    P2.4 replaced the four eager folder scans of the vendored ``basicsr``
    packages with explicit import lists - the same modules are loaded by
    ``import basicsr``, but the closure computation no longer has to model a
    dynamic scan.  2026-11-03, keep-fix5-implement1 Phase 4.1: the
    ``models``/``data``/``losses`` packages are deleted and the ``archs``
    list is reduced to the four architectures the pack uses - the list below
    is the regenerated closure manifest.
    """
    report = scanner.scan_repo(REPO_ROOT)
    closure = set(report["closure_files"])
    for expected in (
        # archs/__init__.py - explicit list (Phase 4.1)
        "modules/deps/basicsr/archs/gmflow_arch.py",
        "modules/deps/basicsr/archs/keep_arch.py",
        "modules/deps/basicsr/archs/spectral_norm_arch.py",
        "modules/deps/basicsr/archs/vqgan_arch.py",
        # utils/__init__.py - explicit list (Phase 4.1)
        "modules/deps/basicsr/utils/img_util.py",
        "modules/deps/basicsr/utils/logger.py",
        "modules/deps/basicsr/utils/misc.py",
        # reached through 'from <package> import <submodule>'
        "modules/deps/basicsr/utils/registry.py",
        "modules/deps/basicsr/version.py",
    ):
        assert expected in closure, (
            f"{expected} is not in the runtime closure - the explicit import "
            "lists of the vendored basicsr are not resolved"
        )
    # keep-fix5-implement1 Phase 4.1: the deleted packages are gone from the
    # closure, and ``metrics/`` stays on disk but off the import path.
    for absent in (
        "modules/deps/basicsr/models/keep_model.py",
        "modules/deps/basicsr/data/reds_dataset.py",
        "modules/deps/basicsr/losses/basic_loss.py",
        "modules/deps/basicsr/utils/flow_util.py",
        "modules/deps/basicsr/metrics/niqe.py",
    ):
        assert absent not in closure, (
            f"{absent} is deleted or off the import path but in the closure"
        )


@pytest.mark.unit
def test_exit_code_contract(scanner, monkeypatch):
    """Exit code 0 = no findings, 1 = findings found (a result, not a crash)."""
    assert scanner.main(["--quiet"]) == 0, (
        "the real tree must be free of findings, so the scanner must exit 0"
    )
    monkeypatch.setattr(
        scanner,
        "scan_repo",
        lambda *args, **kwargs: {
            "entry_points": [],
            "closure_file_count": 0,
            "closure_files": [],
            "finding_count": 0,
            "findings_by_pattern": {},
            "findings": [],
        },
    )
    assert scanner.main(["--quiet"]) == 0, (
        "a clean scan must exit 0"
    )


@pytest.mark.unit
def test_audit_doc_lists_every_finding(scanner):
    report = scanner.scan_repo(REPO_ROOT)
    text = AUDIT_DOC.read_text(encoding="utf-8")
    for finding in report["findings"]:
        location = f"{finding['path']}:{finding['line']}"
        assert location in text, f"{location} is not in the fix list"


# ---------------------------------------------------------------------------
# P2.3 - the mechanical Python 3.13 / numpy 2 fixes
# ---------------------------------------------------------------------------

#: Imports every module of the runtime closure in a **fresh** interpreter (the
#: import must not leak into the pytest session) and reports the failures.
#: Since Round 21 the vendored part of the closure is imported under the
#: pack-private namespace: the loader anchors the private root first, then
#: the walk imports every closure module by its private name.
_CLOSURE_IMPORT_SCRIPT = """\
import importlib, json, sys

sys.path.insert(0, sys.argv[1])
import conftest  # registers the package alias and the ComfyUI mocks

from src.keep_vendored import _ensure_vendor_root

_ensure_vendor_root()  # the private root, before the closure walk below

failures = []
for name in json.loads(sys.argv[2]):
    try:
        importlib.import_module(name)
    except BaseException as exc:
        failures.append({"module": name, "type": type(exc).__name__, "error": str(exc)})
print("KEEP_CLOSURE_FAILURES:" + json.dumps(failures))
"""


def _closure_module_names(report):
    """The private-namespace names of the vendored part of the closure.

    Since Round 21 the vendored tree lives under ``_keep_vendor``
    (``src/keep_vendored.py``) - the closure walk imports it by that name,
    the same name the pack runtime binds.
    """
    names = []
    for rel in report["closure_files"]:
        if not rel.startswith("modules/deps/"):
            continue
        dotted = rel[len("modules/deps/"):].replace("/", ".").removesuffix(".py")
        if dotted.endswith(".__init__"):
            dotted = dotted[: -len(".__init__")]
        names.append("_keep_vendor." + dotted)
    return names


@pytest.mark.unit
def test_every_closure_module_imports(scanner):
    """P2.3 acceptance: every module of the runtime closure still imports.

    The imports run in a subprocess: the vendored packages register their
    classes in global registries and must not pollute the pytest session.
    """
    report = scanner.scan_repo(REPO_ROOT)
    names = _closure_module_names(report)
    assert names, "the runtime closure is empty"

    result = subprocess.run(
        [sys.executable, "-c", _CLOSURE_IMPORT_SCRIPT, str(REPO_ROOT), json.dumps(names)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    sentinel = [
        line for line in result.stdout.splitlines() if line.startswith("KEEP_CLOSURE_FAILURES:")
    ]
    assert sentinel, (
        f"the closure import probe produced no result\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    failures = json.loads(sentinel[-1].split(":", 1)[1])
    assert failures == [], f"closure modules failed to import: {failures}"
