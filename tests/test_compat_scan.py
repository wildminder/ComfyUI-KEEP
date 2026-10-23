"""P0.3 - compatibility scanner tests.

The scanner is unit-tested against fixtures (known-bad snippets are flagged,
known-good ones are not), including a fixture for the vendored ``pip.main``
shim.  The tests then check the scanner against the **real** vendored tree and
against the audit document that lists the fix list.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCANNER = REPO_ROOT / "tools" / "scan_compat.py"
AUDIT_DOC = REPO_ROOT / "docs" / "audit" / "2026-10-22-compat-scan.md"

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
    report = scanner.scan_repo(REPO_ROOT)
    findings = report["findings"]
    assert report["closure_file_count"] > 0
    located = {(f["path"], f["line"], f["pattern"]) for f in findings}

    assert (
        "modules/deps/basicsr/archs/keep_arch.py",
        2,
        "re_import",
    ) in located, "the `from re import T` breakage must be found"
    assert (
        "modules/deps/basicsr/utils/video_util.py",
        13,
        "import_pip",
    ) in located, "the vendored pip shim must be found"
    assert (
        "modules/deps/basicsr/utils/video_util.py",
        14,
        "pip_main",
    ) in located, "the vendored pip.main call must be found"
    assert (
        "modules/deps/basicsr/archs/gmflow_arch.py",
        81,
        "pdb",
    ) in located, "the live pdb breakpoint must be found"
    assert (
        "modules/deps/basicsr/archs/keep_arch.py",
        9,
        "torch_profiler",
    ) in located, "the torch.profiler import must be found"


@pytest.mark.unit
def test_closure_contains_the_runtime_files(scanner):
    report = scanner.scan_repo(REPO_ROOT)
    closure = set(report["closure_files"])
    for expected in (
        "modules/deps/basicsr/archs/keep_arch.py",
        "modules/deps/basicsr/archs/gmflow_arch.py",
        "modules/deps/basicsr/archs/arch_util.py",
        "modules/deps/basicsr/utils/video_util.py",
        "modules/deps/basicsr/utils/img_util.py",
        "modules/deps/facelib/utils/face_restoration_helper.py",
    ):
        assert expected in closure, f"{expected} is not in the runtime closure"


@pytest.mark.unit
def test_audit_doc_lists_every_finding(scanner):
    report = scanner.scan_repo(REPO_ROOT)
    text = AUDIT_DOC.read_text(encoding="utf-8")
    for finding in report["findings"]:
        location = f"{finding['path']}:{finding['line']}"
        assert location in text, f"{location} is not in the fix list"
