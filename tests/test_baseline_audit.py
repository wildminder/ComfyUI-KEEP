"""P0.1 - baseline audit tests.

These tests assert the *recorded baseline* against the **current code**, not
against the audit document: the document is only checked for the two required
signatures, as the plan requires.

Recorded baseline (docs/audit/2026-10-22-baseline-import.md):

* the package **imports** through ComfyUI's own loader (spec_from_file_location
  + alias, no server) - the swallow-and-stub blocks of B9 hide the real errors;
* the **model build** fails at the eager basicsr arch scan with
  ``ImportError: cannot import name 'T' from 're'`` (Python 3.13 removed
  ``re.T``), which the ``DummyArchRegistry`` fallback in ``modules/utils.py``
  converts into ``ARCH_REGISTRY.get('KEEP') is None`` and therefore into
  ``RuntimeError: KEEP architecture class not found in ARCH_REGISTRY.``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE = REPO_ROOT / "tools" / "baseline_probe.py"
AUDIT_DOC = REPO_ROOT / "docs" / "audit" / "2026-10-22-baseline-import.md"
DEPS_DIR = REPO_ROOT / "modules" / "deps"


def _run_probe(tmp_path: Path, *extra: str) -> dict:
    """Run the baseline probe in a fresh interpreter and return its report.

    A subprocess is used on purpose: the probe reproduces the *real* import
    behaviour of the package (including the ``sys.path`` mutation of B4 and the
    partially imported ``basicsr`` package), which must not leak into the
    pytest session.
    """
    report_path = tmp_path / "baseline-report.json"
    result = subprocess.run(
        [sys.executable, str(PROBE), "--json", str(report_path), *extra],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"baseline probe failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert report_path.is_file(), "baseline probe did not write its report"
    return json.loads(report_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def baseline_report(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("baseline")
    report = _run_probe(tmp_path)
    if "fatal" in report:
        pytest.skip(
            "baseline probe could not locate a ComfyUI source tree: "
            f"{report['fatal']} (set COMFYUI_ROOT to point at one)"
        )
    return report


@pytest.mark.unit
def test_package_imports_through_comfyui_loader(baseline_report):
    """The package imports cleanly today - the failure is at model-build time."""
    assert baseline_report["import"]["ok"] is True, baseline_report["import"]
    assert baseline_report["import"]["error"] is None
    # the three compatibility-guaranteed node ids are registered
    assert baseline_report["node_class_mappings"] == [
        "KEEP_FaceUpscaleImage",
        "KEEP_ModelLoader",
        "KEEP_ProcessImageSequence",
    ]


@pytest.mark.unit
def test_utils_exposes_stub_arch_registry(baseline_report):
    """(a) modules/utils.py exposes the *stub* ARCH_REGISTRY path.

    The dummy registry must be in place (the real one could not even be
    imported) and it must not know the 'KEEP' architecture.
    """
    arch_registry = baseline_report["arch_registry"]
    assert arch_registry["is_stub"] is True, arch_registry
    assert arch_registry["class_name"] == "DummyArchRegistry", arch_registry
    assert arch_registry["get_keep"] is False, arch_registry
    assert baseline_report["stub_summary"]["fired"], baseline_report["stub_summary"]
    assert "utils_arch_registry_dummy" in baseline_report["stub_summary"]["fired"]


@pytest.mark.unit
def test_vendored_keep_arch_import_raises_import_error_T():
    """(b) importing the vendored ``keep_arch`` raises ImportError about 'T'.

    Executed in-process (with a clean-up of the interpreter state) so the
    assertion is about the vendored source, not about a recorded string.
    """
    saved_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "basicsr" or name.startswith("basicsr.")
    }
    for name in saved_modules:
        del sys.modules[name]
    sys.path.insert(0, str(DEPS_DIR))
    try:
        with pytest.raises(ImportError) as excinfo:
            import basicsr.archs.keep_arch  # noqa: F401
        message = str(excinfo.value)
        assert "cannot import name 'T' from 're'" in message, message
        assert "'T'" in message, message
    finally:
        sys.path[:] = saved_path
        for name in [
            n for n in list(sys.modules) if n == "basicsr" or n.startswith("basicsr.")
        ]:
            del sys.modules[name]
        sys.modules.update(saved_modules)


@pytest.mark.unit
def test_eager_arch_scan_is_the_real_failure(baseline_report):
    """The eager arch scan is where the import actually breaks (B1)."""
    arch_scan = baseline_report["arch_scan"]
    assert arch_scan["ok"] is False, arch_scan
    assert arch_scan["error"]["type"] == "ImportError", arch_scan
    assert "cannot import name 'T' from 're'" in arch_scan["error"]["message"]
    # the traceback must point at the vendored arch module
    assert "keep_arch.py" in arch_scan["error"]["traceback"]
    assert "archs" in arch_scan["error"]["traceback"]


@pytest.mark.unit
def test_model_build_fails_not_the_import(baseline_report):
    """Acceptance: the failure mode is *model-build*, not import.

    The build is attempted with a stubbed weight file present, so a missing
    weight cannot explain the failure.
    """
    model_build = baseline_report["model_build"]
    assert model_build["ok"] is False, model_build
    assert model_build["stub_weight_present"] is True, model_build
    assert model_build["error"]["type"] == "RuntimeError", model_build
    assert "ARCH_REGISTRY" in model_build["error"]["message"], model_build
    assert "keep_model_loader.py" in model_build["error"]["traceback"]


@pytest.mark.unit
def test_diffusers_symbols_are_unusable_in_this_env(baseline_report):
    """B2: the vendored diffusers import cannot be satisfied in this env."""
    diffusers = baseline_report["diffusers"]
    assert diffusers["ok"] is False, diffusers
    assert diffusers["error"]["type"] == "ImportError", diffusers
    assert "cached_download" in diffusers["error"]["message"]


@pytest.mark.unit
def test_pip_shim_is_reachable_and_fires_without_ffmpeg(tmp_path):
    """B10: the vendored video_util pip shim runs at import time.

    On this host ``ffmpeg-python`` is present (installed --user by an earlier
    baseline reproduction), so the shim stays inert; with ``ffmpeg`` blocked the
    probe records the ``pip.main([...])`` call instead of performing it.
    """
    report = _run_probe(tmp_path, "--simulate-missing-ffmpeg")
    pip_shim = report["pip_shim"]
    assert pip_shim["pip_main_calls"] == [["install", "--user", "ffmpeg-python"]], (
        pip_shim
    )
    assert pip_shim["ffmpeg_importable"] is False


@pytest.mark.unit
def test_vendored_deps_are_shadowed_globally(baseline_report):
    """B4: modules/deps is injected into sys.path - global shadowing."""
    shadowing = baseline_report["sys_path_shadowing"]
    assert shadowing["deps_path_on_sys_path"] is True, shadowing
    assert shadowing["facelib_in_sys_modules"] is True, shadowing


@pytest.mark.unit
def test_audit_doc_contains_both_signatures():
    """The audit document must record both signatures of the baseline."""
    assert AUDIT_DOC.is_file(), f"missing audit document: {AUDIT_DOC}"
    text = AUDIT_DOC.read_text(encoding="utf-8")
    assert "cannot import name 'T' from 're'" in text, "missing B1 signature"
    assert "DummyArchRegistry" in text, "missing B9/stub signature"
    assert "KEEP architecture class not found in ARCH_REGISTRY." in text, (
        "missing model-build failure signature"
    )
