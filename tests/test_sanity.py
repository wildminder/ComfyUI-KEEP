"""P1.1 - harness sanity.

The plan requires the harness to be green **before** the refactor starts, so
this file pins the two facts every later phase relies on:

* the interpreter is a supported Python (the plan's embedded interpreter is
  3.13; the CI matrix of P1.5 adds 3.10 and 3.12 - see the deviation note in
  ``docs/audit/2026-10-23-p1-test-infrastructure.md``),
* ``torch`` is importable - every tier of the suite imports it, and the
  ``sample_image`` fixture of the root conftest is a torch tensor.
"""

from __future__ import annotations

import sys

import pytest

#: The Python versions the project supports (P1.5 CI matrix + the local
#: embedded interpreter, which is 3.13).
SUPPORTED_PYTHON_VERSIONS = {(3, 10), (3, 12), (3, 13)}


@pytest.mark.unit
def test_python_and_torch():
    """The interpreter is supported and torch is importable."""
    version = sys.version_info[:2]
    assert version in SUPPORTED_PYTHON_VERSIONS, (
        f"unsupported Python {version[0]}.{version[1]}; supported: "
        f"{sorted(SUPPORTED_PYTHON_VERSIONS)}"
    )

    import torch  # noqa: PLC0415 - imported here on purpose: the import is the assertion

    assert isinstance(torch.__version__, str) and torch.__version__, (
        "torch must expose a version string"
    )
    assert hasattr(torch, "Tensor"), "torch must expose the Tensor type"
