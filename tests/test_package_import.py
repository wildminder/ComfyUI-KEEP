"""P3.2 / P3.6 - the package imports.

The package is imported the way ComfyUI imports it (the root ``__init__.py``
is executed under an alias) - once against the **real** ComfyUI source tree
(the ``headless`` tier, P3.2) and once against the **mocks** of the root
conftest (the ``mock_integration`` tier, P3.6).

The headless variant needs the ComfyUI source tree: without ``COMFYUI_ROOT``
the test **skips** with a message that names the variable (the acceptance
gate runs without environment variables).
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADLESS_HELPER = REPO_ROOT / "tests" / "_headless.py"

#: The environment variable that points at the ComfyUI source tree.
COMFYUI_ROOT_ENV = "COMFYUI_ROOT"

#: The node ids of the pack.
PACK_NODE_IDS = [
    "KEEP_FaceUpscaleImage",
    "KEEP_ModelLoader",
    "KEEP_ProcessImageSequence",
]

#: The probe: it registers the package the way ComfyUI does (the root
#: ``__init__.py`` is executed under an alias) and reports the result.
_HEADLESS_PROBE = """
import json, os, sys

comfyui_root, repo_root = sys.argv[1], sys.argv[2]
sys.path.insert(0, comfyui_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "tests"))

import _headless

module = _headless.register_package(
    repo_root, alias="ComfyUI_KEEP_headless", execute=True
)
print(
    "KEEP_IMPORT_REPORT:"
    + json.dumps(
        {
            "node_ids": sorted(getattr(module, "NODE_CLASS_MAPPINGS", {})),
            "display_names": dict(
                getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
            ),
            "has_logger": hasattr(module, "logger"),
        }
    )
)
"""


def _comfyui_root() -> str:
    """The ComfyUI source tree - or a skip that names the variable."""
    value = (os.environ.get(COMFYUI_ROOT_ENV) or "").strip()
    if not value:
        pytest.skip(
            f"{COMFYUI_ROOT_ENV} is not set - the headless import needs the "
            "ComfyUI source tree on sys.path"
        )
    root = Path(value).expanduser()
    if not (root / "folder_paths.py").is_file():
        pytest.skip(
            f"{COMFYUI_ROOT_ENV}={value!r} is not a ComfyUI source tree "
            "(folder_paths.py is missing)"
        )
    return str(root.resolve())


@pytest.mark.mock_integration
def test_the_package_imports_for_real_under_the_mocks():
    """P3.6: the package imports for real under the mocks - no exception.

    This is the strict version of the import smoke test: the root
    ``__init__.py`` is executed and the three node mappings are exported.
    """
    conftest = importlib.import_module("conftest")
    package = conftest.execute_package_init()
    assert sorted(package.NODE_CLASS_MAPPINGS) == PACK_NODE_IDS
    assert sorted(package.NODE_DISPLAY_NAME_MAPPINGS) == PACK_NODE_IDS


@pytest.mark.headless
def test_the_package_imports_against_the_real_comfyui_tree():
    """P3.2: the whole package imports against the real ComfyUI source tree."""
    comfyui_root = _comfyui_root()
    result = subprocess.run(
        [sys.executable, "-c", _HEADLESS_PROBE, comfyui_root, str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"the package did not import\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    sentinel = [
        line for line in result.stdout.splitlines() if line.startswith("KEEP_IMPORT_REPORT:")
    ]
    assert sentinel, (
        f"the probe produced no report\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    report = json.loads(sentinel[-1].split(":", 1)[1])
    assert report["node_ids"] == PACK_NODE_IDS, report
    assert report["display_names"]["KEEP_ModelLoader"] == "Load KEEP Models", report
    assert report["has_logger"] is True, report


@pytest.mark.headless
def test_the_headless_helper_knows_the_package():
    """The P1.4 helper resolves the package directory of this checkout."""
    comfyui_root = _comfyui_root()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[2]); import _headless; "
            "print('KEEP_ALIAS', _headless.alias_for(sys.argv[1]))",
            str(REPO_ROOT),
            str(REPO_ROOT / "tests"),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KEEP_ALIAS ComfyUI_KEEP" in result.stdout, result.stdout
    assert Path(comfyui_root).is_dir()
