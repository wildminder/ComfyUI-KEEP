"""P3 review fix - the engines import **without** a ComfyUI server.

The plan's §3 import rule (plan:163-164) says:

    "``comfy.*`` imports inside ``src/`` are lazy (function-level) so the
    pack imports without a ComfyUI server"

The static half of the rule is guard (f) in ``tests/test_structure.py``.
This file is the **runtime** half: a probe blocks every ``comfy`` /
``comfy_api`` import and imports the engines.  A module-level ``comfy``
import (or a vendored module that imports ``comfy`` eagerly) makes the
probe fail with an ``ImportError``.

The probe registers the repository under an alias **without** executing the
root ``__init__.py`` - the engines are what is under test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The probe: every ``comfy`` import is blocked, then the engines are
#: imported.  ``KEEP_NO_COMFY_IMPORT True`` is the proof.
_NO_COMFY_PROBE = """
import importlib.util
import sys


class _ComfyBlocker:
    # Every ``comfy`` / ``comfy_api`` import fails.
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("comfy", "comfy_api"):
            raise ImportError("comfy is blocked: " + fullname)
        return None


sys.meta_path.insert(0, _ComfyBlocker())

root = sys.argv[1]
sys.path.insert(0, root)

alias = "ComfyUI_KEEP_NO_COMFY"
spec = importlib.util.spec_from_file_location(
    alias, root + "/__init__.py", submodule_search_locations=[root]
)
sys.modules[alias] = importlib.util.module_from_spec(spec)

for name in (
    "keep_model_configs",
    "keep_paths",
    "keep_image_utils",
    "keep_progress",
    "keep_processor",
    "keep_model_loader",
):
    importlib.import_module(alias + ".src." + name)

print("KEEP_NO_COMFY_IMPORT True")
"""

#: The engines of the pack - the ones the rule covers.
ENGINE_MODULES = (
    "keep_model_configs",
    "keep_paths",
    "keep_image_utils",
    "keep_progress",
    "keep_processor",
    "keep_model_loader",
)


@pytest.mark.unit
def test_the_engines_import_without_a_comfy_server():
    """The engines import while every ``comfy`` import is blocked."""
    result = subprocess.run(
        [sys.executable, "-c", _NO_COMFY_PROBE, str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "the engines must import without a ComfyUI server\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "KEEP_NO_COMFY_IMPORT True" in result.stdout, result.stdout


@pytest.mark.unit
def test_the_probe_would_catch_a_module_level_comfy_import(tmp_path):
    """The probe is not vacuous: a module-level ``comfy`` import fails it.

    The probe is run against a **copy** of the repository in which
    ``src/keep_paths.py`` imports ``comfy`` at module level - the probe has
    to fail there.
    """
    import shutil

    copy_root = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT,
        copy_root,
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
    target = copy_root / "src" / "keep_paths.py"
    text = target.read_text(encoding="utf-8")
    # the import goes *after* the ``__future__`` import (it has to stay first)
    text = text.replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\n\nfrom comfy import model_management\n",
        1,
    )
    target.write_text(text, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-c", _NO_COMFY_PROBE, str(copy_root)],
        cwd=str(copy_root),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode != 0, (
        "the probe must fail on a module-level comfy import\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "KEEP_NO_COMFY_IMPORT True" not in result.stdout, result.stdout
    assert "comfy is blocked" in result.stderr, result.stderr
