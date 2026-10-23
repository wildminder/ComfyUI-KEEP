"""P0 - the root conftest's package-alias bootstrap.

The plan (P1.2) mandates the package alias ``ComfyUI_KEEP`` (with
``submodule_search_locations`` and ``__path__`` set) and requires that the
**node tests execute** the root ``__init__.py`` - the V3 entrypoint of
P3.4/P4.4.  The P0 bootstrap therefore has to

* register the package under the plan's alias - and **not** under the
  repository directory basename, which is not a valid identifier,
* offer a way to execute the real entry point instead of the
  register-without-execute stub, and
* really execute it when a ComfyUI tree is available.

The bootstrap must not be bypassed: a package alias that is registered but
never executed would silently skip the V3 entrypoint in every node test.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFTEST = REPO_ROOT / "conftest.py"

EXPECTED_ALIAS = "ComfyUI_KEEP"
EXPECTED_NODE_IDS = [
    "KEEP_FaceUpscaleImage",
    "KEEP_ModelLoader",
    "KEEP_ProcessImageSequence",
]


def _load_conftest():
    spec = importlib.util.spec_from_file_location("keep_root_conftest", CONFTEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def conftest_module():
    return _load_conftest()


@pytest.mark.unit
def test_plan_alias_is_registered(conftest_module):
    """The package is registered under the alias the plan names."""
    assert conftest_module.PACKAGE_ALIAS == EXPECTED_ALIAS
    alias_module = sys.modules.get(EXPECTED_ALIAS)
    assert alias_module is not None, f"{EXPECTED_ALIAS} is not registered"
    assert list(alias_module.__path__) == [str(REPO_ROOT)]
    search_locations = alias_module.__spec__.submodule_search_locations
    assert list(search_locations) == [str(REPO_ROOT)]


@pytest.mark.unit
def test_directory_basename_alias_is_not_registered(conftest_module):
    """The invalid-identifier alias of the old bootstrap is gone."""
    assert REPO_ROOT.name not in sys.modules, (
        f"{REPO_ROOT.name!r} must not be registered: it is not a valid "
        "identifier and is not the alias the plan specifies"
    )


@pytest.mark.unit
def test_conftest_provides_an_execute_hook(conftest_module):
    """There must be a way to execute the root __init__.py (P3.4/P4.4)."""
    assert callable(conftest_module.execute_package_init)


@pytest.mark.unit
def test_execute_package_init_runs_the_real_entrypoint(tmp_path):
    """Executing the alias really runs the root __init__.py.

    The execution needs ComfyUI on ``sys.path`` (``folder_paths``,
    ``comfy.utils``); without a ComfyUI tree the test skips with a reason.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import baseline_probe  # noqa: PLC0415 - the probe owns the root resolution

    comfyui_root = baseline_probe.resolve_comfyui_root(None)
    if comfyui_root is None:
        pytest.skip(
            "no ComfyUI source tree found: set COMFYUI_ROOT to enable the "
            "package-entrypoint execution"
        )

    out_path = tmp_path / "entrypoint.json"
    script = tmp_path / "run_entrypoint.py"
    script.write_text(
        "import importlib.util, json, sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"spec = importlib.util.spec_from_file_location('keep_root_conftest', {str(CONFTEST)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "package = module.execute_package_init()\n"
        "with open(sys.argv[1], 'w', encoding='utf-8') as handle:\n"
        "    json.dump({\n"
        "        'module_name': package.__name__,\n"
        "        'has_logger': hasattr(package, 'logger'),\n"
        "        'node_ids': sorted(getattr(package, 'NODE_CLASS_MAPPINGS', {})),\n"
        "    }, handle)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script), str(out_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        env=_environ_with_comfyui(comfyui_root),
    )
    assert out_path.is_file(), (
        f"the entrypoint script did not write its result (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["module_name"] == EXPECTED_ALIAS
    assert data["has_logger"] is True, (
        "the executed package must define the logger that modules/* imports "
        "via 'from .. import logger'"
    )
    assert data["node_ids"] == EXPECTED_NODE_IDS


def _environ_with_comfyui(comfyui_root: str) -> dict:
    import os  # noqa: PLC0415 - local import keeps the module import list short

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [comfyui_root, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return env
