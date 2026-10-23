"""P3.4 - the package entry point and the logger.

The root ``__init__.py`` is the whole public surface of the pack while it
registers as **V1** (P4.4 switches it to the V3 ``comfy_entrypoint``):

* it sets up the logger (now ``src/keep_logger.py``),
* it exports the node mappings.

The tests assert the entry point is *boring* (one mapping style, no side
effects beyond the logger) and that the P1.2 logger constraint is **lifted**:
the engines import without the package entry point being executed.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INIT_FILE = REPO_ROOT / "__init__.py"

PACK_NODE_IDS = [
    "KEEP_FaceUpscaleImage",
    "KEEP_ModelLoader",
    "KEEP_ProcessImageSequence",
]

#: The probe: the package is registered under an alias **without** executing
#: the root ``__init__.py`` and **without** the placeholder logger of the test
#: harness.  The engines must import all the same - that is the lifted
#: constraint.
_NO_INIT_PROBE = """
import importlib.util, sys

root = sys.argv[1]
sys.path.insert(0, root)
import conftest  # the ComfyUI mocks

alias = "ComfyUI_KEEP_NO_INIT"
spec = importlib.util.spec_from_file_location(
    alias, root + "/__init__.py", submodule_search_locations=[root]
)
module = importlib.util.module_from_spec(spec)
sys.modules[alias] = module
# the alias has **no** logger attribute - the engines must not need it

importlib.import_module(alias + ".src.keep_model_loader")
importlib.import_module(alias + ".src.keep_processor")
importlib.import_module(alias + ".src.keep_model_configs")
importlib.import_module(alias + ".nodes")
print("KEEP_CONSTRAINT_LIFTED True")
"""


@pytest.mark.mock_integration
def test_the_package_imports_under_the_mocks():
    """The entry point executes and exports the three node mappings."""
    conftest = importlib.import_module("conftest")
    package = conftest.execute_package_init()
    assert sorted(package.NODE_CLASS_MAPPINGS) == PACK_NODE_IDS
    assert package.NODE_DISPLAY_NAME_MAPPINGS["KEEP_FaceUpscaleImage"] == (
        "KEEP Single Image"
    )


@pytest.mark.mock_integration
def test_the_logger_has_exactly_one_handler():
    """The logger of the package owns exactly one handler."""
    conftest = importlib.import_module("conftest")
    package = conftest.execute_package_init()
    logger_module = importlib.import_module("ComfyUI_KEEP.src.keep_logger")
    assert package.logger is logger_module.logger
    assert len(package.logger.handlers) == 1, package.logger.handlers
    assert package.logger.level == 20  # INFO


@pytest.mark.mock_integration
def test_the_entry_point_defines_exactly_one_mapping_style():
    """Either the V1 mappings or the V3 entry point - never both."""
    import ast

    tree = ast.parse(INIT_FILE.read_text(encoding="utf-8"), filename=str(INIT_FILE))
    defines_entrypoint = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "comfy_entrypoint"
        for node in tree.body
    )
    assert not defines_entrypoint, (
        "the V3 entry point arrives with P4.4 - until then the package "
        "registers as V1"
    )
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "NODE_CLASS_MAPPINGS" in imported
    assert "NODE_DISPLAY_NAME_MAPPINGS" in imported


@pytest.mark.unit
def test_the_engines_import_without_the_package_entry_point():
    """P3.4: the ``from .. import logger`` constraint is lifted."""
    result = subprocess.run(
        [sys.executable, "-c", _NO_INIT_PROBE, str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"the engines need the package entry point\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "KEEP_CONSTRAINT_LIFTED True" in result.stdout, result.stdout
