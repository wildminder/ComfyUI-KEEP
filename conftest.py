"""Root conftest for ComfyUI-KEEP.

P0 note: the modernization plan assigns the full root conftest (ComfyUI mocks,
package-alias bootstrap, fixtures) to step **P1.2**.  P0 needs a *subset* of it
already, because pytest treats the repository root as a Python package (it has
an ``__init__.py``) and imports that file during collection:

    Package.setup() -> importtestmodule(<repo>/__init__.py)
    -> ImportError: attempted relative import with no known parent package

(the root ``__init__.py`` does ``from .modules import utils`` and therefore needs
ComfyUI on ``sys.path``).  ``collect_ignore`` does **not** prevent this, because
the import happens in ``Package.setup()`` and not during file collection.

The fix below pre-registers the root package in ``sys.modules`` **without
executing** it - the same trick the reference pack ComfyUI-DyPE uses.  pytest
then finds the module already imported and does not execute the file.

Two names are registered, and they are **not** interchangeable:

* ``ComfyUI_KEEP`` - the package alias the plan mandates (P1.2: "register the
  package under ``sys.modules`` alias ``ComfyUI_KEEP`` **with**
  ``submodule_search_locations`` and ``__path__`` set").  It is registered
  *without executing* the file, which is allowed only for ``modules/*``
  consumers; the root ``__init__.py`` needs ComfyUI (``folder_paths``,
  ``comfy.utils``) and those mocks are installed in P1.2.  Use
  :func:`execute_package_init` to run the real entry point (node tests,
  P3.4/P4.4) instead of the stub.
* ``__init__`` - a **pytest artifact**, not a package alias: pytest imports the
  root ``__init__.py`` under that name because the repository directory name
  is not a valid identifier.  It only exists to stop the collection-time
  import; P1.1's ``--import-mode=importlib`` removes the need for it.

P1.2 **replaces** this bootstrap (it does not merely extend it): the alias set
becomes exactly the plan's, the ComfyUI mocks are added, and the collection
artifact is dropped once the import mode is switched.
"""

import importlib.machinery
import importlib.util
import os
import sys
import types

# Regular packages, not test modules - they are imported explicitly by the
# tests (via the P1.2 alias) and must not be collected by pytest.
collect_ignore = ["__init__.py", "src", "nodes"]

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

#: The package alias mandated by the plan (P1.2).
PACKAGE_ALIAS = "ComfyUI_KEEP"


def _package_spec(alias: str) -> importlib.machinery.ModuleSpec:
    """Spec of the root ``__init__.py`` under ``alias``."""
    return importlib.util.spec_from_file_location(
        alias,
        os.path.join(_PROJECT_ROOT, "__init__.py"),
        submodule_search_locations=[_PROJECT_ROOT],
    )


def register_package_alias(alias: str = PACKAGE_ALIAS) -> types.ModuleType:
    """Register the package under ``alias`` **without executing** it.

    The module object gets ``__path__`` and ``submodule_search_locations``, so
    ``import <alias>.modules.utils`` resolves inside the repository - but the
    root ``__init__.py`` itself is not run (it needs ComfyUI).
    """
    existing = sys.modules.get(alias)
    if existing is not None:
        return existing
    module = importlib.util.module_from_spec(_package_spec(alias))
    module.__path__ = [_PROJECT_ROOT]  # type: ignore[attr-defined]
    sys.modules[alias] = module
    return module


def execute_package_init(alias: str = PACKAGE_ALIAS) -> types.ModuleType:
    """Execute the root ``__init__.py`` under ``alias`` and return the module.

    This is the path the node tests use (P1.2 mocks + P3.4/P4.4 V3
    entrypoint): the package is really executed, so ``comfy_entrypoint`` /
    ``NODE_CLASS_MAPPINGS`` and the ``logger`` the ``modules/*`` code imports
    via ``from .. import logger`` are real.  The register-without-execute stub
    is replaced.
    """
    module = importlib.util.module_from_spec(_package_spec(alias))
    sys.modules[alias] = module
    module.__spec__.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# The plan's package alias (P1.2) - registered, not executed.
register_package_alias()

# pytest's collection-time import name - an artifact of pytest's prepend import
# mode, not a package alias.  It keeps pytest from executing the root
# ``__init__.py`` while collecting the tests.
if "__init__" not in sys.modules:
    _module = importlib.util.module_from_spec(_package_spec("__init__"))
    _module.__path__ = [_PROJECT_ROOT]  # type: ignore[attr-defined]
    sys.modules["__init__"] = _module

# ``import __init__`` must not be mistaken for a real module by anything else
# that runs in this interpreter.
sys.modules["__init__"].__spec__.submodule_search_locations = [_PROJECT_ROOT]  # type: ignore[union-attr]
assert isinstance(sys.modules["__init__"], types.ModuleType)
