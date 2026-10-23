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

P1.2 will extend this file with the package-alias bootstrap, the
``comfy_api`` / ``comfy`` / ``folder_paths`` mocks and the shared fixtures.
"""

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

# pytest imports the root ``__init__.py`` under the module name ``__init__``
# (the repository directory name is not a valid identifier, so it cannot become
# a package name).  Pre-registering that name - and the plain directory name -
# keeps pytest from executing the file.
for _alias in ("__init__", os.path.basename(_PROJECT_ROOT)):
    if _alias in sys.modules:
        continue
    _spec = importlib.util.spec_from_file_location(
        _alias,
        os.path.join(_PROJECT_ROOT, "__init__.py"),
        submodule_search_locations=[_PROJECT_ROOT],
    )
    _module = importlib.util.module_from_spec(_spec)
    _module.__path__ = [_PROJECT_ROOT]  # type: ignore[attr-defined]
    sys.modules[_alias] = _module

# ``import __init__`` must not be mistaken for a real module by anything else
# that runs in this interpreter.
sys.modules["__init__"].__spec__.submodule_search_locations = [_PROJECT_ROOT]  # type: ignore[union-attr]
assert isinstance(sys.modules["__init__"], types.ModuleType)
