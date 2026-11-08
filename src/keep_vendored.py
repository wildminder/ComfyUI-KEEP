"""The pack-private namespace of the vendored third-party trees (Round 21).

pip ``_vendor`` semantics: the trees under ``modules/deps`` (``basicsr``,
``facelib``) are reachable under exactly **one** canonical name per process -
``_keep_vendor`` - never as a top-level ``basicsr`` / ``facelib`` and never
through the shared top-level ``modules`` name.

Why: the old dual-mode pattern (package-relative first, flat fallback second)
bound the shared top-level ``modules`` name whenever the repository root sat
on ``sys.path``.  On such a path, whichever root ships a ``modules/`` package
first wins - and with it, which :class:`ARCH_REGISTRY` object the pack binds
(first-importer-wins).  Under ``_keep_vendor`` the instantiation of the
vendored tree is **single by construction**: one ``sys.modules`` root, created
once, anchored to this pack's own files (``modules/deps`` next to ``src/``),
independent of the name the host registers the pack under - the private name
does not derive from the runtime parent package, so it is the same in every
import mode (the V3 package mode, the pytest alias mode and the headless
alias mode).

The module is stdlib-only (``importlib``, ``sys``, ``pathlib``): importing it
must not pull torch or ComfyUI, so the lazy-import rules of the engines (the
plan's §3 import rule, guard (f)) stay untouched.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: The reserved top-level name of the vendored trees.  Pack private - it does
#: not derive from the package name the host registers the pack under.
VENDOR_PACKAGE = "_keep_vendor"

#: The vendored tree on disk (``modules/deps`` next to ``src/``).  The private
#: root's ``__path__`` points here, so every subpackage of the trees resolves
#: from the pack's own files - also the three-dot cross-tree relative imports
#: (``facelib`` -> ``basicsr``), because both trees stay siblings under one
#: parent.
_DEPS_DIR = Path(__file__).resolve().parent.parent / "modules" / "deps"

#: The ``__init__.py`` of the vendored tree - the origin of the private root.
#: It is the identity the squatter check below compares against.
_DEPS_INIT = _DEPS_DIR / "__init__.py"


def _ensure_vendor_root() -> None:
    """Put the private root into ``sys.modules`` - exactly once.

    A private name that is already occupied by a foreign module (an origin
    that is not this pack's ``modules/deps/__init__.py``) fails **loudly**:
    two instantiations of the vendored tree would mean two ``ARCH_REGISTRY``
    objects, and a silent wrong-tree bind is the failure mode this round
    closes.
    """
    existing = sys.modules.get(VENDOR_PACKAGE)
    if existing is not None:
        origin = getattr(getattr(existing, "__spec__", None), "origin", None)
        if origin is None or Path(origin).resolve() != _DEPS_INIT.resolve():
            raise ImportError(
                f"the pack-private namespace {VENDOR_PACKAGE!r} is already "
                f"occupied by a foreign module (origin: {origin!r}) - the "
                f"vendored trees of the pack live in {_DEPS_DIR}"
            )
        return

    spec = importlib.util.spec_from_file_location(
        VENDOR_PACKAGE,
        str(_DEPS_INIT),
        submodule_search_locations=[str(_DEPS_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    # The root is registered before it is executed, so the imports inside the
    # vendored tree (and a re-entrant ``vendor()`` call) see the same root.
    sys.modules[VENDOR_PACKAGE] = module
    try:
        spec.loader.exec_module(module)  # runs modules/deps/__init__.py once
    except BaseException:
        # importlib's own semantics: a root that fails to execute does not
        # stay in ``sys.modules`` - the next call retries from scratch (and
        # fails loudly again).
        sys.modules.pop(VENDOR_PACKAGE, None)
        raise


def vendor(relative: str) -> ModuleType:
    """One module of the vendored trees, under the pack-private namespace.

    ``vendor("basicsr.utils.registry")`` returns the module the old flat
    import called ``modules.deps.basicsr.utils.registry`` - but its canonical
    name is ``_keep_vendor.basicsr.utils.registry``, so the relative imports
    **inside** the vendored tree (including the three-dot cross-tree imports
    from ``facelib`` into ``basicsr``) resolve under the same private root and
    every consumer of the process shares one instantiated tree: the second
    and later ``vendor()`` calls return cached module objects.
    """
    _ensure_vendor_root()
    return importlib.import_module(f"{VENDOR_PACKAGE}.{relative}")
