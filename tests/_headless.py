"""Headless bootstrap helper for ComfyUI-KEEP (P1.4).

"Headless" means: drive the real node code without a ComfyUI server.  That
needs four things, and this helper provides them:

1. **the ComfyUI source tree on ``sys.path``** - resolved from the
   ``COMFYUI_ROOT`` environment variable (or an explicit argument).  There are
   **no hardcoded paths** in this module: a helper that only works on the
   machine of its author is a test that passes for the wrong reason.
2. **the extra model paths** - ``folder_paths`` does *not* read
   ``extra_model_paths.yaml`` at import time; the server does it in
   ``main.py``.  :func:`load_extra_model_paths` does the same, so the test
   sees the same model tree as the server.
3. **the package under an importable alias** - a custom-node folder is called
   ``ComfyUI-KEEP``, which is not a valid Python identifier; the package is
   registered under an alias with ``__path__`` and
   ``submodule_search_locations`` set.
4. **optional preview stubs** - the ``ui.*`` preview helpers read the server's
   execution context and raise outside a server.  Only the *preview* is
   stubbed; generation, patching and offload stay real.

Every failure is an :class:`HeadlessBootstrapError` whose message names the
environment variable to set.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import types
from pathlib import Path

#: Environment variable pointing at the ComfyUI source tree.
COMFYUI_ROOT_ENV = "COMFYUI_ROOT"

#: Environment variable pointing at the directory with the model weights.
MODEL_DIR_ENV = "KEEP_MODEL_DIR"

#: The preview helpers that are replaced by :func:`stub_ui`.
PREVIEW_HELPERS = ("PreviewImage", "PreviewAudio")


class HeadlessBootstrapError(RuntimeError):
    """Raised when the headless environment cannot be prepared."""


def _env_path(name: str) -> str:
    """The value of an environment variable (empty string when unset/blank)."""
    return (os.environ.get(name) or "").strip()


def resolve_comfyui_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The ComfyUI source tree.

    Resolution order: the explicit argument, then the ``COMFYUI_ROOT``
    environment variable.  There is no fallback default - a wrong default
    would make the tests pass against the wrong tree.
    """
    if explicit:
        root = Path(explicit).expanduser()
        if not (root / "folder_paths.py").is_file():
            raise HeadlessBootstrapError(
                f"{root} is not a ComfyUI source tree (folder_paths.py is missing). "
                f"Point {COMFYUI_ROOT_ENV} at a ComfyUI checkout."
            )
        return root.resolve()

    value = _env_path(COMFYUI_ROOT_ENV)
    if not value:
        raise HeadlessBootstrapError(
            f"{COMFYUI_ROOT_ENV} is not set. Set {COMFYUI_ROOT_ENV}=<path to the "
            f"ComfyUI checkout> to run the headless tests."
        )
    root = Path(value).expanduser()
    if not (root / "folder_paths.py").is_file():
        raise HeadlessBootstrapError(
            f"{COMFYUI_ROOT_ENV}={value!r} is not a ComfyUI source tree "
            f"(folder_paths.py is missing)."
        )
    return root.resolve()


def resolve_model_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The directory that holds the model weights.

    Resolution order: the explicit argument, then the ``KEEP_MODEL_DIR``
    environment variable.  The directory does not have to exist yet - the
    caller decides whether a missing directory is fatal.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()

    value = _env_path(MODEL_DIR_ENV)
    if not value:
        raise HeadlessBootstrapError(
            f"{MODEL_DIR_ENV} is not set. Set {MODEL_DIR_ENV}=<absolute path to the "
            f"directory with the KEEP weights> to run the model tests."
        )
    return Path(value).expanduser().resolve()


def alias_for(package_dir: str | os.PathLike[str]) -> str:
    """A valid Python alias for a package directory (``ComfyUI-KEEP`` -> ``ComfyUI_KEEP``)."""
    return re.sub(r"\W", "_", Path(package_dir).name)


def register_package(
    package_dir: str | os.PathLike[str],
    alias: str | None = None,
    execute: bool = False,
) -> types.ModuleType:
    """Register a package directory under an importable alias.

    The module gets ``__path__`` and ``submodule_search_locations``, so its
    subpackages resolve inside the directory.  With ``execute=False`` the
    ``__init__.py`` is *not* run (the package can then be imported without
    ComfyUI); with ``execute=True`` the real entry point runs.
    """
    package_dir = Path(package_dir).expanduser().resolve()
    init_file = package_dir / "__init__.py"
    if not init_file.is_file():
        raise HeadlessBootstrapError(
            f"{package_dir} is not a package: {init_file} does not exist."
        )

    alias = alias or alias_for(package_dir)
    spec = importlib.util.spec_from_file_location(
        alias,
        str(init_file),
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise HeadlessBootstrapError(f"cannot build an import spec for {init_file}")

    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    sys.modules[alias] = module
    if execute:
        spec.loader.exec_module(module)
    return module


def load_extra_model_paths(
    comfyui_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Load ComfyUI's ``extra_model_paths.yaml`` the way the server does.

    ``folder_paths`` resolves only the default ``models/`` tree on its own;
    the server applies the extra paths in ``main.py`` through
    ``utils.extra_config.load_extra_path_config``.  Returns ``True`` when a
    configuration was loaded, ``False`` when there is none.
    """
    root = Path(comfyui_root).expanduser().resolve()
    config = Path(config_path).expanduser() if config_path else root / "extra_model_paths.yaml"
    if not config.is_file():
        return False

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # The real module is ``utils.extra_config`` (top-level ``utils``), not
    # ``comfy.utils.extra_config`` - ``comfy/utils.py`` has no submodules.
    import utils.extra_config as extra_config  # noqa: PLC0415 - needs the root on sys.path

    extra_config.load_extra_path_config(str(config))
    return True


def stub_ui(ui_module: types.ModuleType | None = None) -> list[str]:
    """Replace the preview helpers of the ``ui`` module with no-ops.

    Only the preview is stubbed - everything else (generation, patching,
    offload) stays real.  Returns the names that were stubbed.
    """
    if ui_module is None:
        from comfy_api.latest import ui as ui_module  # noqa: PLC0415 - optional dependency

    stubbed: list[str] = []
    for name in PREVIEW_HELPERS:
        if not hasattr(ui_module, name):
            continue
        setattr(ui_module, name, _make_noop(name))
        stubbed.append(name)
    return stubbed


def _make_noop(name: str):
    """A no-op replacement for a preview helper."""

    def _noop(*args, **kwargs):
        return None

    _noop.__name__ = name
    _noop.__qualname__ = name
    return _noop
