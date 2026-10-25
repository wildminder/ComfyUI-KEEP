"""The model folders of the KEEP pack (P3.5).

The pack needs three model folders:

* ``keep_models`` - the KEEP checkpoints; the ``KEEP`` **sub-directory** is
  part of the on-disk layout and is preserved (``models/keep_models/KEEP``),
* ``facedetection`` - the facelib detection / parsing weights,
* ``gmflow`` - the optional GMFlow checkpoint (P5.1).

The folders are registered with ComfyUI's ``folder_paths``, so every weight
path is resolved through the host (primary root first, then the extra model
roots).  This module is the **only** place that knows about
``folder_paths.models_dir`` - the engines never concatenate a model path
themselves.

The **extra model roots** of the host are discovered from the
``extra_model_paths.yaml`` of the user (:func:`apply_extra_model_paths`):
the explicit keys (``keep_models:`` ...) are read by ComfyUI's own loader,
and the layout convention (``<base_path>/models/<pack folder>``) covers the
users whose yaml only describes the standard folders.  A weight that lives
in such a root is **found** where it is - and a missing one is **downloaded**
next to the pack's other weights (:func:`resolve_download_root`).

The import of ``folder_paths`` is guarded: the pack must import without a
ComfyUI server (the test harness does exactly that).

This module only **resolves** the paths.  The download of a weight is not
here: since **P5.2** it lives in :mod:`src.keep_download` (manifest-driven,
sha256-verified) - the helper that used to live here is gone.
"""

from __future__ import annotations

import os

try:  # ComfyUI provides the top-level ``folder_paths`` module
    import folder_paths
except ImportError:  # the pack imports without ComfyUI
    folder_paths = None  # type: ignore[assignment]

from .keep_logger import logger

#: The model folders of the pack (registered with ComfyUI).
MODEL_FOLDERS = ("keep_models", "facedetection", "gmflow")

#: The name of the yaml ComfyUI reads its extra model paths from.
EXTRA_MODEL_PATHS_FILENAME = "extra_model_paths.yaml"

#: The yaml files that were applied - ``path -> (mtime, roots)``.  The mtime
#: is the cache key: the yaml is parsed **once per change**.
_APPLIED_EXTRA_MODEL_PATHS: dict[str, tuple[float, tuple[str, ...]]] = {}


def _registered_paths(name: str) -> list[str]:
    """The registered roots of a model folder (empty when unknown)."""
    if folder_paths is None:
        return []
    try:
        return [str(path) for path in folder_paths.get_folder_paths(name)]
    except Exception:  # noqa: BLE001 - an unknown folder is not an error
        return []


def _pack_roots() -> list[str]:
    """Every registered root of the pack's folders."""
    roots: list[str] = []
    for name in MODEL_FOLDERS:
        roots.extend(_registered_paths(name))
    return roots


def _registry() -> dict | None:
    """The registry of ComfyUI's model folders (``None`` when absent)."""
    if folder_paths is None:
        return None
    return getattr(folder_paths, "folder_names_and_paths", None)


def _snapshot_registry(registry: dict) -> dict:
    """A copy of the registry - the path lists are copied, not shared.

    The shape of the registry differs between the host
    (``([paths], {extensions})``) and the test harness
    (``[(path, is_default), ...]``); both are copied the same way.
    """
    snapshot: dict = {}
    for name, value in registry.items():
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], list):
            snapshot[name] = (list(value[0]), value[1])
        elif isinstance(value, list):
            snapshot[name] = list(value)
        else:
            snapshot[name] = value
    return snapshot


def _restore_registry(registry: dict, snapshot: dict, keep: set[str]) -> None:
    """Put the registry back - only the pack's own folders may have grown."""
    for name in list(registry):
        if name in keep:
            continue
        if name in snapshot:
            registry[name] = snapshot[name]
        else:
            del registry[name]


def _default_extra_model_paths() -> str | None:
    """The yaml ComfyUI itself reads - the one next to ``folder_paths.py``.

    The derivation is the one of the server (``main.py``): the yaml sits
    next to the module that owns the model roots.  The path is never
    hardcoded and never written into a tracked file.
    """
    module_file = getattr(folder_paths, "__file__", None)
    if not module_file:
        return None
    return os.path.join(
        os.path.dirname(os.path.realpath(module_file)), EXTRA_MODEL_PATHS_FILENAME
    )


def _yaml_sections(yaml_path: str) -> dict:
    """The sections of the yaml - ``{}`` when it cannot be read."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is a dependency
        logger.debug(f"PyYAML is not available: {exc}")
        return {}
    try:
        with open(yaml_path, "r", encoding="utf-8") as stream:
            sections = yaml.safe_load(stream)
    except Exception as exc:  # noqa: BLE001 - a broken yaml is not an error
        logger.debug(f"the extra model paths yaml {yaml_path} is unreadable: {exc}")
        return {}
    return sections if isinstance(sections, dict) else {}


def _section_base_path(section: dict, yaml_dir: str) -> str | None:
    """The ``base_path`` of a yaml section - normalised like ComfyUI does.

    The normalisation mirrors ``utils/extra_config.py``: the variables and
    ``~`` are expanded, a relative path is resolved against the directory of
    the yaml and the result is normalised.
    """
    base_path = section.get("base_path")
    if not base_path:
        return None
    base_path = os.path.expandvars(os.path.expanduser(str(base_path)))
    if not os.path.isabs(base_path):
        base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
    return os.path.normpath(base_path)


def _apply_explicit_keys(yaml_path: str) -> None:
    """Let the loader of the host read the yaml (the explicit folder keys).

    ``utils.extra_config.load_extra_path_config`` is the authoritative
    interpretation of the file - its normalisation (variables, ``~``,
    relative paths, multi-line lists) is the one of the server.  The loader
    is **not** scoped to a folder name: it touches every key of every
    section.  The registry is therefore restored afterwards, so the pack can
    only ever add a root to its own three folders.
    """
    registry = _registry()
    if registry is None:
        return
    try:
        from utils.extra_config import load_extra_path_config  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - no ComfyUI, no explicit keys
        logger.debug(f"the extra_config of ComfyUI is not available: {exc}")
        return

    snapshot = _snapshot_registry(registry)
    try:
        load_extra_path_config(yaml_path)
    except Exception as exc:  # noqa: BLE001 - a broken yaml is not an error
        logger.debug(f"the extra model paths of {yaml_path} are unreadable: {exc}")
    finally:
        _restore_registry(registry, snapshot, set(MODEL_FOLDERS))


def _apply_layout_convention(yaml_path: str) -> list[str]:
    """Register ``<base_path>/models/<pack folder>`` of every section.

    The standard layout of ComfyUI is ``<base_path>/models/<folder>``
    (``folder_paths.models_dir`` is ``<base_path>/models``), so a yaml that
    only describes the standard folders still points at the place where the
    weights of such a user live.  The roots are **appended**
    (``is_default=False``): the primary root stays the first one.
    """
    sections = _yaml_sections(yaml_path)
    if not sections:
        return []
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    roots: list[str] = []
    for section in sections.values():
        if not isinstance(section, dict):
            continue
        base_path = _section_base_path(section, yaml_dir)
        if not base_path:
            continue
        for name in MODEL_FOLDERS:
            root = os.path.normpath(os.path.join(base_path, "models", name))
            folder_paths.add_model_folder_path(name, root, is_default=False)
            roots.append(root)
            logger.debug(f"registered the extra model root {name!r} -> {root}")
    return roots


def apply_extra_model_paths(
    yaml_path: str | None = None, *, force: bool = False
) -> list[str]:
    """Register the extra model roots of the pack (the yaml of the host).

    ComfyUI reads ``extra_model_paths.yaml`` itself; the pack reads the
    **same** file, so the weights of a user who keeps them in an extra model
    root are found where they are instead of being downloaded into the
    primary root again.

    Two discovery paths:

    * the **explicit keys** - ``keep_models:``, ``facedetection:`` and
      ``gmflow:`` are read by the loader of the host, so the pack follows
      the interpretation of ComfyUI;
    * the **layout convention** - every section with a ``base_path`` also
      points at ``<base_path>/models/<pack folder>``, which is where the
      weights of such a user really live.

    The yaml is parsed **once per change** (its mtime is the cache key);
    ``force=True`` re-applies it.  A missing yaml, a broken yaml or a
    missing ComfyUI is a debug message and never an error - the pack then
    works with its primary root alone.

    Returns the extra roots that were registered for the pack's folders.
    """
    if folder_paths is None:
        logger.debug("no ComfyUI: the extra model paths are not applied")
        return []

    if yaml_path:
        path = os.path.abspath(os.path.expanduser(yaml_path))
    else:
        path = _default_extra_model_paths()
    if not path:
        logger.debug("the path of the extra model paths yaml is unknown")
        return []

    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        logger.debug(f"no extra model paths yaml at {path}: {exc}")
        return []

    cached = _APPLIED_EXTRA_MODEL_PATHS.get(path)
    unchanged = cached is not None and cached[0] == mtime
    if unchanged and not force:
        known = set(_pack_roots())
        if all(root in known for root in cached[1]):
            return list(cached[1])
        # the registry was reset (the test harness does that per test) -
        # the yaml is applied again
        logger.debug(f"the extra model paths of {path} are applied again")

    before = {name: _registered_paths(name) for name in MODEL_FOLDERS}
    _apply_explicit_keys(path)
    explicit_roots = [
        root
        for name in MODEL_FOLDERS
        for root in _registered_paths(name)
        if root not in before[name]
    ]
    layout_roots = _apply_layout_convention(path)

    roots = list(dict.fromkeys(explicit_roots + layout_roots))
    if unchanged:
        # a re-application of the same file: the roots of the first run are
        # kept (they are registered already and are not added a second time)
        roots = list(dict.fromkeys(list(cached[1]) + roots))
    _APPLIED_EXTRA_MODEL_PATHS[path] = (mtime, tuple(roots))
    logger.debug(f"the extra model paths of {path} were applied: {roots}")
    return roots


def register_model_folders() -> bool:
    """Register the KEEP model folders with ComfyUI's ``folder_paths``.

    The registration is idempotent: ComfyUI's ``add_model_folder_path`` does
    not add a path twice, and with ``is_default=True`` it puts the path
    **first** (the real implementation moves it to the front - see
    ``folder_paths.py:382``).  The primary root is therefore always the first
    root of a folder: the lookup (``get_full_path_or_raise``) follows that
    order, the download target follows the root that holds the pack
    (:func:`resolve_download_root`).

    The extra model roots of the host (the yaml of the user) are applied at
    the end, so every lookup sees them.

    Returns ``False`` when ComfyUI is not available (the pack then imports,
    but no weight can be resolved).
    """
    if folder_paths is None:
        return False

    primaries = {}
    for name in MODEL_FOLDERS:
        primary = os.path.join(folder_paths.models_dir, name)
        primaries[name] = primary
        folder_paths.add_model_folder_path(name, primary, is_default=True)
        logger.debug(f"registered the model folder {name!r} -> {primary}")

    # the extra roots of the host (the yaml of the user)
    apply_extra_model_paths()

    # the primary root stays the **first** root: it is registered once more,
    # which moves it to the front when the yaml asked for a "default" root
    for name, primary in primaries.items():
        folder_paths.add_model_folder_path(name, primary, is_default=True)
    return True


def get_model_folder_paths(name: str) -> list[str]:
    """The roots of a model folder - the **primary root first**."""
    register_model_folders()
    return _registered_paths(name)


def get_primary_folder(name: str) -> str:
    """The primary root of a model folder."""
    roots = get_model_folder_paths(name)
    if not roots:
        raise FileNotFoundError(
            f"the model folder {name!r} is not registered - the pack cannot "
            "resolve its weights without ComfyUI's folder_paths"
        )
    return roots[0]


def model_relative_path(file_name: str, subdir: str | None = None) -> str:
    """The path of a weight file inside a model folder."""
    return os.path.join(subdir, file_name) if subdir else file_name


def find_model_file(folder_name: str, file_name: str, subdir: str | None = None) -> str | None:
    """An **existing** weight file in any registered root (primary first).

    The lookup goes through ``folder_paths.get_full_path_or_raise``, so every
    model root the host knows (the extra model paths of ``extra_model_paths``
    included) is searched - the primary root first.  ``None`` when the file
    is in none of the roots.
    """
    if folder_paths is None:
        return None
    relative = model_relative_path(file_name, subdir)
    try:
        # the registration puts the primary root into the search list
        register_model_folders()
        return folder_paths.get_full_path_or_raise(folder_name, relative)
    except (FileNotFoundError, OSError, KeyError, ValueError):
        # not in any root - the caller decides (download or fail)
        return None


def resolve_model_file(folder_name: str, file_name: str, subdir: str | None = None) -> str:
    """The path of an **existing** weight file (primary root first).

    Raises ``FileNotFoundError`` (with the searched roots) when the file is
    not in any of the registered roots.
    """
    found = find_model_file(folder_name, file_name, subdir)
    if found is not None:
        return found
    if folder_paths is None:
        raise FileNotFoundError(
            f"cannot resolve {file_name!r}: ComfyUI's folder_paths is not available"
        )
    raise FileNotFoundError(
        f"the weight {file_name!r} is not in any root of the model folder "
        f"{folder_name!r}: {get_model_folder_paths(folder_name)}"
    )


def resolve_model_root(folder_name: str, file_names) -> str:
    """The registered root that holds the given weight files (primary first).

    Some consumers (the facelib helper) build the weight paths from a
    **single** root, so the root has to be the one the weights really live
    in - an extra model root included.  The primary root is the fallback
    (it is where the download puts a missing weight).
    """
    roots = get_model_folder_paths(folder_name)
    if not roots:
        raise FileNotFoundError(
            f"the model folder {folder_name!r} is not registered - the pack "
            "cannot resolve its weights without ComfyUI's folder_paths"
        )
    names = list(file_names)
    for root in roots:
        if all(os.path.isfile(os.path.join(root, name)) for name in names):
            return root
    return roots[0]


def resolve_download_root(folder_name: str, sibling_paths=()) -> str:
    """The root a **missing** weight is downloaded to - the one of the pack.

    The **lookup** keeps the primary root first (``get_full_path_or_raise``),
    but the **download target** follows the root that already holds the
    pack: the root with the most of the pack's files wins, a tie goes to the
    earliest root (the primary one) and when no root holds any of them the
    primary root is the target.

    ``sibling_paths`` are the paths of the pack's own files inside the
    folder (the relative paths of the manifest) - the siblings of the file
    that is downloaded.
    """
    roots = get_model_folder_paths(folder_name)
    if not roots:
        raise FileNotFoundError(
            f"the model folder {folder_name!r} is not registered - the pack "
            "cannot resolve its weights without ComfyUI's folder_paths"
        )
    siblings = [str(relative) for relative in sibling_paths if relative]
    if not siblings:
        return roots[0]

    best_root = roots[0]
    best_count = -1
    for root in roots:
        count = sum(
            1 for relative in siblings if os.path.isfile(os.path.join(root, relative))
        )
        if count > best_count:
            best_root = root
            best_count = count
    return best_root


def resolve_model_target(
    folder_name: str,
    file_name: str,
    subdir: str | None = None,
    sibling_paths=(),
) -> str:
    """The path a weight file is downloaded to.

    The target is the root that holds the pack
    (:func:`resolve_download_root`); without ``sibling_paths`` that is the
    primary root - the behaviour of the earlier versions of the pack.

    The parent directories are created; the file itself is not touched.
    """
    root = resolve_download_root(folder_name, sibling_paths)
    target = os.path.join(root, model_relative_path(file_name, subdir))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return target
