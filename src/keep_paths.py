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


def _registered_paths(name: str) -> list[str]:
    """The registered roots of a model folder (empty when unknown)."""
    if folder_paths is None:
        return []
    try:
        return [str(path) for path in folder_paths.get_folder_paths(name)]
    except Exception:  # noqa: BLE001 - an unknown folder is not an error
        return []


def register_model_folders() -> bool:
    """Register the KEEP model folders with ComfyUI's ``folder_paths``.

    The registration is idempotent: ComfyUI's ``add_model_folder_path`` does
    not add a path twice, and with ``is_default=True`` it puts the path
    **first** (the real implementation moves it to the front - see
    ``folder_paths.py:382``).  The primary root is therefore always the first
    root of a folder: the lookup (``get_full_path_or_raise``) and the
    download target both follow that order.

    Returns ``False`` when ComfyUI is not available (the pack then imports,
    but no weight can be resolved).
    """
    if folder_paths is None:
        return False

    for name in MODEL_FOLDERS:
        primary = os.path.join(folder_paths.models_dir, name)
        folder_paths.add_model_folder_path(name, primary, is_default=True)
        logger.debug(f"registered the model folder {name!r} -> {primary}")
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


def resolve_model_target(folder_name: str, file_name: str, subdir: str | None = None) -> str:
    """The path a weight file is downloaded to (the primary root).

    The parent directories are created; the file itself is not touched.
    """
    target = os.path.join(
        get_primary_folder(folder_name), model_relative_path(file_name, subdir)
    )
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return target
