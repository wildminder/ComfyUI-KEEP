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
"""

from __future__ import annotations

import hashlib
import os

from torch.hub import download_url_to_file

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


def load_file_from_url_comfy(url, folder_name, file_name=None, subdir=None, sha256=None, progress=True):
    """Download a weight file into a registered model folder.

    A file that is already there is kept - its hash is checked when one is
    known (a mismatch triggers a re-download).  The lookup covers **every**
    registered root of the folder (the primary root first), so a weight that
    lives in an extra model root is used where it is instead of being
    downloaded again.  A missing weight is downloaded to the primary root
    (:func:`resolve_model_target`), so the on-disk layout of the pack
    (including the ``keep_models/KEEP`` sub-directory) is preserved.
    """
    if file_name is None:
        file_name = os.path.basename(url)
        if '?' in file_name:
            file_name = file_name.split('?')[0]

    existing = find_model_file(folder_name, file_name, subdir)
    if existing is not None:
        if sha256:
            with open(existing, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash == sha256:
                logger.debug(f"File {file_name} already exists and hash matches. Skipping download.")
                return existing
            logger.warning(f"File {file_name} exists but hash mismatch. Redownloading.")
        else:
            logger.debug(f"File {file_name} already exists. Skipping download (no hash check).")
            return existing

    cached_file = resolve_model_target(folder_name, file_name, subdir)

    # The weight is not in any root - but the download target may exist
    # anyway (a file the host does not know as a model file).  It is kept
    # when its hash is not known, and re-downloaded when the hash does not
    # match.
    if os.path.exists(cached_file):
        if not sha256:
            logger.debug(f"File {file_name} already exists. Skipping download (no hash check).")
            return cached_file
        with open(cached_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash == sha256:
            logger.debug(f"File {file_name} already exists and hash matches. Skipping download.")
            return cached_file
        logger.warning(f"File {file_name} exists but hash mismatch. Redownloading.")

    logger.info(f'Downloading: "{file_name}" from {url} to {cached_file}')

    try:
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    except Exception as e:
        logger.error(f"Error downloading {url} using torch.hub.download_url_to_file: {e}")
        if os.path.exists(cached_file):
            os.remove(cached_file)
        raise

    if sha256:
        with open(cached_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != sha256:
            if os.path.exists(cached_file):
                os.remove(cached_file)
            raise ValueError(f"Downloaded file {file_name} hash mismatch. Expected {sha256}, got {file_hash}.")

    return cached_file
