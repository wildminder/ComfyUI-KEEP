"""The manifest-driven downloader of the KEEP pack (P5.2).

Every weight of the pack comes from the ``WEIGHT_MANIFEST``
(:mod:`src.keep_model_configs`): the manifest knows the URL, the sha256 and the
on-disk destination of a file, and this module turns that into an actual file
on disk.

The rules are:

* the destination is resolved through :mod:`src.keep_paths` - the registered
  model folders of ComfyUI, the **primary root first**, so a weight that lives
  in an extra model root is used where it is instead of being downloaded again;
* the on-disk layout is the **legacy** one (``models/keep_models/KEEP``,
  ``models/facedetection``, ``models/gmflow``);
* a downloaded file is verified against the sha256 of the manifest - a mismatch
  deletes the file and downloads it **once** more;
* a file whose manifest entry carries **no** hash (the Asian checkpoint, which
  is not published upstream) is downloaded **without** a verification, and the
  downloader says so loudly;
* a failed download leaves **no** partial file behind and raises an error that
  names the URL and the destination.

The transport is ``torch.hub.download_url_to_file``; the module level name is
the seam the tests replace (no network in the test suite).
"""

from __future__ import annotations

import hashlib
import os

from torch.hub import download_url_to_file

from . import keep_paths
from .keep_logger import logger
from .keep_model_configs import WEIGHT_MANIFEST, WeightEntry

#: The names of the entries that were already reported as "no hash" - the
#: warning is loud, but it must not be repeated for every call.
_WARNED_UNVERIFIED: set[str] = set()


def sha256_of(path: str) -> str:
    """The sha256 of a file (streamed - the weights are large)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_path(entry: WeightEntry) -> str:
    """The path the entry is downloaded to (the primary root + the layout)."""
    return keep_paths.resolve_model_target(
        entry.dest_folder, entry.name, entry.subdir
    )


def find_weight(entry: WeightEntry) -> str | None:
    """An **existing** file of the entry (any registered root, primary first)."""
    return keep_paths.find_model_file(entry.dest_folder, entry.name, entry.subdir)


def verify(entry: WeightEntry, path: str) -> bool:
    """Is the file at ``path`` the file the manifest describes?

    ``True`` when the entry has no hash (nothing to compare - the caller warns
    about it), ``True`` when the hash matches, ``False`` on a mismatch.
    """
    if not entry.sha256:
        return True
    if not os.path.isfile(path):
        return False
    return sha256_of(path) == entry.sha256


def _warn_unverified(entry: WeightEntry) -> None:
    """The loud warning for an entry without a hash."""
    if entry.name in _WARNED_UNVERIFIED:
        return
    _WARNED_UNVERIFIED.add(entry.name)
    logger.warning(
        f"KEEP: {entry.name} has no verified sha256 in the manifest "
        f"(unverified={entry.unverified}, available={entry.available}) - the "
        "download is NOT verified. Do not trust the file; place it manually "
        f"from a source you trust. The expected destination is "
        f"{entry.dest_folder}/{entry.subdir}/{entry.name}."
    )


def _remove_partials(target: str) -> None:
    """Remove the target and the temporary file of the transport."""
    for path in (target, target + ".tmp"):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:  # pragma: no cover - a locked file
                logger.warning(f"KEEP: the partial file {path} could not be removed: {exc}")


def _fetch(entry: WeightEntry, target: str, progress: bool) -> None:
    """The transport call (the seam the tests replace)."""
    download_url_to_file(entry.url, target, hash_prefix=None, progress=progress)


def download_weight(entry: WeightEntry, progress: bool = True) -> str:
    """Download the entry to its destination and verify it.

    Returns the path of the downloaded file.  Raises ``FileNotFoundError`` when
    the download fails (the message names the URL and the destination) and
    ``ValueError`` when the file does not match the manifest hash twice.
    """
    target = target_path(entry)

    # a file that is already there is kept when it is the right one
    existing = find_weight(entry)
    if existing is not None:
        if verify(entry, existing):
            if not entry.sha256:
                _warn_unverified(entry)
            logger.debug(f"KEEP: {entry.name} is already present ({existing})")
            return existing
        logger.warning(
            f"KEEP: the sha256 of {existing} does not match the manifest - the "
            "file is deleted and downloaded again"
        )
        _remove_partials(existing)

    for attempt in (1, 2):
        logger.info(f"KEEP: downloading {entry.name} from {entry.url} to {target}")
        try:
            _fetch(entry, target, progress)
        except Exception as exc:  # noqa: BLE001 - the error is re-raised below
            _remove_partials(target)
            raise FileNotFoundError(
                f"KEEP: the weight {entry.name!r} could not be downloaded.\n"
                f"  url:  {entry.url}\n"
                f"  dest: {target}\n"
                f"  reason: {exc}\n"
                "Download the file manually into the destination and try again."
            ) from exc

        if not entry.sha256:
            _warn_unverified(entry)
            return target

        actual = sha256_of(target)
        if actual == entry.sha256:
            return target

        logger.warning(
            f"KEEP: the sha256 of the downloaded {entry.name} does not match "
            f"the manifest (expected {entry.sha256}, got {actual})"
        )
        _remove_partials(target)
        if attempt == 2:
            raise ValueError(
                f"KEEP: the downloaded {entry.name!r} does not match the "
                f"manifest hash (expected {entry.sha256}, got {actual}) - the "
                f"file was deleted. url: {entry.url}, dest: {target}"
            )
    raise AssertionError("unreachable")  # pragma: no cover


def ensure_weight(entry: WeightEntry, progress: bool = True) -> str | None:
    """The path of the weight - the cached file, or a fresh download.

    ``None`` when the entry is **optional** and cannot be provided; a
    ``FileNotFoundError`` when a **required** entry cannot be provided.
    """
    if not entry.available:
        # The asset is not published upstream.  A file that the user placed
        # manually is used, anything else is an actionable error.
        existing = find_weight(entry)
        if existing is not None:
            if verify(entry, existing):
                logger.info(
                    f"KEEP: {entry.name} is not published upstream - the manually "
                    f"placed file {existing} is used"
                )
                return existing
        if entry.required:
            raise FileNotFoundError(
                f"KEEP: the weight {entry.name!r} is not published upstream "
                f"(HTTP 404) and is not installed.\n"
                f"  url:  {entry.url}\n"
                f"  dest: {target_path(entry)}\n"
                "Place the file manually at the destination and try again."
            )
        logger.warning(
            f"KEEP: the optional weight {entry.name!r} is not published "
            f"upstream (HTTP 404) - it is skipped. url: {entry.url}"
        )
        return None

    try:
        return download_weight(entry, progress=progress)
    except (FileNotFoundError, ValueError):
        if entry.required:
            raise
        logger.warning(
            f"KEEP: the optional weight {entry.name!r} could not be provided - "
            "it is skipped"
        )
        return None


def ensure_required(entry: WeightEntry, progress: bool = True) -> str:
    """The path of a **required** weight (raises when it is not available)."""
    path = ensure_weight(entry, progress=progress)
    if path is None:  # pragma: no cover - a required entry never returns None
        raise FileNotFoundError(
            f"KEEP: the required weight {entry.name!r} could not be provided.\n"
            f"  url:  {entry.url}\n"
            f"  dest: {target_path(entry)}"
        )
    return path


def ensure_optional(entry: WeightEntry, progress: bool = True) -> str | None:
    """The path of an **optional** weight (``None`` when it is not available)."""
    return ensure_weight(entry, progress=progress)


def manifest_entry(name: str) -> WeightEntry:
    """The manifest entry of a file name."""
    for entry in WEIGHT_MANIFEST:
        if entry.name == name:
            return entry
    raise KeyError(f"KEEP: no manifest entry named {name!r}")
