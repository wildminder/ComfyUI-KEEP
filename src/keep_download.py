"""The manifest-driven downloader of the KEEP pack (P5.2).

Every weight of the pack comes from the ``WEIGHT_MANIFEST``
(:mod:`src.keep_model_configs`): the manifest knows the URL, the sha256 and the
on-disk destination of a file, and this module turns that into an actual file
on disk.

The rules are:

* the destination is resolved through :mod:`src.keep_paths` - the registered
  model folders of ComfyUI, the **primary root first**, so a weight that lives
  in an extra model root is used where it is instead of being downloaded again;
* a **missing** weight is downloaded next to the **other weights of the pack**:
  the target is the root that already holds them
  (:func:`src.keep_paths.resolve_download_root`), not the primary root of a
  fresh installation;
* the on-disk layout is the **legacy** one (``models/keep_models/KEEP``,
  ``models/facedetection``, ``models/gmflow``);
* a downloaded file is verified against the sha256 of the manifest before it is
  put in place - the transport writes into a temporary file, so a failed or
  wrong download never destroys what was there;
* a file that does not match the manifest is only replaced when the pack owns
  it (the primary root of the installation); a file of an **extra** model root
  belongs to the user and is never deleted;
* a file whose manifest entry carries **no** hash (the Asian checkpoint, which
  is not published upstream) is downloaded **without** a verification, and the
  downloader says so loudly;
* a failed download leaves **no** partial file behind and raises an error that
  names the URL and the destination.

The transport is ``torch.hub.download_url_to_file``; the module level name is
the seam the tests replace (no network in the test suite).  It is called
**without** its own progress bar: the bar of ``torch.hub`` is written to
stderr, untagged, and would bypass the logger of the pack.  A **completed**
download instead reports one step to the progress bar of the ComfyUI host
(``comfy.utils.ProgressBar``) - a no-op when the pack runs without a server.
"""

from __future__ import annotations

import hashlib
import os

from torch.hub import download_url_to_file

from . import keep_paths
from .keep_logger import logger, tagged
from .keep_model_configs import WEIGHT_MANIFEST, WeightEntry

#: The names of the entries that were already reported as "no hash" - the
#: warning is loud, but it must not be repeated for every call.
_WARNED_UNVERIFIED: set[str] = set()

#: The suffix of the temporary file of a download: the transport writes into
#: it and the file is only put in place after it was verified.
PART_SUFFIX = ".part"


def sha256_of(path: str) -> str:
    """The sha256 of a file (streamed - the weights are large)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def folder_siblings(entry: WeightEntry) -> list[str]:
    """The paths of the pack's weights in the folder of the entry.

    They are the **siblings** of the entry - the other weights of the same
    model folder.  They decide where a missing weight is downloaded to: next
    to them (see :func:`src.keep_paths.resolve_download_root`).
    """
    return [
        other.relative_path
        for other in WEIGHT_MANIFEST
        if other.dest_folder == entry.dest_folder
    ]


def target_path(entry: WeightEntry) -> str:
    """The path the entry is downloaded to (the root that holds the pack)."""
    return keep_paths.resolve_model_target(
        entry.dest_folder, entry.name, entry.subdir, folder_siblings(entry)
    )


def find_weight(entry: WeightEntry) -> str | None:
    """An **existing** file of the entry (any registered root, primary first)."""
    return keep_paths.find_model_file(entry.dest_folder, entry.name, entry.subdir)


def _primary_target(entry: WeightEntry) -> str:
    """The destination of the weight in the **primary** root of the pack."""
    try:
        root = keep_paths.get_primary_folder(entry.dest_folder)
    except FileNotFoundError:
        return target_path(entry)
    target = os.path.join(
        root, keep_paths.model_relative_path(entry.name, entry.subdir)
    )
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return target


def _is_in_primary_root(entry: WeightEntry, path: str) -> bool:
    """Does the path live in the primary root of the pack?

    The primary root belongs to the installation - the pack owns the files
    there.  A file in an **extra** model root belongs to the user: the pack
    never deletes it.
    """
    try:
        root = keep_paths.get_primary_folder(entry.dest_folder)
    except FileNotFoundError:
        return False
    root = os.path.normcase(os.path.abspath(root))
    path = os.path.normcase(os.path.abspath(path))
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:  # a different drive (Windows)
        return False


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
        f"{entry.name} has no verified sha256 in the manifest "
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
                logger.warning(f"the partial file {path} could not be removed: {exc}")


def _fetch(entry: WeightEntry, target: str) -> None:
    """The transport call (the seam the tests replace).

    The transport is called **without** a progress bar: the bar of
    ``torch.hub`` is written to stderr, untagged, and would bypass the logger
    of the pack.
    """
    download_url_to_file(entry.url, target, hash_prefix=None, progress=False)


def _place(part: str, target: str) -> None:
    """Put the verified file in place (the old file is replaced atomically)."""
    try:
        os.replace(part, target)
    except OSError as exc:
        # the target is held by another process (Windows) - the verified file
        # stays where it is, so nothing is lost
        raise FileNotFoundError(
            tagged(
                f"the verified file could not be put in place at {target}: {exc}\n"
                f"  the file is at {part} - move it into place manually."
            )
        ) from exc


def _host_progress_bar(total: float) -> ProgressBar | None:
    """The progress bar of the ComfyUI host (``None`` without ComfyUI).

    The import is **lazy** (function level) and guarded: the engines of the
    pack must import without a ComfyUI server (the plan's section 3 rule),
    and a progress bar must never break a download.  Outside a server run the
    bar of the host records the updates and no-ops (its hook is ``None``).
    """
    try:
        from comfy.utils import ProgressBar  # lazy: src/ imports without ComfyUI
    except Exception as exc:  # noqa: BLE001 - a progress bar never breaks a download
        logger.debug(f"the host progress bar is not available ({exc})")
        return None
    return ProgressBar(total=total)


def _report_download_done(bar: ProgressBar | None) -> None:
    """The one bar step of a **completed** download (a failed one stays at 0)."""
    if bar is not None:
        bar.update_absolute(1)


def download_weight(entry: WeightEntry, progress: bool = True) -> str:
    """Download the entry to its destination and verify it.

    When ``progress`` is true, a **completed** download advances the progress
    bar of the ComfyUI host by one step (one step per weight file; the loader
    node downloads a handful of weights, so the bar of the node advances per
    weight).  The transport itself still runs without its own (untagged) bar.

    The download is **atomic**: the transport writes into a temporary file and
    the file is only put in place after it was verified.  A failed download
    therefore never destroys what was there.

    A file that does not match the manifest is only replaced when the pack
    **owns** it (the primary root of the installation).  A file in an extra
    model root belongs to the user: it is never deleted - the verified file is
    downloaded into the primary root instead.

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
            logger.debug(f"{entry.name} is already present ({existing})")
            return existing
        if _is_in_primary_root(entry, existing):
            # the pack owns the file (its own installation): it is replaced
            # by the verified download
            logger.warning(
                f"the sha256 of {existing} does not match the manifest - the "
                "file is replaced by a verified download"
            )
            target = existing
        else:
            # the file belongs to the user (an extra model root): it is never
            # deleted.  The verified file is downloaded into the primary root
            # of the pack instead.
            logger.warning(
                f"the sha256 of {existing} does not match the manifest - the "
                "file is kept untouched (it is not owned by the pack) and the "
                "verified file is downloaded into the primary root"
            )
            target = _primary_target(entry)

    bar = _host_progress_bar(1) if progress else None
    part = target + PART_SUFFIX
    for attempt in (1, 2):
        logger.info(f"downloading {entry.name} from {entry.url} to {target}")
        try:
            _fetch(entry, part)
        except Exception as exc:  # noqa: BLE001 - the error is re-raised below
            _remove_partials(part)
            raise FileNotFoundError(
                tagged(
                    f"the weight {entry.name!r} could not be downloaded.\n"
                    f"  url:  {entry.url}\n"
                    f"  dest: {target}\n"
                    f"  reason: {exc}\n"
                    "Download the file manually into the destination and try again."
                )
            ) from exc

        if not entry.sha256:
            _warn_unverified(entry)
            _place(part, target)
            _report_download_done(bar)
            return target

        actual = sha256_of(part)
        if actual == entry.sha256:
            _place(part, target)
            _report_download_done(bar)
            return target

        logger.warning(
            f"the sha256 of the downloaded {entry.name} does not match "
            f"the manifest (expected {entry.sha256}, got {actual})"
        )
        _remove_partials(part)
        if attempt == 2:
            raise ValueError(
                tagged(
                    f"the downloaded {entry.name!r} does not match the "
                    f"manifest hash (expected {entry.sha256}, got {actual}) - "
                    f"nothing was put in place. url: {entry.url}, dest: {target}"
                )
            )
    raise AssertionError("unreachable")  # pragma: no cover


def ensure_weight(entry: WeightEntry, progress: bool = True) -> str | None:
    """The path of the weight - the cached file, or a fresh download.

    ``progress`` is passed to :func:`download_weight` (a completed download
    advances the progress bar of the ComfyUI host by one step).

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
                    f"{entry.name} is not published upstream - the manually "
                    f"placed file {existing} is used"
                )
                return existing
        if entry.required:
            raise FileNotFoundError(
                tagged(
                    f"the weight {entry.name!r} is not published upstream "
                    f"(HTTP 404) and is not installed.\n"
                    f"  url:  {entry.url}\n"
                    f"  dest: {target_path(entry)}\n"
                    "Place the file manually at the destination and try again."
                )
            )
        logger.warning(
            f"the optional weight {entry.name!r} is not published "
            f"upstream (HTTP 404) - it is skipped. url: {entry.url}"
        )
        return None

    try:
        return download_weight(entry, progress=progress)
    except (FileNotFoundError, ValueError):
        if entry.required:
            raise
        logger.warning(
            f"the optional weight {entry.name!r} could not be provided - "
            "it is skipped"
        )
        return None


def ensure_required(entry: WeightEntry, progress: bool = True) -> str:
    """The path of a **required** weight (raises when it is not available)."""
    path = ensure_weight(entry, progress=progress)
    if path is None:  # pragma: no cover - a required entry never returns None
        raise FileNotFoundError(
            tagged(
                f"the required weight {entry.name!r} could not be provided.\n"
                f"  url:  {entry.url}\n"
                f"  dest: {target_path(entry)}"
            )
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
    raise KeyError(tagged(f"no manifest entry named {name!r}"))
