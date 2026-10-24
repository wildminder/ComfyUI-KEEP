"""Repo-wide tripwire: no tracked file may carry a machine-local path.

The rule is the owner's: the published package must not name a local drive, a
local directory tree or a local interpreter.  The forbidden fragments are built
from parts, because this file is itself tracked and would otherwise trip the
very check it performs.

The scan covers the **tracked** files only: the internal development documents
live under the untracked ``.dev`` tree and are never part of the package.

Two fragment sets are checked.  ``FORBIDDEN`` applies to the whole tracked
tree; ``POSIX_FORBIDDEN`` applies outside the vendored tree only, because the
single POSIX literal inside it is upstream code that this repository does not
own.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The forbidden fragments, built from parts so that this file does not
#: contain them literally.  A local tree is matched as a **path segment**: the
#: separators belong to the fragment, so an identifier that merely carries the
#: same letters - a ``load_device`` call, a ``__device__`` kernel attribute -
#: is not a hit.
FORBIDDEN = (
    "C:" + "\\",
    "C:" + "/",
    "/" + "_" + "Dev" + "/",
    "/" + "_" + "dev" + "/",
    "\\" + "_" + "Dev" + "\\",
    "\\" + "_" + "dev" + "\\",
)

#: POSIX home roots.  They are forbidden outside the vendored tree only (see
#: ``VENDORED_PREFIX``): the one literal inside it is upstream code.
POSIX_FORBIDDEN = (
    "/" + "home/",
    "/" + "Users/",
    "/" + "root/",
)

#: The vendored third-party tree.  It is upstream code that this repository
#: does not own, so the POSIX check skips it.
VENDORED_PREFIX = "modules/deps/"

#: Suffixes that hold binary content and are skipped.
BINARY_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pth", ".pt", ".ckpt", ".bin", ".safetensors"}
)


def _tracked_files() -> list[str]:
    """The tracked file list (``git ls-files``), relative to the repository root."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"git is not usable in this checkout: {exc}")
    return [name for name in result.stdout.split("\0") if name]


def _tracked_texts() -> Iterator[tuple[str, str]]:
    """Yield ``(name, text)`` for every tracked file that can be read as text."""
    for name in _tracked_files():
        path = REPO_ROOT / name
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - a tracked file that cannot be read
            continue
        yield name, text


def _offenders(fragments: Iterable[str], texts: Iterable[tuple[str, str]]) -> list[str]:
    """The tracked files whose text carries one of ``fragments``."""
    return [
        f"{name} contains {fragment!r}"
        for name, text in texts
        for fragment in fragments
        if fragment in text
    ]


@pytest.mark.unit
def test_no_tracked_file_carries_a_machine_local_path():
    """No tracked file names a local drive, a local tree or the local interpreter."""
    offenders = _offenders(FORBIDDEN, _tracked_texts())
    assert not offenders, "machine-local paths in tracked files:\n" + "\n".join(offenders)


@pytest.mark.unit
def test_no_tracked_file_outside_the_vendored_tree_carries_a_posix_path():
    """No tracked file of ours names a POSIX home root."""
    texts = ((name, text) for name, text in _tracked_texts() if not name.startswith(VENDORED_PREFIX))
    offenders = _offenders(POSIX_FORBIDDEN, texts)
    assert not offenders, "POSIX machine paths in tracked files:\n" + "\n".join(offenders)


@pytest.mark.unit
def test_a_posix_machine_path_is_caught():
    """A tracked file that names a POSIX home root makes the tripwire fail."""
    text = 'model_dir = "' + "/" + "home" + "/someone" + '/models"'
    assert _offenders(POSIX_FORBIDDEN, [("example.py", text)])


@pytest.mark.unit
def test_a_device_identifier_is_not_a_machine_local_path():
    """An identifier that carries the same letters is not a local tree."""
    identifiers = ("load_device", "offload_device", "__device__", "get_torch_device", "is_dev_only")
    texts = [(f"example_{index}.py", f"value = {name}") for index, name in enumerate(identifiers)]
    assert not _offenders(FORBIDDEN, texts)


@pytest.mark.unit
def test_the_internal_documents_are_not_tracked():
    """The development documents live outside the repository's index."""
    tracked = _tracked_files()
    offenders = [name for name in tracked if name.startswith("docs/") or name.startswith(".dev")]
    assert not offenders, "internal documents must not be tracked:\n" + "\n".join(offenders)
