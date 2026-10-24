"""Repo-wide tripwire: no tracked file may carry a machine-local path.

The rule is the owner's: the published package must not name a local drive, a
local directory tree or a local interpreter.  The forbidden fragments are built
from parts, because this file is itself tracked and would otherwise trip the
very check it performs.

The scan covers the **tracked** files only: the internal development documents
live under the untracked ``.dev`` tree and are never part of the package.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The forbidden fragments, built from parts so that this file does not
#: contain them literally.
FORBIDDEN = (
    "C:" + "\\",
    "C:" + "/",
    "_" + "Dev",
)

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


@pytest.mark.unit
def test_no_tracked_file_carries_a_machine_local_path():
    """No tracked file names a local drive, a local tree or the local interpreter."""
    offenders: list[str] = []
    for name in _tracked_files():
        path = REPO_ROOT / name
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - a tracked file that cannot be read
            continue
        for fragment in FORBIDDEN:
            if fragment in text:
                offenders.append(f"{name} contains {fragment!r}")
    assert not offenders, "machine-local paths in tracked files:\n" + "\n".join(offenders)


@pytest.mark.unit
def test_the_internal_documents_are_not_tracked():
    """The development documents live outside the repository's index."""
    tracked = _tracked_files()
    offenders = [name for name in tracked if name.startswith("docs/") or name.startswith(".dev")]
    assert not offenders, "internal documents must not be tracked:\n" + "\n".join(offenders)
