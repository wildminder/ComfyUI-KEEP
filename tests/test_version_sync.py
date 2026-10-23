"""P6.4 - the version of the pack and the README changelog agree.

The test is **installed** with P3.7 (the metadata of the pack is written
there) and was not strict until the README rewrite of **P6.2** gave the
changelog its ``### v2.0.0`` entry.  Since **P6.4** the assertion is
**strict**: a version bump without a changelog entry (or the other way
round) fails the test.

The file is runnable standalone (``python tests/test_version_sync.py``) and
parses ``pyproject.toml`` with ``tomllib`` when it is available, otherwise
with a version regex (the Python 3.10 CI job has no ``tomli``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
README = REPO_ROOT / "README.md"


def pyproject_version() -> str:
    """The version of ``pyproject.toml``."""
    try:
        import tomllib  # noqa: PLC0415 - the stdlib parser is optional
    except ModuleNotFoundError:  # Python 3.10
        tomllib = None  # type: ignore[assignment]

    if tomllib is not None:
        with PYPROJECT.open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])

    match = re.search(
        r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.MULTILINE
    )
    assert match, "pyproject.toml has no version"
    return match.group(1)


def readme_changelog_version() -> str | None:
    """The latest version of the README changelog (``### vX.Y.Z``)."""
    if not README.is_file():
        return None
    text = README.read_text(encoding="utf-8")
    matches = re.findall(r"^###\s*v(\d+\.\d+\.\d+)", text, re.MULTILINE)
    return matches[0] if matches else None


@pytest.mark.unit
def test_the_version_matches_the_readme_changelog():
    """The pyproject version is the latest entry of the README changelog."""
    version = pyproject_version()
    changelog = readme_changelog_version()
    assert changelog is not None, "the README has no changelog entry yet"
    assert changelog == version, (
        f"pyproject.toml says {version}, the README changelog says {changelog}"
    )


@pytest.mark.unit
def test_the_changelog_entry_carries_a_date():
    """The latest changelog entry has a date (``### vX.Y.Z — YYYY-MM-DD``)."""
    text = README.read_text(encoding="utf-8")
    match = re.search(
        r"^###\s*v(\d+\.\d+\.\d+)\s+—\s+(\d{4}-\d{2}-\d{2})\s*$",
        text,
        re.MULTILINE,
    )
    assert match, (
        "the README has no changelog entry of the form "
        "'### vX.Y.Z — YYYY-MM-DD'"
    )
    assert match.group(1) == pyproject_version(), (
        f"the dated changelog entry is {match.group(1)}, the pack is "
        f"{pyproject_version()}"
    )


if __name__ == "__main__":  # pragma: no cover - the standalone entry point
    version = pyproject_version()
    changelog = readme_changelog_version()
    if changelog != version:
        print(
            f"version mismatch: pyproject.toml={version} README={changelog}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"version sync ok: {version}")
