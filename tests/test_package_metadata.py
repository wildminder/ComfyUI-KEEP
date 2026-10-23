"""P3.7 - the registry metadata of the pack.

``pyproject.toml`` is the metadata the ComfyUI Registry reads.  The test
parses it with ``tomllib`` (the standard library since Python 3.11) and
asserts that the metadata is complete and that the dependency list is the
true runtime set of P2.6 (``requirements.txt``).

The parser import is **guarded**: the pack supports Python 3.10
(``requires-python = ">=3.10"``), where ``tomllib`` does not exist.  The
3.10 CI job installs ``tomli`` (the same parser under its old name), so the
metadata is asserted on 3.10 as well - the module never dies during
collection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10 - the CI job installs ``tomli``
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # no TOML parser at all
        tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

#: The metadata of the pack (P3.7).
EXPECTED_VERSION = "2.0.0"
EXPECTED_REQUIRES_PYTHON = ">=3.10"
EXPECTED_COMFY = {
    "PublisherId": "wildai",
    "DisplayName": "KEEP",
    "requires-comfyui": ">=0.28.0",
}

#: The preview asset of the registry entry (P3.7 - the ``Icon`` key).
EXPECTED_ICON = "assets/keep-preview.png"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    assert PYPROJECT.is_file(), f"missing {PYPROJECT}"
    if tomllib is None:
        pytest.skip(
            "no TOML parser available - install ``tomli`` on Python 3.10 "
            "(the CI job does)"
        )
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _requirement_entries() -> set[str]:
    return {
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


@pytest.mark.unit
def test_the_pyproject_parses(pyproject):
    """The metadata is parseable (tomllib on Python 3.11+, tomli on 3.10)."""
    assert "project" in pyproject
    assert pyproject["project"]["name"] == "ComfyUI-KEEP"


@pytest.mark.unit
def test_the_version_is_the_release_version(pyproject):
    """The pack is version ``2.0.0``."""
    assert pyproject["project"]["version"] == EXPECTED_VERSION


@pytest.mark.unit
def test_the_python_requirement_is_declared(pyproject):
    """The pack supports Python 3.10 and newer."""
    assert pyproject["project"]["requires-python"] == EXPECTED_REQUIRES_PYTHON


@pytest.mark.unit
def test_the_comfy_metadata_is_complete(pyproject):
    """Every ``[tool.comfy]`` key of the registry is present."""
    comfy = pyproject.get("tool", {}).get("comfy")
    assert comfy is not None, "pyproject.toml has no [tool.comfy] table"
    for key, value in EXPECTED_COMFY.items():
        assert key in comfy, f"[tool.comfy] has no {key}"
        assert comfy[key] == value, f"[tool.comfy].{key}: {comfy[key]!r} != {value!r}"
    assert "Icon" in comfy, "[tool.comfy] has no Icon"


@pytest.mark.unit
def test_the_icon_points_at_the_preview_asset(pyproject):
    """The ``Icon`` is not an empty string - it points at the preview asset.

    An empty ``Icon`` passes the "the key exists" check above, but the
    registry entry has no preview then: the value must name a file that is
    really in the repository.
    """
    comfy = pyproject.get("tool", {}).get("comfy") or {}
    icon = comfy.get("Icon", "")
    assert isinstance(icon, str) and icon.strip(), (
        f"[tool.comfy].Icon is empty - the registry entry would have no preview"
    )
    assert icon == EXPECTED_ICON, icon
    asset = REPO_ROOT / icon
    assert asset.is_file(), f"the icon {icon!r} is not in the repository"


@pytest.mark.unit
def test_the_dependencies_are_the_true_runtime_set(pyproject):
    """The declared dependencies are exactly the ones of ``requirements.txt``."""
    declared = set(pyproject["project"]["dependencies"])
    assert declared == _requirement_entries(), sorted(
        declared ^ _requirement_entries()
    )
