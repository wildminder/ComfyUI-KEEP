"""P6.2 - the README is consistent with the repository.

The README is the place where the pack makes claims about itself, so the
claims are **checked** instead of trusted:

* every **relative link** of the README points at a file that exists,
* every **internal anchor** of the README resolves to an ``<a id="...">`` in
  the README itself,
* every **path** the README mentions (in a link, in an image ``src`` or in
  backticks) exists in the repository - the paths of the ComfyUI
  installation (the model folders, ``ComfyUI/custom_nodes``) are the only
  exception, and they are listed explicitly,
* every **node id** the README mentions is a node the pack registers - and
  every node the pack registers is mentioned,
* the **weights table** is the table the manifest generates (the manifest is
  the single source of truth: a URL, a hash, a size or a destination that
  changes breaks this test),
* the third-party license file exists and is linked,
* the **S-Lab non-commercial notice** and the **VideoHelperSuite note** are
  really in the text.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
NODES_DIR = REPO_ROOT / "nodes"
THIRD_PARTY_LICENSE = REPO_ROOT / "LICENSE-THIRD-PARTY.md"
DOCS_TABLES = REPO_ROOT / "tools" / "docs_tables.py"

#: The socket types of the pack - they are *not* node ids.
SOCKET_TYPES = {"KEEP_MODEL_PACK"}

#: The path roots that live in the **ComfyUI installation**, not in the
#: repository (the model folders and the custom node folder).
EXTERNAL_PATH_ROOTS = {"ComfyUI", "custom_nodes", "models"}

#: The extensions that make a backticked token a path claim of the
#: repository.  The weight extensions (.pth) are not part of it: the weights
#: live in the model folders of ComfyUI.
REPO_FILE_EXTENSIONS = (
    ".cfg",
    ".json",
    ".md",
    ".png",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
)

LINK_RE = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)")
REFERENCE_RE = re.compile(r"^\[[^\]]+\]:\s*(?P<target>\S+)", re.MULTILINE)
SRC_RE = re.compile(r'src="(?P<target>[^"]+)"')
BACKTICK_RE = re.compile(r"`(?P<token>[^`\n]+)`")
ANCHOR_RE = re.compile(r'id="(?P<id>[^"]+)"')
HTML_HREF_RE = re.compile(r'href="(?P<target>#[^"]+)"')
NODE_ID_RE = re.compile(r"(?<![\w.\-])KEEP_[A-Za-z0-9_]+(?![\w.\-])")


@pytest.fixture(scope="module")
def readme_text() -> str:
    assert README.is_file(), f"missing {README}"
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def docs_tables():
    """The table generator of ``tools/docs_tables.py``."""
    spec = importlib.util.spec_from_file_location("keep_docs_tables", DOCS_TABLES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest():
    """The weight manifest of the pack."""
    module = importlib.import_module("ComfyUI_KEEP.src.keep_model_configs")
    return module.WEIGHT_MANIFEST


@pytest.fixture(scope="module")
def registered_node_ids() -> set[str]:
    """The node ids the pack registers (parsed from the node sources)."""
    ids: set[str] = set()
    for source in sorted(NODES_DIR.glob("*.py")):
        ids.update(re.findall(r'node_id="([^"]+)"', source.read_text(encoding="utf-8")))
    return ids


def _is_external(target: str) -> bool:
    return target.split("/", 1)[0] in EXTERNAL_PATH_ROOTS


def _repo_path(target: str) -> Path:
    return REPO_ROOT / target.lstrip("./")


@pytest.mark.unit
def test_the_relative_links_of_the_readme_exist(readme_text):
    """Every relative link of the README points at a file that exists."""
    targets = [match.group("target") for match in LINK_RE.finditer(readme_text)]
    targets += [match.group("target") for match in REFERENCE_RE.finditer(readme_text)]
    assert targets, "the README has no links at all - the test would be vacuous"

    for target in targets:
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if target.startswith("#"):
            continue  # the anchors are checked by their own test
        assert _repo_path(target).exists(), (
            f"the README links {target!r}, which does not exist in the repository"
        )


@pytest.mark.unit
def test_the_images_of_the_readme_exist(readme_text):
    """Every image of the README is a file that exists."""
    targets = [match.group("target") for match in SRC_RE.finditer(readme_text)]
    assert targets, "the README has no image - the test would be vacuous"

    for target in targets:
        if target.startswith(("http://", "https://", "data:")):
            continue
        assert _repo_path(target).is_file(), (
            f"the README shows {target!r}, which is not a file of the repository"
        )


@pytest.mark.unit
def test_the_internal_anchors_of_the_readme_resolve(readme_text):
    """Every ``#anchor`` of the README has an ``id="..."`` in the README."""
    anchors = {match.group("id") for match in ANCHOR_RE.finditer(readme_text)}
    assert anchors, "the README has no explicit anchors"

    linked = {
        match.group("target")[1:]
        for match in LINK_RE.finditer(readme_text)
        if match.group("target").startswith("#")
    }
    linked |= {
        match.group("target")[1:] for match in HTML_HREF_RE.finditer(readme_text)
    }
    assert linked, "the README has no internal links - the test would be vacuous"

    missing = sorted(linked - anchors)
    assert not missing, f"the README links to the unknown anchors {missing}"


@pytest.mark.unit
def test_every_path_the_readme_mentions_exists(readme_text):
    """Every path of the README exists (the ComfyUI paths are the exception)."""
    tokens = []
    for match in BACKTICK_RE.finditer(readme_text):
        token = match.group("token").strip()
        if not re.fullmatch(r"[A-Za-z0-9_./\-]+", token):
            continue
        if "/" not in token and not token.endswith(REPO_FILE_EXTENSIONS):
            continue
        tokens.append(token)

    assert tokens, "the README mentions no path - the test would be vacuous"

    for token in tokens:
        if _is_external(token):
            continue
        path = _repo_path(token.rstrip("/"))
        assert path.exists(), (
            f"the README mentions {token!r}, which does not exist in the repository"
        )


@pytest.mark.unit
def test_the_node_ids_of_the_readme_exist(readme_text, registered_node_ids):
    """Every node id of the README is a node the pack registers."""
    mentioned = {
        token
        for token in NODE_ID_RE.findall(readme_text)
        if token not in SOCKET_TYPES
    }
    assert mentioned, "the README mentions no node id - the test would be vacuous"

    unknown = sorted(mentioned - registered_node_ids)
    assert not unknown, (
        f"the README mentions the node ids {unknown}, which the pack does not "
        f"register (it registers {sorted(registered_node_ids)})"
    )


@pytest.mark.unit
def test_every_node_of_the_pack_is_documented(readme_text, registered_node_ids):
    """Every node the pack registers is mentioned in the README."""
    mentioned = {
        token
        for token in NODE_ID_RE.findall(readme_text)
        if token not in SOCKET_TYPES
    }
    missing = sorted(registered_node_ids - mentioned)
    assert not missing, f"the README does not document the nodes {missing}"


@pytest.mark.unit
def test_the_weights_table_matches_the_manifest(readme_text, manifest, docs_tables):
    """The weights table of the README is the table of the manifest."""
    table = docs_tables.weights_table(manifest)
    assert table in readme_text, (
        "the weights table of the README does not match the manifest - "
        "regenerate it with tools/docs_tables.py"
    )
    for entry in manifest:
        assert entry.name in readme_text, (
            f"the weights table does not mention {entry.name}"
        )


@pytest.mark.unit
def test_the_weights_table_has_one_row_per_manifest_entry(readme_text, manifest):
    """No weight of the manifest is missing from the README table."""
    lines = readme_text.splitlines()
    header = "| File | Size | SHA-256 | Destination | Notes |"
    assert header in lines, "the README has no weights table"
    start = lines.index(header)
    rows = []
    for line in lines[start + 1 :]:
        if not line.startswith("|"):
            break
        if line.startswith("|:"):
            continue  # the separator of the table
        rows.append(line)
    assert len(rows) == len(manifest), (
        f"the README has {len(rows)} weight rows, the manifest has {len(manifest)}"
    )


@pytest.mark.unit
def test_the_third_party_license_exists_and_is_linked(readme_text):
    """P6.3 - the third-party license file exists and the README links it."""
    assert THIRD_PARTY_LICENSE.is_file(), (
        f"missing {THIRD_PARTY_LICENSE.name} - the third-party attributions are "
        "mandatory (the KEEP weights are non-commercial)"
    )
    assert "LICENSE-THIRD-PARTY.md" in readme_text, (
        "the README does not link the third-party license file"
    )
    targets = [match.group("target") for match in LINK_RE.finditer(readme_text)]
    assert "LICENSE-THIRD-PARTY.md" in targets, (
        "the third-party license file is mentioned, but not linked"
    )


@pytest.mark.unit
def test_the_s_lab_non_commercial_notice_is_in_the_readme(readme_text):
    """The non-commercial notice of the S-Lab License 1.0 is documented."""
    assert "S-Lab License 1.0" in readme_text, (
        "the README does not name the license of the KEEP weights"
    )
    assert re.search(r"non-commercial", readme_text, re.IGNORECASE), (
        "the README does not carry the non-commercial notice"
    )


@pytest.mark.unit
def test_the_videohelpersuite_dependency_is_documented(readme_text):
    """The VideoHelperSuite dependency of the shipped example is documented."""
    assert "VideoHelperSuite" in readme_text, (
        "the README does not mention the VideoHelperSuite dependency"
    )
    for node_type in ("VHS_LoadVideo", "VHS_VideoCombine"):
        assert node_type in readme_text, (
            f"the README does not name the VideoHelperSuite node {node_type}"
        )


@pytest.mark.unit
def test_the_model_folders_of_the_readme_are_the_ones_of_the_pack(readme_text):
    """The on-disk layout the README documents is the layout of the pack."""
    for folder in (
        "models/keep_models/KEEP",
        "models/facedetection",
        "models/gmflow",
    ):
        assert folder in readme_text, f"the README does not document {folder}"


@pytest.mark.unit
def test_the_asian_checkpoint_is_documented_as_unavailable(readme_text):
    """The dead `Asian` download is documented (the P0.4 requirement)."""
    assert "not published upstream (HTTP 404)" in readme_text, (
        "the README does not say that the Asian checkpoint is not published "
        "upstream"
    )
    assert "KEEP_Asian-4765ebe0.pth" in readme_text, (
        "the README does not mention the Asian checkpoint at all"
    )
