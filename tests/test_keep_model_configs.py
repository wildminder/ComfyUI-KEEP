"""P5.1 - the weight manifest of the KEEP pack.

The manifest is the single source of truth for the weights: the URL, the
sha256 and the on-disk destination of every file.  The tests assert

* that every entry carries a URL and a destination and that the destinations
  reproduce the **legacy** on-disk layout exactly (``keep_models`` + the
  ``KEEP`` sub-directory, ``facedetection``, ``gmflow``);
* that every sha256 that is present matches ``^[0-9a-f]{64}$``;
* that **exactly one** entry has no hash (``sha256: null`` + ``unverified:
  true`` + ``available: false`` - the Asian checkpoint, which is not published
  upstream) and that **exactly one** entry carries a ``sha256_prefix``;
* that the names are unique and that every entry but the GMFlow checkpoint and
  the Asian checkpoint is **required**;
* that the manifest agrees with the **provenance record** of P5.0 (the hashes
  are not typed by hand twice) and with the source table of the fetch script.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = REPO_ROOT / "tools" / "weight_hashes.json"

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

#: The eight files of the manifest.
EXPECTED_NAMES = (
    "KEEP-b76feb75.pth",
    "KEEP_Asian-4765ebe0.pth",
    "detection_Resnet50_Final.pth",
    "detection_mobilenet0.25_Final.pth",
    "yolov5n-face.pth",
    "yolov5l-face.pth",
    "parsing_parsenet.pth",
    "gmflow_sintel-0c07dcb3.pth",
)


@pytest.fixture(scope="module")
def configs_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_model_configs")


@pytest.fixture(scope="module")
def provenance() -> dict:
    """The machine-readable provenance record of P5.0."""
    data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {entry["name"]: entry for entry in data["entries"].values()}


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_manifest_has_the_eight_expected_entries(configs_module):
    entries = configs_module.WEIGHT_MANIFEST
    assert [entry.name for entry in entries] == list(EXPECTED_NAMES)
    names = [entry.name for entry in entries]
    assert len(set(names)) == len(names), "the names must be unique"


@pytest.mark.unit
def test_every_entry_has_a_url_and_a_destination(configs_module):
    for entry in configs_module.WEIGHT_MANIFEST:
        assert entry.url, entry.name
        assert entry.url.startswith("https://"), entry.name
        assert entry.dest_folder, entry.name
        assert isinstance(entry.subdir, str), entry.name


@pytest.mark.unit
def test_the_manifest_keeps_the_legacy_on_disk_layout(configs_module):
    """The compatibility guarantee: keep_models + KEEP subdir, facedetection."""
    layout = {
        entry.name: (entry.dest_folder, entry.subdir)
        for entry in configs_module.WEIGHT_MANIFEST
    }
    assert layout["KEEP-b76feb75.pth"] == ("keep_models", "KEEP")
    assert layout["KEEP_Asian-4765ebe0.pth"] == ("keep_models", "KEEP")
    assert layout["detection_Resnet50_Final.pth"] == ("facedetection", "")
    assert layout["detection_mobilenet0.25_Final.pth"] == ("facedetection", "")
    assert layout["yolov5n-face.pth"] == ("facedetection", "")
    assert layout["yolov5l-face.pth"] == ("facedetection", "")
    assert layout["parsing_parsenet.pth"] == ("facedetection", "")
    assert layout["gmflow_sintel-0c07dcb3.pth"] == ("gmflow", "")


@pytest.mark.unit
def test_every_present_sha256_is_a_sha256(configs_module):
    for entry in configs_module.WEIGHT_MANIFEST:
        if entry.sha256 is None:
            continue
        assert SHA256_PATTERN.match(entry.sha256), (
            f"{entry.name}: {entry.sha256!r} is not a sha256"
        )


@pytest.mark.unit
def test_exactly_one_entry_has_no_hash(configs_module):
    """The Asian checkpoint: no hash, unverified, not available upstream."""
    without_hash = [entry for entry in configs_module.WEIGHT_MANIFEST if entry.sha256 is None]
    assert len(without_hash) == 1, [entry.name for entry in without_hash]
    asian = without_hash[0]
    assert asian.name == "KEEP_Asian-4765ebe0.pth"
    assert asian.unverified is True
    assert asian.available is False
    assert asian.required is False, (
        "a required entry would break the default KEEP model (P0.4 section 4.1)"
    )


@pytest.mark.unit
def test_exactly_one_entry_carries_a_sha256_prefix(configs_module):
    prefixes = [
        entry for entry in configs_module.WEIGHT_MANIFEST if entry.sha256_prefix
    ]
    assert len(prefixes) == 1, [entry.name for entry in prefixes]
    assert prefixes[0].name == "KEEP_Asian-4765ebe0.pth"
    assert prefixes[0].sha256_prefix == "4765ebe0"


@pytest.mark.unit
def test_the_required_flags_are_the_planned_ones(configs_module):
    """Everything is required except GMFlow (defensive) and Asian (404)."""
    optional = {
        entry.name for entry in configs_module.WEIGHT_MANIFEST if not entry.required
    }
    assert optional == {"gmflow_sintel-0c07dcb3.pth", "KEEP_Asian-4765ebe0.pth"}


@pytest.mark.unit
def test_the_manifest_matches_the_provenance_record(configs_module, provenance):
    """The hashes are not typed twice - they come from the P5.0 record."""
    assert set(provenance) == set(EXPECTED_NAMES)
    for entry in configs_module.WEIGHT_MANIFEST:
        recorded = provenance[entry.name]
        assert entry.url == recorded["url"], entry.name
        assert entry.dest_folder == recorded["dest_folder"], entry.name
        assert entry.subdir == recorded["subdir"], entry.name
        assert entry.sha256 == recorded["sha256"], entry.name
        assert entry.size == recorded["size"], entry.name


@pytest.mark.unit
def test_the_manifest_matches_the_source_table_of_the_fetch_script(configs_module):
    """The tool and the production manifest must not drift apart."""
    from tools import fetch_weight_hashes as fwh

    sources = {source.name: source for source in fwh.WEIGHT_SOURCES}
    assert set(sources) == set(EXPECTED_NAMES)
    for entry in configs_module.WEIGHT_MANIFEST:
        source = sources[entry.name]
        assert entry.url == source.url, entry.name
        assert entry.dest_folder == source.dest_folder, entry.name
        assert entry.subdir == source.subdir, entry.name
        assert entry.required == source.required, entry.name


@pytest.mark.unit
def test_the_model_configs_point_at_the_manifest(configs_module):
    """The two KEEP checkpoints are manifest entries - not a second table."""
    for model_type, config in configs_module.KEEP_MODEL_CONFIGS.items():
        weight = config["weight"]
        entry = configs_module.WEIGHT_MANIFEST_BY_NAME[weight]
        assert entry.name == weight
        # the derived fields must agree with the manifest
        assert config["url"] == entry.url
        assert config["dest_folder"] == entry.dest_folder
        assert config["subdir"] == entry.subdir
        assert config["sha256"] == entry.sha256


@pytest.mark.unit
def test_the_facelib_list_is_derived_from_the_manifest(configs_module):
    """The facelib weights are manifest entries (no second URL table)."""
    facelib = {
        entry.name
        for entry in configs_module.WEIGHT_MANIFEST
        if entry.dest_folder == "facedetection"
    }
    assert set(configs_module.FACELIB_MODEL_URLS) == facelib
    assert configs_module.FACELIB_DEST_FOLDER == "facedetection"
    for name, (url, sha256) in configs_module.FACELIB_MODEL_URLS.items():
        entry = configs_module.WEIGHT_MANIFEST_BY_NAME[name]
        assert url == entry.url
        assert sha256 == entry.sha256
