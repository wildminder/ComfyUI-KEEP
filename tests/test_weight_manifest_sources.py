"""P5.0 - the provenance of the weight manifest (one verified download each).

The sha256 values of the manifest come from a **one time verified download**
(P5.0), because this host has no weights at all (P0.2: 0 of 8 present).  The
tests assert

* that the provenance document records **all 8** weights - **7** with a
  verified sha256 and **1** (the Asian checkpoint) with ``sha256: null`` /
  ``unverified: true`` / ``available: false`` and the **404** as evidence;
* that the recorded name prefix (the upstream ``-<8 hex>`` convention) agrees
  with the recorded hash;
* that a URL which answers **404** is **never downloaded** (the transport is a
  test double - no network);
* that the script is **idempotent** (a re-run renders byte-identical output)
  and that it **refuses to invent a hash** when the network is disabled.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tools import fetch_weight_hashes as fwh

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = REPO_ROOT / "tools" / "weight_hashes.json"
DOC_FILE = REPO_ROOT / ".dev" / "docs" / "audit" / "2026-10-22-weight-manifest-sources.md"

#: The audit artifacts are development documentation: they are not shipped with
#: the pack and may be absent from a working tree.
pytestmark = pytest.mark.skipif(
    not DOC_FILE.exists(),
    reason=f"the audit artifact {DOC_FILE.name} is not in the working tree",
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _recorded_entries() -> list[dict]:
    """The machine-readable block of the provenance document."""
    text = DOC_FILE.read_text(encoding="utf-8")
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    assert match, f"no machine-readable provenance block in {DOC_FILE}"
    return json.loads(match.group(1))["entries"]


class _RecordingTransport:
    """A transport double: it records every call and never uses the network."""

    def __init__(self, statuses: dict[str, int] | None = None) -> None:
        self.statuses = statuses or {}
        self.head_calls: list[str] = []
        self.download_calls: list[str] = []

    def head(self, url: str):
        self.head_calls.append(url)
        status = self.statuses.get(url, 200)
        return status, 1234

    def download(self, url: str, dest: Path, attempts: int = 3):
        self.download_calls.append(url)
        raise AssertionError("the transport must not download a 404 URL")


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_provenance_doc_records_all_eight_weights():
    """The document covers exactly the eight manifest entries."""
    entries = _recorded_entries()
    expected = [source.name for source in fwh.WEIGHT_SOURCES]
    assert [entry["name"] for entry in entries] == expected
    assert len(entries) == 8


@pytest.mark.unit
def test_seven_entries_carry_a_verified_sha256():
    """Seven weights were downloaded once and hashed; one is not published."""
    entries = _recorded_entries()
    verified = [entry for entry in entries if entry.get("sha256")]
    assert len(verified) == 7, [e["name"] for e in verified]
    for entry in verified:
        assert SHA256_PATTERN.match(entry["sha256"]), entry["name"]
        assert entry["available"] is True
        assert entry["unverified"] is False
        assert entry["http_status"] == 200
        assert entry["size"] and entry["size"] > 0


@pytest.mark.unit
def test_the_unavailable_asset_is_recorded_without_a_hash():
    """The Asian checkpoint: 404, no hash, no invented provenance."""
    entries = {entry["name"]: entry for entry in _recorded_entries()}
    asian = entries["KEEP_Asian-4765ebe0.pth"]
    assert asian["sha256"] is None, "a hash was invented for a missing file"
    assert asian["unverified"] is True
    assert asian["available"] is False
    assert asian["http_status"] == 404
    assert asian["size"] is None
    # the only provenance the file name carries
    assert asian["filename_prefix"] == "4765ebe0"
    assert asian["required"] is False, (
        "a required entry would break the default KEEP model (P0.4 section 4.1)"
    )


@pytest.mark.unit
def test_the_recorded_name_prefix_agrees_with_the_recorded_hash():
    """The upstream ``-<8 hex>`` convention is cross-checked, not assumed.

    The record carries the outcome of the check (``prefix_match``); the test
    asserts that the flag agrees with the hash that was actually computed.
    """
    for entry in _recorded_entries():
        prefix = entry.get("filename_prefix")
        sha256 = entry.get("sha256")
        if not prefix or not sha256:
            continue
        expected = sha256.startswith(prefix)
        assert entry["prefix_match"] is expected, (
            f"{entry['name']}: prefix_match={entry['prefix_match']} but the "
            f"hash {sha256} {'starts' if expected else 'does not start'} with {prefix}"
        )


@pytest.mark.unit
def test_the_verified_checkpoints_follow_the_upstream_naming_convention():
    """The two assets whose name carries a hash really start with it.

    This is the cross-check the plan asks for ("the ``-b76feb75``-style hashes
    in the release asset names are the first 8 hex chars").
    """
    entries = {entry["name"]: entry for entry in _recorded_entries()}
    for name in ("KEEP-b76feb75.pth", "gmflow_sintel-0c07dcb3.pth"):
        entry = entries[name]
        assert entry["sha256"].startswith(entry["filename_prefix"]), (
            f"{name}: the sha256 does not start with the upstream name prefix "
            f"{entry['filename_prefix']} - the download was not the expected file"
        )


@pytest.mark.unit
def test_the_document_keeps_the_legacy_on_disk_layout():
    """The layout is a compatibility guarantee (keep_models + KEEP subdir)."""
    entries = {entry["name"]: entry for entry in _recorded_entries()}
    layout = {
        name: (entry["dest_folder"], entry["subdir"]) for name, entry in entries.items()
    }
    assert layout["KEEP-b76feb75.pth"] == ("keep_models", "KEEP")
    assert layout["KEEP_Asian-4765ebe0.pth"] == ("keep_models", "KEEP")
    assert layout["detection_Resnet50_Final.pth"] == ("facedetection", "")
    assert layout["detection_mobilenet0.25_Final.pth"] == ("facedetection", "")
    assert layout["yolov5n-face.pth"] == ("facedetection", "")
    assert layout["yolov5l-face.pth"] == ("facedetection", "")
    assert layout["parsing_parsenet.pth"] == ("facedetection", "")
    assert layout["gmflow_sintel-0c07dcb3.pth"] == ("gmflow", "")


# ---------------------------------------------------------------------------
# the script
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_404_url_is_never_downloaded(tmp_path):
    """The 404 answer ends the attempt - there is nothing to hash."""
    source = next(s for s in fwh.WEIGHT_SOURCES if s.name == "KEEP_Asian-4765ebe0.pth")
    transport = _RecordingTransport({source.url: 404})

    record = fwh.fetch_entry(source, transport, tmp_path)

    assert transport.head_calls == [source.url]
    assert transport.download_calls == [], "a 404 URL must not be downloaded"
    assert record["available"] is False
    assert record["unverified"] is True
    assert record["sha256"] is None
    assert record["http_status"] == 404


@pytest.mark.unit
def test_a_live_url_is_downloaded_once_and_hashed(tmp_path, monkeypatch):
    """A 200 answer leads to exactly one download and a recorded hash."""
    source = fwh.WEIGHT_SOURCES[0]
    calls: list[str] = []

    class _Transport:
        def head(self, url):
            calls.append(("head", url))
            return 200, 4

        def download(self, url, dest, attempts=3, expected_size=None):
            calls.append(("download", url))
            dest.write_bytes(b"data")
            return "a" * 64, 4

    record = fwh.fetch_entry(source, _Transport(), tmp_path)

    assert [call[0] for call in calls] == ["head", "download"]
    assert record["sha256"] == "a" * 64
    assert record["available"] is True
    assert record["unverified"] is False


@pytest.mark.unit
def test_the_script_is_idempotent(tmp_path, monkeypatch):
    """A re-run renders byte-identical output and downloads nothing."""
    cache_file = tmp_path / "cache.json"
    doc_file = tmp_path / "doc.md"
    cache_file.write_text(CACHE_FILE.read_text(encoding="utf-8"), encoding="utf-8")

    def _no_network(*args, **kwargs):
        raise AssertionError("the offline mode must not touch the network")

    monkeypatch.setattr(fwh.HttpTransport, "head", _no_network)
    monkeypatch.setattr(fwh.HttpTransport, "download", _no_network)

    assert fwh.main(["--offline", "--cache", str(cache_file), "--doc", str(doc_file)]) == 0
    first = doc_file.read_text(encoding="utf-8")
    first_cache = cache_file.read_text(encoding="utf-8")

    assert fwh.main(["--offline", "--cache", str(cache_file), "--doc", str(doc_file)]) == 0
    assert doc_file.read_text(encoding="utf-8") == first, "the re-run differs"
    assert cache_file.read_text(encoding="utf-8") == first_cache, "the cache changed"


@pytest.mark.unit
def test_the_committed_document_is_in_sync_with_the_committed_cache():
    """The document is a pure function of the cache - they cannot drift."""
    cache = fwh.load_cache(CACHE_FILE)
    assert DOC_FILE.read_text(encoding="utf-8") == fwh.render_doc(cache), (
        ".dev/docs/audit/2026-10-22-weight-manifest-sources.md is out of sync with "
        "tools/weight_hashes.json - re-run tools/fetch_weight_hashes.py"
    )


@pytest.mark.unit
def test_the_offline_mode_refuses_to_invent_a_hash(tmp_path, capsys):
    """Without a cache entry and without the network the script fails loudly."""
    cache_file = tmp_path / "cache.json"
    doc_file = tmp_path / "doc.md"
    cache_file.write_text(json.dumps({"release": fwh.RELEASE, "entries": {}}), encoding="utf-8")

    assert fwh.main(["--offline", "--cache", str(cache_file), "--doc", str(doc_file)]) == 1
    message = capsys.readouterr().err
    assert fwh.FETCH_ENV in message, (
        "the error must name the environment variable that enables the fetch"
    )
    assert not doc_file.exists(), "no document may be written from missing data"
