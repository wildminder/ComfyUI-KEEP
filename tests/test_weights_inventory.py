"""P0.2 - weights inventory tests.

The inventory records what is on **this host**.  On this host all 8 weights are
absent, so the recorded sha256 is ``null`` - never invented.  The tests

* assert that the audit document lists exactly the 8 expected weights with the
  legacy on-disk layout (``keep_models`` + ``KEEP`` subdir, ``facedetection``,
  ``gmflow``),
* assert that an absent file carries no hash and no size (no invented facts),
* re-scan the same folders and assert that every **present** file still has the
  recorded size/sha256, and that files recorded as absent are still absent.

If no models root can be resolved the re-scan skips cleanly, naming the
environment variables that would provide one.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCANNER = REPO_ROOT / "tools" / "scan_weights.py"
INVENTORY_DOC = REPO_ROOT / "docs" / "audit" / "2026-10-22-weights-inventory.md"

#: The audit artifacts are development documentation: they are not shipped with
#: the pack and may be absent from a working tree.
pytestmark = pytest.mark.skipif(
    not INVENTORY_DOC.exists(),
    reason=f"the audit artifact {INVENTORY_DOC.name} is not in the working tree",
)


def _load_scanner():
    spec = importlib.util.spec_from_file_location("keep_scan_weights", SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recorded_inventory() -> dict:
    text = INVENTORY_DOC.read_text(encoding="utf-8")
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    assert match, f"no machine-readable inventory block in {INVENTORY_DOC}"
    return json.loads(match.group(1))


@pytest.mark.unit
def test_inventory_lists_all_eight_weights():
    scanner = _load_scanner()
    inventory = _recorded_inventory()
    recorded = {entry["name"]: entry for entry in inventory["entries"]}
    expected = {spec["name"]: spec for spec in scanner.EXPECTED_WEIGHTS}
    assert set(recorded) == set(expected), (
        f"inventory covers {sorted(recorded)}, expected {sorted(expected)}"
    )
    assert inventory["total_count"] == 8


@pytest.mark.unit
def test_inventory_preserves_legacy_on_disk_layout():
    """The compatibility guarantee: keep_models + KEEP subdir, facedetection."""
    inventory = _recorded_inventory()
    layout = {e["name"]: (e["dest_folder"], e["subdir"]) for e in inventory["entries"]}
    assert layout["KEEP-b76feb75.pth"] == ("keep_models", "KEEP")
    assert layout["KEEP_Asian-4765ebe0.pth"] == ("keep_models", "KEEP")
    assert layout["detection_Resnet50_Final.pth"] == ("facedetection", "")
    assert layout["detection_mobilenet0.25_Final.pth"] == ("facedetection", "")
    assert layout["yolov5n-face.pth"] == ("facedetection", "")
    assert layout["yolov5l-face.pth"] == ("facedetection", "")
    assert layout["parsing_parsenet.pth"] == ("facedetection", "")
    assert layout["gmflow_sintel-0c07dcb3.pth"] == ("gmflow", "")


@pytest.mark.unit
def test_absent_files_carry_no_invented_hash():
    inventory = _recorded_inventory()
    for entry in inventory["entries"]:
        if entry["present"]:
            assert entry["sha256"], f"{entry['name']} is present but has no sha256"
            assert entry["size"] is not None
        else:
            assert entry["sha256"] is None, (
                f"{entry['name']} is absent but carries a hash - that would be invented"
            )
            assert entry["size"] is None, (
                f"{entry['name']} is absent but carries a size"
            )


@pytest.mark.unit
def test_rescan_matches_recorded_inventory():
    """Re-scan the same folders; present files must still match their hash."""
    scanner = _load_scanner()
    models_root = scanner.resolve_models_root(None)
    if models_root is None:
        pytest.skip(
            "no ComfyUI models root found: set KEEP_MODEL_DIR (models dir) or "
            "COMFYUI_ROOT (ComfyUI checkout) to enable the re-scan"
        )

    recorded = {e["name"]: e for e in _recorded_inventory()["entries"]}
    fresh = scanner.scan(models_root)
    fresh_by_name = {e["name"]: e for e in fresh["entries"]}

    assert set(fresh_by_name) == set(recorded)
    for name, entry in fresh_by_name.items():
        old = recorded[name]
        assert entry["present"] == old["present"], (
            f"{name}: presence changed since the inventory was recorded "
            f"(recorded present={old['present']}, now present={entry['present']})"
        )
        if entry["present"]:
            assert old["sha256"] is not None, (
                f"{name} is present but the inventory recorded no hash"
            )
            assert entry["sha256"] == old["sha256"], (
                f"{name}: sha256 changed "
                f"({old['sha256']} -> {entry['sha256']})"
            )
            assert entry["size"] == old["size"], f"{name}: size changed"


@pytest.mark.unit
def test_audit_doc_states_the_absence_explicitly():
    text = INVENTORY_DOC.read_text(encoding="utf-8")
    assert "all 8 weights are absent" in text.lower(), (
        "the inventory must state explicitly that the weights are absent"
    )
    assert "null" in text, "absent weights must be recorded with a null hash"
