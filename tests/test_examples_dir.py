"""P0.5 - canonical examples directory tests.

``example_workflows/`` is the canonical path (the plan renamed
``workflow-examples/`` in P0.5); every later test references it.

The shipped example must

* live in ``example_workflows/`` (and the old directory must be gone),
* contain the workflow json and its png preview,
* parse as JSON,
* use exactly the three KEEP node ids, and only whitelisted foreign node types
  (core ComfyUI nodes plus the VideoHelperSuite types, which P6.1 whitelists
  explicitly and documents in the README).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "example_workflows"
LEGACY_DIR = REPO_ROOT / "workflow-examples"
WORKFLOW_JSON = EXAMPLES_DIR / "keep-workflow.json"
WORKFLOW_PNG = EXAMPLES_DIR / "keep-workflow.png"

PACK_NODE_IDS = {
    "KEEP_ModelLoader",
    "KEEP_FaceUpscaleImage",
    "KEEP_ProcessImageSequence",
}

#: Foreign node types the shipped example may use: core ComfyUI nodes and the
#: VideoHelperSuite nodes (the example is a video workflow).
FOREIGN_NODE_WHITELIST = {
    "LoadImage",
    "PreviewImage",
    "SaveImage",
    "UpscaleModelLoader",
    "VHS_LoadVideo",
    "VHS_VideoCombine",
}


@pytest.mark.unit
def test_canonical_examples_dir_exists():
    assert EXAMPLES_DIR.is_dir(), f"missing canonical examples dir: {EXAMPLES_DIR}"
    assert not LEGACY_DIR.exists(), (
        "the legacy workflow-examples/ directory must be gone after the P0.5 rename"
    )


@pytest.mark.unit
def test_examples_dir_contains_json_and_png():
    assert WORKFLOW_JSON.is_file(), f"missing {WORKFLOW_JSON}"
    assert WORKFLOW_PNG.is_file(), f"missing {WORKFLOW_PNG}"
    assert WORKFLOW_PNG.stat().st_size > 0


@pytest.mark.unit
def test_workflow_json_parses():
    data = json.loads(WORKFLOW_JSON.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "nodes" in data, "the workflow must be in the ComfyUI UI format"
    assert isinstance(data["nodes"], list) and data["nodes"]


@pytest.mark.unit
def test_workflow_uses_exactly_the_three_pack_node_ids():
    data = json.loads(WORKFLOW_JSON.read_text(encoding="utf-8"))
    types = {node.get("type") for node in data["nodes"]}
    keep_types = {t for t in types if isinstance(t, str) and t.startswith("KEEP_")}
    assert keep_types == PACK_NODE_IDS, (
        f"the example must use exactly {sorted(PACK_NODE_IDS)}, got {sorted(keep_types)}"
    )


@pytest.mark.unit
def test_workflow_foreign_types_are_whitelisted():
    data = json.loads(WORKFLOW_JSON.read_text(encoding="utf-8"))
    types = {node.get("type") for node in data["nodes"]}
    foreign = types - PACK_NODE_IDS
    unexpected = foreign - FOREIGN_NODE_WHITELIST
    assert not unexpected, f"unexpected node types in the example: {sorted(unexpected)}"


@pytest.mark.unit
def test_workflow_node_ids_match_the_registered_nodes():
    """The ids in the example must be the ids the node registers today."""
    nodes_source = (REPO_ROOT / "nodes.py").read_text(encoding="utf-8")
    for node_id in PACK_NODE_IDS:
        assert f'"{node_id}"' in nodes_source, (
            f"{node_id} is used by the example but is not registered by nodes.py"
        )
