"""P6.1 - the shipped example workflows are valid against the V3 node set.

Every workflow json in ``example_workflows/`` must

* parse (it is the ComfyUI UI format, version 0.4),
* use only node types that ComfyUI really has: a **pack node id** (the ids are
  regex-parsed from ``nodes/*.py``, so a renamed or removed node is caught),
  a **whitelisted core node** or a **whitelisted VideoHelperSuite type**
  (``VHS_LoadVideo`` / ``VHS_VideoCombine`` - the shipped example is a video
  workflow and therefore needs the VideoHelperSuite pack; the dependency is
  documented in the README),
* have a preview image next to it (the file the registry and the README show),
* have consistent links (every link points at a node that exists, every
  connected input refers to a link that exists), and
* match the **V3 schema** of the pack nodes they use: the same number of
  widget values (in the same order), the same socket names and types, and
  widget values that are legal for their widget (a combo value must be one of
  its options, a number must be inside its range).

The last group is what makes "the shipped example still loads with the
migrated ids" a checked statement instead of a claim: a workflow whose widget
values no longer fit the V3 schema would be rejected by ComfyUI.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "example_workflows"
NODES_DIR = REPO_ROOT / "nodes"

#: The core ComfyUI node types the examples may use.
CORE_NODE_WHITELIST = {
    "LoadImage",
    "PreviewImage",
    "SaveImage",
    "UpscaleModelLoader",
    "RepeatImageBatch",
    "ImageBatch",
}

#: The VideoHelperSuite node types the shipped example uses.  They are a
#: **dependency of the example**, not of the pack: the README documents that
#: the video example needs the VideoHelperSuite pack.
VHS_NODE_WHITELIST = {
    "VHS_LoadVideo",
    "VHS_VideoCombine",
}

#: The node ids the pack registers - the compatibility guarantee of P4.
EXPECTED_PACK_NODE_IDS = {
    "KEEP_ModelLoader",
    "KEEP_FaceUpscaleImage",
    "KEEP_ProcessImageSequence",
}

#: The socket types that are **widgets** in the V3 schema (they are the
#: ``widgets_values`` of a saved workflow).
WIDGET_SOCKET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "MULTICOMBO"}

#: The png magic bytes.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _workflow_files() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pack_node_ids() -> set[str]:
    """The node ids the pack registers - parsed from the sources.

    The regex is the one the sources use (``node_id="..."`` in the V3
    schemas), so the whitelist of the test follows the code, not a copy of it.
    """
    ids: set[str] = set()
    for source in sorted(NODES_DIR.glob("*.py")):
        ids.update(re.findall(r'node_id="([^"]+)"', source.read_text(encoding="utf-8")))
    return ids


def _widget_inputs(schema) -> list:
    return [i for i in schema.inputs if i.get_io_type() in WIDGET_SOCKET_TYPES]


def _socket_inputs(schema) -> list:
    return [i for i in schema.inputs if i.get_io_type() not in WIDGET_SOCKET_TYPES]


def _output_name(output) -> str:
    """The name a saved workflow uses for an output of the schema."""
    return getattr(output, "display_name", None) or getattr(output, "id", "")


@pytest.fixture(scope="module")
def pack_schemas() -> dict:
    """The V3 schema of every pack node, by node id."""
    module = importlib.import_module("ComfyUI_KEEP.nodes")
    schemas = {}
    for class_name in module.__all__:
        schema = getattr(module, class_name).define_schema()
        schemas[schema.node_id] = schema
    return schemas


@pytest.mark.unit
def test_the_examples_directory_holds_the_shipped_workflow():
    """The canonical examples directory holds the shipped example workflow."""
    names = {path.name for path in _workflow_files()}
    assert "keep-workflow.json" in names, (
        "the shipped example workflow must stay in example_workflows/"
    )


@pytest.mark.unit
def test_every_example_workflow_parses():
    """Every example is a parseable ComfyUI UI workflow."""
    for path in _workflow_files():
        data = _load(path)
        assert isinstance(data, dict), f"{path.name} is not a workflow object"
        assert isinstance(data.get("nodes"), list) and data["nodes"], (
            f"{path.name} has no nodes"
        )
        assert isinstance(data.get("links"), list), f"{path.name} has no links list"
        assert data.get("version") == 0.4, (
            f"{path.name} is not in the UI format 0.4 (got {data.get('version')!r})"
        )


@pytest.mark.unit
def test_the_pack_registers_the_three_migrated_node_ids():
    """The ids the examples whitelist are the ids the pack registers."""
    assert _pack_node_ids() == EXPECTED_PACK_NODE_IDS, (
        f"the pack registers {sorted(_pack_node_ids())}, "
        f"the examples expect {sorted(EXPECTED_PACK_NODE_IDS)}"
    )


@pytest.mark.unit
def test_every_node_type_is_a_pack_core_or_vhs_node():
    """No example uses a node type that does not exist."""
    known = _pack_node_ids() | CORE_NODE_WHITELIST | VHS_NODE_WHITELIST
    for path in _workflow_files():
        data = _load(path)
        types = {node.get("type") for node in data["nodes"]}
        unknown = {t for t in types if t not in known}
        assert not unknown, (
            f"{path.name} uses node types that do not exist: {sorted(unknown)}"
        )


@pytest.mark.unit
def test_every_workflow_has_its_preview_image():
    """Every example ships the preview image that belongs to it."""
    for path in _workflow_files():
        preview = path.with_suffix(".png")
        assert preview.is_file(), f"the preview {preview.name} of {path.name} is missing"
        data = preview.read_bytes()
        assert data[:8] == PNG_MAGIC, f"{preview.name} is not a PNG file"
        assert len(data) > 0, f"{preview.name} is empty"


@pytest.mark.unit
def test_the_links_of_every_workflow_are_consistent():
    """Every link of every example points at nodes and slots that exist."""
    for path in _workflow_files():
        data = _load(path)
        nodes = {node["id"]: node for node in data["nodes"]}
        assert len(nodes) == len(data["nodes"]), f"{path.name} has duplicate node ids"

        link_ids = []
        for link in data["links"]:
            assert isinstance(link, list) and len(link) == 6, (
                f"{path.name} has a malformed link: {link!r}"
            )
            link_id, source_id, source_slot, target_id, target_slot, kind = link
            link_ids.append(link_id)
            assert source_id in nodes, f"{path.name}: link {link_id} has no source node"
            assert target_id in nodes, f"{path.name}: link {link_id} has no target node"
            source = nodes[source_id]
            target = nodes[target_id]
            assert source_slot < len(source.get("outputs", [])), (
                f"{path.name}: link {link_id} uses the output {source_slot} of "
                f"{source.get('type')}, which has {len(source.get('outputs', []))}"
            )
            assert target_slot < len(target.get("inputs", [])), (
                f"{path.name}: link {link_id} uses the input {target_slot} of "
                f"{target.get('type')}, which has {len(target.get('inputs', []))}"
            )
            source_type = source["outputs"][source_slot].get("type")
            target_type = target["inputs"][target_slot].get("type")
            assert source_type == target_type == kind, (
                f"{path.name}: link {link_id} connects {source_type} to {target_type} "
                f"(the link says {kind})"
            )
        assert len(link_ids) == len(set(link_ids)), f"{path.name} has duplicate link ids"

        # every connected input refers to a link that exists and points back
        for node in data["nodes"]:
            for index, socket in enumerate(node.get("inputs", [])):
                link_id = socket.get("link")
                if link_id is None:
                    continue
                assert link_id in link_ids, (
                    f"{path.name}: the input {socket.get('name')} of node "
                    f"{node['id']} refers to the unknown link {link_id}"
                )
                link = next(l for l in data["links"] if l[0] == link_id)
                assert link[3] == node["id"] and link[4] == index, (
                    f"{path.name}: the link {link_id} does not point at the input "
                    f"{index} of node {node['id']}"
                )

        # every output lists exactly the links that start at it
        for node in data["nodes"]:
            for slot, output in enumerate(node.get("outputs", [])):
                listed = output.get("links") or []
                expected = [
                    link[0]
                    for link in data["links"]
                    if link[1] == node["id"] and link[2] == slot
                ]
                assert sorted(listed) == sorted(expected), (
                    f"{path.name}: the output {slot} of node {node['id']} lists the "
                    f"links {listed}, the graph has {expected}"
                )

        assert data.get("last_node_id", 0) >= max(nodes, default=0), (
            f"{path.name}: last_node_id is smaller than the highest node id"
        )
        assert data.get("last_link_id", 0) >= max(link_ids, default=0), (
            f"{path.name}: last_link_id is smaller than the highest link id"
        )


@pytest.mark.unit
def test_the_pack_nodes_of_every_workflow_match_their_schema(pack_schemas):
    """The pack nodes of the examples fit the V3 schema of the pack.

    This is the "the example still loads with the migrated ids" check: a
    workflow with the wrong number of widget values, an unknown socket name or
    an illegal widget value is rejected by ComfyUI.
    """
    for path in _workflow_files():
        data = _load(path)
        for node in data["nodes"]:
            schema = pack_schemas.get(node.get("type"))
            if schema is None:
                continue  # a core or VHS node - not ours to check

            widgets = _widget_inputs(schema)
            values = node.get("widgets_values")
            assert isinstance(values, list), (
                f"{path.name}: the widget values of {node['type']} are not a list"
            )
            assert len(values) == len(widgets), (
                f"{path.name}: {node['type']} has {len(values)} widget values, "
                f"the schema has {len(widgets)} widgets"
            )

            for widget, value in zip(widgets, values):
                kind = widget.get_io_type()
                if kind == "COMBO":
                    assert value in (widget.options or []), (
                        f"{path.name}: {value!r} is not an option of the widget "
                        f"{widget.id} ({widget.options})"
                    )
                elif kind == "BOOLEAN":
                    assert isinstance(value, bool), (
                        f"{path.name}: the widget {widget.id} of {node['type']} is "
                        f"not a boolean ({value!r})"
                    )
                elif kind in {"INT", "FLOAT"}:
                    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                        f"{path.name}: the widget {widget.id} of {node['type']} is "
                        f"not a number ({value!r})"
                    )
                    if widget.min is not None:
                        assert value >= widget.min, (
                            f"{path.name}: the widget {widget.id} of {node['type']} is "
                            f"below its minimum ({value} < {widget.min})"
                        )
                    if widget.max is not None:
                        assert value <= widget.max, (
                            f"{path.name}: the widget {widget.id} of {node['type']} is "
                            f"above its maximum ({value} > {widget.max})"
                        )
                elif kind == "STRING":
                    assert isinstance(value, str), (
                        f"{path.name}: the widget {widget.id} of {node['type']} is "
                        f"not a string ({value!r})"
                    )

            sockets = {socket.id: socket for socket in _socket_inputs(schema)}
            listed = {socket.get("name"): socket for socket in node.get("inputs", [])}
            for name, socket in listed.items():
                assert name in sockets, (
                    f"{path.name}: {node['type']} has the input {name!r}, which the "
                    f"schema does not know"
                )
                assert socket.get("type") == sockets[name].get_io_type(), (
                    f"{path.name}: the input {name!r} of {node['type']} is "
                    f"{socket.get('type')}, the schema says {sockets[name].get_io_type()}"
                )
            for socket in _socket_inputs(schema):
                if not socket.optional:
                    assert socket.id in listed, (
                        f"{path.name}: {node['type']} misses the required input "
                        f"{socket.id!r}"
                    )
                    assert listed[socket.id].get("link") is not None, (
                        f"{path.name}: the required input {socket.id!r} of "
                        f"{node['type']} is not connected"
                    )

            expected_outputs = {
                _output_name(output): output.get_io_type() for output in schema.outputs
            }
            actual_outputs = {
                output.get("name"): output.get("type") for output in node.get("outputs", [])
            }
            assert actual_outputs == expected_outputs, (
                f"{path.name}: the outputs of {node['type']} are {actual_outputs}, "
                f"the schema says {expected_outputs}"
            )


@pytest.mark.unit
def test_the_shipped_example_uses_the_v3_ids_and_the_vhs_nodes():
    """The shipped example uses the migrated ids and the VHS dependency."""
    data = _load(EXAMPLES_DIR / "keep-workflow.json")
    types = {node.get("type") for node in data["nodes"]}
    assert EXPECTED_PACK_NODE_IDS <= types, (
        f"the shipped example must use the three pack nodes, it uses {sorted(types)}"
    )
    assert VHS_NODE_WHITELIST <= types, (
        "the shipped example is a video workflow - it needs the VideoHelperSuite "
        f"nodes {sorted(VHS_NODE_WHITELIST)}"
    )
