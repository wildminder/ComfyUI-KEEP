"""P4.1 - the V3 schema of the nodes keeps the compatibility contract.

The node ids, the display names, the category, the socket types and every
widget (name, default, min/max/step and order) are compatibility guarantees
of the pack: existing user workflows must keep working unchanged.

The contract below is the **recorded V1 ``INPUT_TYPES``** (P0.1).  After the
V3 migration the widgets live in the schema, so the test converts the V3
schema into the very same shape and compares it with the record.  A V3
schema that drops a widget, renames one, changes a default or reorders the
inputs fails here.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NODES_DIR = REPO_ROOT / "nodes"
EXAMPLES_DIR = REPO_ROOT / "example_workflows"

#: The node ids of the pack - they must never change.
PACK_NODE_IDS = (
    "KEEP_ModelLoader",
    "KEEP_FaceUpscaleImage",
    "KEEP_ProcessImageSequence",
)

#: The display names of the pack - they must never change.
PACK_DISPLAY_NAMES = {
    "KEEP_ModelLoader": "Load KEEP Models",
    "KEEP_FaceUpscaleImage": "KEEP Single Image",
    "KEEP_ProcessImageSequence": "KEEP Image Sequence",
}

#: The category of the pack - it must never change.
PACK_CATEGORY = "ComfyUI-KEEP"

#: The expected widget contract of every node (the recorded V1
#: ``INPUT_TYPES`` - the source of truth of this file).
EXPECTED_INPUT_TYPES = {
    "KEEP_ModelLoader": {
        "required": {
            "model": (["KEEP", "Asian"], {"default": "KEEP"}),
            "detection_model": (
                [
                    "retinaface_resnet50",
                    "retinaface_mobile0.25",
                    "YOLOv5l",
                    "YOLOv5n",
                ],
                {"default": "retinaface_resnet50"},
            ),
        },
        "optional": {
            "bg_upscale_model": ("UPSCALE_MODEL",),
            "face_upscale_model": ("UPSCALE_MODEL",),
        },
    },
    "KEEP_FaceUpscaleImage": {
        "required": {
            "image": ("IMAGE",),
            "keep_model": ("KEEP_MODEL_PACK",),
            "final_upscale_factor": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "The final upscaling factor for the output image. The image will be resized to this scale after processing.",
                },
            ),
            "has_aligned_face": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Check if the input image is an already aligned 512x512 face.",
                },
            ),
            "only_center_face": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "If the image has multiple faces, only process the one closest to the center.",
                },
            ),
            "draw_bounding_box": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Draw a bounding box around the detected face on the output image.",
                },
            ),
        }
    },
    "KEEP_ProcessImageSequence": {
        "required": {
            "images": ("IMAGE",),
            "keep_model": ("KEEP_MODEL_PACK",),
            "final_upscale_factor": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "The final upscaling factor for the output frames. They will be resized to this scale after processing.",
                },
            ),
            "has_aligned_frames": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Check if the input frames are already aligned 512x512 faces.",
                },
            ),
            "only_center_face": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "If frames have multiple faces, only process the one closest to the center.",
                },
            ),
            "draw_bounding_box": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Draw a bounding box around the detected face on the output frames.",
                },
            ),
            "max_clip_length": (
                "INT",
                {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Maximum number of frames to process in a single batch to manage VRAM.",
                },
            ),
        }
    },
}

#: The expected output sockets of every node (the recorded V1
#: ``RETURN_TYPES`` / ``RETURN_NAMES``).
EXPECTED_OUTPUTS = {
    "KEEP_ModelLoader": {
        "types": ("KEEP_MODEL_PACK",),
        "names": ("keep_model_pack",),
    },
    "KEEP_FaceUpscaleImage": {
        "types": ("IMAGE",),
        # the V1 node had no RETURN_NAMES - the socket type is the name
        "names": ("IMAGE",),
    },
    "KEEP_ProcessImageSequence": {
        "types": ("IMAGE",),
        "names": ("processed_images",),
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
#: The class of every node id (the names are stable - they are the class
#: names of the pack).
NODE_CLASS_NAMES = {
    "KEEP_ModelLoader": "KEEP_ModelLoaderNode",
    "KEEP_FaceUpscaleImage": "KEEP_FaceUpscaleImageNode",
    "KEEP_ProcessImageSequence": "KEEP_ProcessImageSequenceNode",
}

#: The nodes that are **already** migrated to the V3 schema.  The list grows
#: with the migration steps (P4.1 the loader, P4.2 the single image node,
#: P4.3 the sequence node); the contract test below works with both
#: representations, so it stays meaningful in every step.
V3_NODE_IDS = ("KEEP_ModelLoader", "KEEP_FaceUpscaleImage")


def _pack_nodes(nodes_package) -> dict:
    """``{node_id: node class}`` - read from the exports of ``nodes/``."""
    return {
        node_id: getattr(nodes_package, class_name)
        for node_id, class_name in NODE_CLASS_NAMES.items()
    }


def _is_v3(node_class) -> bool:
    from comfy_api.latest import io

    return isinstance(node_class, type) and issubclass(node_class, io.ComfyNode)


def _v1_contract(input_types: dict) -> dict:
    """The recorded V1 ``INPUT_TYPES`` in the shape of the node contract."""
    contract: dict[str, list[dict]] = {"required": [], "optional": []}
    for section, widgets in input_types.items():
        for name, spec in widgets.items():
            io_type = spec[0]
            options = spec[1] if len(spec) > 1 else {}
            entry = {
                "id": name,
                "type": io_type if isinstance(io_type, str) else "COMBO",
                "tooltip": options.get("tooltip"),
            }
            if not isinstance(io_type, str):
                entry["options"] = list(io_type)
            for field in ("default", "min", "max", "step"):
                if field in options:
                    entry[field] = options[field]
            contract[section].append(entry)
    return contract


def _schema_contract(schema) -> dict:
    """The V3 schema in the shape of the node contract."""
    contract: dict[str, list[dict]] = {"required": [], "optional": []}
    for item in schema.inputs:
        section = "optional" if item.optional else "required"
        entry = {
            "id": item.id,
            "type": item.get_io_type(),
            "tooltip": item.tooltip,
        }
        for field in ("default", "min", "max", "step", "options"):
            value = getattr(item, field, None)
            if value is not None:
                entry[field] = value
        contract[section].append(entry)
    return contract


def _node_contract(node_class) -> dict:
    """The contract of a node - from its V3 schema or from its V1 class.

    The V3 migration moves the widgets from ``INPUT_TYPES`` into the
    schema; the contract of the pack is the same either way, so the test
    compares the *contract*, not the representation.
    """
    if _is_v3(node_class):
        schema = node_class.GET_SCHEMA()
        return {
            "v3": True,
            "node_id": schema.node_id,
            "display_name": schema.display_name,
            "category": schema.category,
            "description": schema.description,
            "inputs": _schema_contract(schema),
            "output_types": tuple(output.get_io_type() for output in schema.outputs),
            "output_names": tuple(
                output.display_name if output.display_name else output.get_io_type()
                for output in schema.outputs
            ),
            "tooltips": {item.id: item.tooltip for item in schema.inputs},
        }
    input_types = node_class.INPUT_TYPES()
    return {
        "v3": False,
        "node_id": None,
        "display_name": None,
        "category": node_class.CATEGORY,
        "description": getattr(node_class, "DESCRIPTION", ""),
        "inputs": _v1_contract(input_types),
        "output_types": tuple(node_class.RETURN_TYPES),
        "output_names": tuple(
            getattr(node_class, "RETURN_NAMES", node_class.RETURN_TYPES)
        ),
        "tooltips": {
            entry["id"]: entry["tooltip"]
            for entries in _v1_contract(input_types).values()
            for entry in entries
        },
    }


@pytest.fixture(scope="module")
def nodes_package():
    """The ``nodes`` package of the node pack (imported under the alias)."""
    return importlib.import_module("ComfyUI_KEEP.nodes")


@pytest.fixture(scope="module")
def pack_nodes(nodes_package):
    """``{node_id: node class}`` of the three pack nodes."""
    return _pack_nodes(nodes_package)


# ---------------------------------------------------------------------------
# the layout
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_every_node_has_its_own_file():
    """One file per node, and the old root ``nodes.py`` is gone."""
    for filename in (
        "keep_model_loader.py",
        "keep_face_upscale_image.py",
        "keep_process_image_sequence.py",
        "__init__.py",
    ):
        assert (NODES_DIR / filename).is_file(), f"missing {NODES_DIR / filename}"
    assert not (REPO_ROOT / "nodes.py").exists(), (
        "the root nodes.py must be deleted - the classes live in nodes/ now"
    )


@pytest.mark.unit
def test_the_pack_registers_exactly_the_three_node_ids(pack_nodes):
    """The three node ids of the pack - and no other node."""
    assert sorted(pack_nodes) == sorted(PACK_NODE_IDS), sorted(pack_nodes)


@pytest.mark.unit
def test_every_node_defines_a_v3_schema_and_execute(pack_nodes):
    """Every migrated node is a V3 node: ``define_schema`` + ``execute``."""
    for node_id in V3_NODE_IDS:
        node_class = pack_nodes[node_id]
        assert _is_v3(node_class), f"{node_id} is not a V3 node yet"
        schema = node_class.GET_SCHEMA()
        assert schema.node_id == node_id, node_id
        assert callable(node_class.execute), node_id


# ---------------------------------------------------------------------------
# the schema contract
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_display_names_and_the_category_are_unchanged(pack_nodes, nodes_package):
    """The display names of the three nodes and the category are unchanged."""
    v1_display_names = getattr(nodes_package, "NODE_DISPLAY_NAME_MAPPINGS", {})
    for node_id, node_class in pack_nodes.items():
        contract = _node_contract(node_class)
        display_name = contract["display_name"] or v1_display_names.get(node_id)
        assert display_name == PACK_DISPLAY_NAMES[node_id], node_id
        assert contract["category"] == PACK_CATEGORY, node_id
        if contract["v3"]:
            assert contract["description"], f"{node_id} needs a schema description"


@pytest.mark.unit
def test_the_widget_contract_is_unchanged(pack_nodes):
    """Every widget keeps its name, type, default, min/max/step and order."""
    for node_id, expected in EXPECTED_INPUT_TYPES.items():
        contract = _node_contract(pack_nodes[node_id])
        actual = contract["inputs"]
        expected_contract = _v1_contract(expected)

        assert list(actual) == list(expected_contract), (
            f"{node_id}: the input sections changed ({list(actual)} != "
            f"{list(expected_contract)})"
        )
        for section in expected_contract:
            actual_entries = actual[section]
            expected_entries = expected_contract[section]
            assert [entry["id"] for entry in actual_entries] == [
                entry["id"] for entry in expected_entries
            ], f"{node_id}/{section}: the widget order changed"
            for actual_entry, expected_entry in zip(actual_entries, expected_entries):
                for key, value in expected_entry.items():
                    if key == "tooltip" and value is None:
                        # the V1 node had no tooltip here - the V3 schema
                        # requires one (the plan's P4.1: every input gets a
                        # tooltip), but the text is new
                        if contract["v3"]:
                            assert actual_entry["tooltip"], (
                                f"{node_id}/{actual_entry['id']}: "
                                "the input needs a tooltip"
                            )
                        continue
                    assert actual_entry.get(key) == value, (
                        f"{node_id}/{actual_entry['id']}/{key}: "
                        f"{actual_entry.get(key)!r} != {value!r}"
                    )


@pytest.mark.unit
def test_every_input_has_a_tooltip(pack_nodes):
    """Every input of every V3 node carries a tooltip."""
    for node_id, node_class in pack_nodes.items():
        if not _is_v3(node_class):
            continue
        schema = node_class.GET_SCHEMA()
        assert schema.inputs, node_id
        for item in schema.inputs:
            assert isinstance(item.tooltip, str) and item.tooltip.strip(), (
                f"{node_id}/{item.id}: the input has no tooltip"
            )


@pytest.mark.unit
def test_the_output_sockets_are_unchanged(pack_nodes):
    """The output socket types and their names are unchanged."""
    for node_id, expected in EXPECTED_OUTPUTS.items():
        contract = _node_contract(pack_nodes[node_id])
        assert contract["output_types"] == expected["types"], (
            f"{node_id}: {contract['output_types']} != {expected['types']}"
        )
        assert contract["output_names"] == expected["names"], (
            f"{node_id}: {contract['output_names']} != {expected['names']}"
        )


@pytest.mark.unit
def test_the_example_workflow_ids_exist_in_the_pack(pack_nodes):
    """The ids of the shipped example are registered by the node."""
    workflow = json.loads(
        (EXAMPLES_DIR / "keep-workflow.json").read_text(encoding="utf-8")
    )
    used = {
        node.get("type")
        for node in workflow["nodes"]
        if isinstance(node.get("type"), str) and node["type"].startswith("KEEP_")
    }
    assert used, "the example must use the KEEP nodes"
    assert used <= set(pack_nodes), sorted(used - set(pack_nodes))


@pytest.mark.unit
def test_the_node_modules_are_importable_under_their_own_names():
    """Every node file is importable as a module of the package."""
    for module_name, class_name in (
        ("keep_model_loader", "KEEP_ModelLoaderNode"),
        ("keep_face_upscale_image", "KEEP_FaceUpscaleImageNode"),
        ("keep_process_image_sequence", "KEEP_ProcessImageSequenceNode"),
    ):
        module = importlib.import_module(f"ComfyUI_KEEP.nodes.{module_name}")
        assert hasattr(module, class_name), f"{module_name} has no {class_name}"
