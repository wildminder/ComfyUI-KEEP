"""P3.1 - the node classes live in ``nodes/``, one file per node.

P3.1 is a **pure move**: the V1 classes of the former root ``nodes.py`` are
moved verbatim into ``nodes/``.  The tests below therefore assert the
*compatibility contract* of the pack against the **live** classes:

* the node ids, the display names and the category,
* the socket types (IMAGE / KEEP_MODEL_PACK / UPSCALE_MODEL),
* every widget: its name, its default, its min/max/step and its order.

A move that changes any of these would break existing user workflows, so the
snapshot is the real acceptance criterion of this step.
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

#: The expected widget contract of every node (V1 ``INPUT_TYPES``).
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

#: The V1 node metadata (return types, function name, category).
EXPECTED_NODE_METADATA = {
    "KEEP_ModelLoader": {
        "RETURN_TYPES": ("KEEP_MODEL_PACK",),
        "RETURN_NAMES": ("keep_model_pack",),
        "FUNCTION": "load_model_pack",
        "CATEGORY": "ComfyUI-KEEP",
    },
    "KEEP_FaceUpscaleImage": {
        "RETURN_TYPES": ("IMAGE",),
        "FUNCTION": "upscale_face_image",
        "CATEGORY": "ComfyUI-KEEP",
    },
    "KEEP_ProcessImageSequence": {
        "RETURN_TYPES": ("IMAGE",),
        "RETURN_NAMES": ("processed_images",),
        "FUNCTION": "process_sequence",
        "CATEGORY": "ComfyUI-KEEP",
    },
}


@pytest.fixture(scope="module")
def nodes_package():
    """The ``nodes`` package of the node pack (imported under the alias)."""
    return importlib.import_module("ComfyUI_KEEP.nodes")


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
def test_node_class_mappings_has_exactly_the_three_ids(nodes_package):
    """``NODE_CLASS_MAPPINGS`` holds exactly the three pack ids."""
    mappings = nodes_package.NODE_CLASS_MAPPINGS
    assert sorted(mappings) == sorted(PACK_NODE_IDS), sorted(mappings)
    for node_id, node_class in mappings.items():
        assert isinstance(node_class, type), node_id
        assert node_class.__name__.endswith("Node"), node_class.__name__


@pytest.mark.unit
def test_display_name_mappings_are_unchanged(nodes_package):
    """The display names of the three nodes are unchanged."""
    assert nodes_package.NODE_DISPLAY_NAME_MAPPINGS == PACK_DISPLAY_NAMES


@pytest.mark.unit
def test_the_widget_contract_is_unchanged(nodes_package):
    """Every widget keeps its name, default, min/max/step and order."""
    for node_id, expected in EXPECTED_INPUT_TYPES.items():
        node_class = nodes_package.NODE_CLASS_MAPPINGS[node_id]
        actual = node_class.INPUT_TYPES()
        assert list(actual) == list(expected), (
            f"{node_id}: the input sections changed ({list(actual)} != {list(expected)})"
        )
        for section, widgets in expected.items():
            assert list(actual[section]) == list(widgets), (
                f"{node_id}/{section}: the widget order changed "
                f"({list(actual[section])} != {list(widgets)})"
            )
            for name, spec in widgets.items():
                assert actual[section][name] == spec, (
                    f"{node_id}/{section}/{name}: {actual[section][name]} != {spec}"
                )


@pytest.mark.unit
def test_the_node_metadata_is_unchanged(nodes_package):
    """The socket types, the function names and the category are unchanged."""
    for node_id, expected in EXPECTED_NODE_METADATA.items():
        node_class = nodes_package.NODE_CLASS_MAPPINGS[node_id]
        for attribute, value in expected.items():
            assert getattr(node_class, attribute) == value, (
                f"{node_id}.{attribute}: {getattr(node_class, attribute)!r} != {value!r}"
            )


@pytest.mark.unit
def test_the_example_workflow_ids_exist_in_the_mappings(nodes_package):
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
    assert used <= set(nodes_package.NODE_CLASS_MAPPINGS), sorted(
        used - set(nodes_package.NODE_CLASS_MAPPINGS)
    )


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
