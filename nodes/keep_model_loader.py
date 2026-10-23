"""KEEP model loader node (V3 schema - P4.1).

The node id (``KEEP_ModelLoader``), the display name ("Load KEEP Models"),
the category (``ComfyUI-KEEP``) and every widget (name, default, min/max/step
and order) are compatibility guarantees of the pack - the V3 schema below
carries exactly the contract of the former V1 ``INPUT_TYPES``.

The engines are imported with the DyPE dual-mode import, so the node module
works both as part of the node package (how ComfyUI loads it) and as a flat
module (the repository root on ``sys.path``).
"""

from comfy_api.latest import io

try:  # the package import (ComfyUI loads the node package)
    from ..src.keep_model_configs import KEEP_MODEL_CONFIGS
    from ..src.keep_model_loader import KEEPModelLoader
except ImportError:  # the flat import (the repository root on sys.path)
    from src.keep_model_configs import KEEP_MODEL_CONFIGS
    from src.keep_model_loader import KEEPModelLoader

GLOBAL_KEEP_MODEL_LOADER = None

#: The face detectors of the vendored facelib helper (the V1 combo).
DETECTION_MODELS = [
    "retinaface_resnet50",
    "retinaface_mobile0.25",
    "YOLOv5l",
    "YOLOv5n",
]


def get_keep_model_loader():
    """The global loader singleton (the cache key of the pack is unchanged)."""
    global GLOBAL_KEEP_MODEL_LOADER
    if GLOBAL_KEEP_MODEL_LOADER is None:
        GLOBAL_KEEP_MODEL_LOADER = KEEPModelLoader()
    return GLOBAL_KEEP_MODEL_LOADER


class KEEP_ModelLoaderNode(io.ComfyNode):
    """Load the KEEP checkpoint, the face helper and the optional upscalers."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_types = list(KEEP_MODEL_CONFIGS.keys())
        return io.Schema(
            node_id="KEEP_ModelLoader",
            display_name="Load KEEP Models",
            category="ComfyUI-KEEP",
            description=(
                "Load the KEEP face restoration model together with the face "
                "detection / parsing models. The result is a model pack for "
                "the KEEP Single Image and KEEP Image Sequence nodes."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=model_types,
                    default=model_types[0] if model_types else "KEEP",
                    tooltip="The KEEP checkpoint to load (KEEP or Asian).",
                ),
                io.Combo.Input(
                    "detection_model",
                    options=list(DETECTION_MODELS),
                    default="retinaface_resnet50",
                    tooltip="The face detector of the face restoration helper.",
                ),
                io.Custom("UPSCALE_MODEL").Input(
                    "bg_upscale_model",
                    optional=True,
                    tooltip=(
                        "An optional ComfyUI upscale model for the background "
                        "of the input image."
                    ),
                ),
                io.Custom("UPSCALE_MODEL").Input(
                    "face_upscale_model",
                    optional=True,
                    tooltip=(
                        "An optional ComfyUI upscale model for the restored "
                        "face patch."
                    ),
                ),
            ],
            outputs=[
                io.Custom("KEEP_MODEL_PACK").Output(
                    id="keep_model_pack",
                    display_name="keep_model_pack",
                    tooltip="The loaded KEEP model pack.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        detection_model,
        bg_upscale_model=None,
        face_upscale_model=None,
    ) -> io.NodeOutput:
        loader = get_keep_model_loader()
        model_pack = loader.load_keep_model_pack(
            model_type_str=model,
            detection_model_str=detection_model,
            bg_upscale_model=bg_upscale_model,
            face_upscale_model=face_upscale_model,
        )
        return io.NodeOutput(model_pack)
