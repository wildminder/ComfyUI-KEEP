"""KEEP model loader node (V1 schema - P3.1).

The class body is the V1 class of the former root ``nodes.py``, moved
**verbatim**: the node id, the display name, the category, the widget names,
their defaults and their order are compatibility guarantees of the pack.

The engines are imported with the DyPE dual-mode import, so the node module
works both as part of the node package (how ComfyUI loads it) and as a flat
module (the repository root on ``sys.path``).
"""

try:  # the package import (ComfyUI loads the node package)
    from ..src.keep_model_configs import KEEP_MODEL_CONFIGS
    from ..src.keep_model_loader import KEEPModelLoader
except ImportError:  # the flat import (the repository root on sys.path)
    from src.keep_model_configs import KEEP_MODEL_CONFIGS
    from src.keep_model_loader import KEEPModelLoader

GLOBAL_KEEP_MODEL_LOADER = None


def get_keep_model_loader():
    global GLOBAL_KEEP_MODEL_LOADER
    if GLOBAL_KEEP_MODEL_LOADER is None:
        GLOBAL_KEEP_MODEL_LOADER = KEEPModelLoader()
    return GLOBAL_KEEP_MODEL_LOADER


class KEEP_ModelLoaderNode:
    _MODEL_TYPES = list(KEEP_MODEL_CONFIGS.keys())

    @classmethod
    def INPUT_TYPES(s):
        detection_models = ['retinaface_resnet50', 'retinaface_mobile0.25', 'YOLOv5l', 'YOLOv5n']
        return {
            "required": {
                "model": (s._MODEL_TYPES, {"default": s._MODEL_TYPES[0] if s._MODEL_TYPES else 'KEEP'}),
                "detection_model": (detection_models, {"default": 'retinaface_resnet50'}),
            },
            "optional": {
                "bg_upscale_model": ("UPSCALE_MODEL",),
                "face_upscale_model": ("UPSCALE_MODEL",),
            }
        }
    RETURN_TYPES = ("KEEP_MODEL_PACK",)
    RETURN_NAMES = ("keep_model_pack",)
    FUNCTION = "load_model_pack"
    CATEGORY = "ComfyUI-KEEP"

    def load_model_pack(self, model, detection_model, bg_upscale_model=None, face_upscale_model=None):
        loader = get_keep_model_loader()
        model_pack = loader.load_keep_model_pack(
            model_type_str=model,
            detection_model_str=detection_model,
            bg_upscale_model=bg_upscale_model,
            face_upscale_model=face_upscale_model
        )
        return (model_pack,)
