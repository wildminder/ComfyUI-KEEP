"""The KEEP node classes - one file per node (P3.1).

The package still registers as **V1** in this phase: the root ``__init__.py``
exports the ``NODE_CLASS_MAPPINGS`` built here, so an existing ComfyUI
installation keeps loading the node exactly as before.  P4.4 replaces them
with the V3 ``comfy_entrypoint``; until then the mappings below are the
single source of truth for the node ids and the display names.
"""

from .keep_model_loader import KEEP_ModelLoaderNode
from .keep_face_upscale_image import KEEP_FaceUpscaleImageNode
from .keep_process_image_sequence import KEEP_ProcessImageSequenceNode

NODE_CLASS_MAPPINGS = {
    "KEEP_ModelLoader": KEEP_ModelLoaderNode,
    "KEEP_FaceUpscaleImage": KEEP_FaceUpscaleImageNode,
    "KEEP_ProcessImageSequence": KEEP_ProcessImageSequenceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KEEP_ModelLoader": "Load KEEP Models",
    "KEEP_FaceUpscaleImage": "KEEP Single Image",
    "KEEP_ProcessImageSequence": "KEEP Image Sequence",
}

__all__ = [
    "KEEP_ModelLoaderNode",
    "KEEP_FaceUpscaleImageNode",
    "KEEP_ProcessImageSequenceNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
