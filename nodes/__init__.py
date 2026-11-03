"""The KEEP node classes - one file per node (P3.1).

The pack registers as **V3** (P4.4): the root ``__init__.py`` exposes the
module level ``comfy_entrypoint`` and the three node classes below.  The V1
mappings are gone - ComfyUI checks the V1 mappings first (``nodes.py:2305``)
and would silently skip the V3 extension of a package that exports both.

The node ids, the display names and the category live in the schemas of the
classes (the source of truth is ``tests/test_nodes_layout.py``).
"""

from .keep_model_loader import KEEP_ModelLoaderNode
from .keep_face_upscale_image import KEEP_FaceUpscaleImageNode
from .keep_process_image_sequence import KEEP_ProcessImageSequenceNode

__all__ = [
    "KEEP_ModelLoaderNode",
    "KEEP_FaceUpscaleImageNode",
    "KEEP_ProcessImageSequenceNode",
]
