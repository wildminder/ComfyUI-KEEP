"""The ComfyUI-KEEP custom node package.

The entry point is deliberately minimal: it sets up the logger and exports
the V1 node mappings.  P4.4 replaces the mappings with the V3
``comfy_entrypoint`` (the node ids, the display names and the category stay
the same either way).
"""

from .src.keep_logger import logger
from .src import keep_paths
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# P3.5: the model folders of the pack (``keep_models``, ``facedetection`` and
# ``gmflow``) are registered with ComfyUI's ``folder_paths``, so every weight
# path is resolved by the host.
keep_paths.register_model_folders()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "logger"]
