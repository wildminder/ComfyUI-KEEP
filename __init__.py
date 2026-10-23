"""The ComfyUI-KEEP custom node package.

The entry point is deliberately minimal: it sets up the logger and exports
the V1 node mappings.  P4.4 replaces the mappings with the V3
``comfy_entrypoint`` (the node ids, the display names and the category stay
the same either way).
"""

from .src.keep_logger import logger
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# The model folders are registered with ComfyUI's ``folder_paths`` in P3.5
# (``src/keep_paths.py``); the registration is wired into this entry point
# there.

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "logger"]
