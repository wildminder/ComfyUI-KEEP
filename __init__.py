"""The ComfyUI-KEEP custom node package (V3 entry point - P4.4).

The package registers itself as a **V3 extension**: the module level
``comfy_entrypoint`` returns a ``ComfyExtension`` that lists the three node
classes.  The V1 mappings are **gone** on purpose: ComfyUI checks the V1
mappings first (``nodes.py:2305``) and would silently skip the V3 extension
of a package that exports both.

The node ids, the display names and the category stay the same either way -
the V3 schema of the nodes carries them.
"""

from comfy_api.latest import ComfyExtension, io

from .src.keep_logger import logger
from .src import keep_paths
from .nodes import (
    KEEP_FaceUpscaleImageNode,
    KEEP_ModelLoaderNode,
    KEEP_ProcessImageSequenceNode,
)

# P3.5: the model folders of the pack (``keep_models``, ``facedetection`` and
# ``gmflow``) are registered with ComfyUI's ``folder_paths``, so every weight
# path is resolved by the host.
keep_paths.register_model_folders()


class KEEPExtension(ComfyExtension):
    """The V3 extension of the KEEP pack."""

    async def on_load(self) -> None:
        """Nothing to do - the model folders are registered at import time."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """The three node classes of the pack."""
        return [
            KEEP_ModelLoaderNode,
            KEEP_FaceUpscaleImageNode,
            KEEP_ProcessImageSequenceNode,
        ]


async def comfy_entrypoint() -> KEEPExtension:
    """The entry point ComfyUI calls (the V3 way)."""
    return KEEPExtension()


__all__ = ["KEEPExtension", "comfy_entrypoint", "logger"]
