"""KEEP image sequence node (V1 schema - P3.1).

The class body is the V1 class of the former root ``nodes.py``, moved
**verbatim**: the socket types, the widget names, their defaults and their
order are compatibility guarantees of the pack.
"""

import torch
import traceback

try:  # the package import (ComfyUI loads the node package)
    from ..modules.keep_model_loader import KEEPModelPack
    from ..modules.keep_processor import KEEPFaceProcessor
    from .. import logger
except ImportError:  # the flat import (the repository root on sys.path)
    from modules.keep_model_loader import KEEPModelPack
    from modules.keep_processor import KEEPFaceProcessor
    import logging
    logger = logging.getLogger("ComfyUI-KEEP")


class KEEP_ProcessImageSequenceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "keep_model": ("KEEP_MODEL_PACK",),
                "final_upscale_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1, "tooltip": "The final upscaling factor for the output frames. They will be resized to this scale after processing."}),
                "has_aligned_frames": ("BOOLEAN", {"default": False, "tooltip": "Check if the input frames are already aligned 512x512 faces."}),
                "only_center_face": ("BOOLEAN", {"default": True, "tooltip": "If frames have multiple faces, only process the one closest to the center."}),
                "draw_bounding_box": ("BOOLEAN", {"default": False, "tooltip": "Draw a bounding box around the detected face on the output frames."}),
                "max_clip_length": ("INT", {"default": 20, "min":1, "max":100, "step":1, "tooltip": "Maximum number of frames to process in a single batch to manage VRAM."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_images",)
    FUNCTION = "process_sequence"
    CATEGORY = "ComfyUI-KEEP"

    def process_sequence(self, images: torch.Tensor, keep_model: KEEPModelPack,
                         final_upscale_factor, has_aligned_frames, only_center_face, draw_bounding_box,
                         max_clip_length):

        if not isinstance(keep_model, KEEPModelPack):
            logger.error(f"Invalid KEEP Model Pack provided. Expected KEEPModelPack, got {type(keep_model)}")
            return (None,)

        try:
            keep_model.load_device()
            processor = KEEPFaceProcessor(keep_model)
            processed_images_tensor = processor.process_image_sequence(
                image_sequence_tensor=images,
                final_upscale_factor=final_upscale_factor,
                has_aligned_frames=has_aligned_frames,
                only_center_face=only_center_face,
                draw_box=draw_bounding_box,
                max_clip_length=max_clip_length
            )
            return (processed_images_tensor,)
        except Exception as e:
            logger.error(f"Error during image sequence processing: {e}")
            traceback.print_exc()
            return (None,)
        finally:
            keep_model.offload()
