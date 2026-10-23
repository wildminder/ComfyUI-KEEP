"""KEEP single image node (V1 schema - P3.1).

The class body is the V1 class of the former root ``nodes.py``, moved
**verbatim**: the socket types, the widget names, their defaults and their
order are compatibility guarantees of the pack.
"""

import torch
import traceback

try:  # the package import (ComfyUI loads the node package)
    from ..src.keep_image_utils import comfy_image_to_cv2, cv2_to_comfy_image
    from ..src.keep_model_loader import KEEPModelPack
    from ..src.keep_processor import KEEPFaceProcessor
    from ..src.keep_logger import logger
except ImportError:  # the flat import (the repository root on sys.path)
    from src.keep_image_utils import comfy_image_to_cv2, cv2_to_comfy_image
    from src.keep_model_loader import KEEPModelPack
    from src.keep_processor import KEEPFaceProcessor
    from src.keep_logger import logger


class KEEP_FaceUpscaleImageNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "keep_model": ("KEEP_MODEL_PACK",),
                "final_upscale_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1, "tooltip": "The final upscaling factor for the output image. The image will be resized to this scale after processing."}),
                "has_aligned_face": ("BOOLEAN", {"default": False, "tooltip": "Check if the input image is an already aligned 512x512 face."}),
                "only_center_face": ("BOOLEAN", {"default": True, "tooltip": "If the image has multiple faces, only process the one closest to the center."}),
                "draw_bounding_box": ("BOOLEAN", {"default": False, "tooltip": "Draw a bounding box around the detected face on the output image."}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_face_image"
    CATEGORY = "ComfyUI-KEEP"

    def upscale_face_image(self, image: torch.Tensor, keep_model: KEEPModelPack,
                           final_upscale_factor, has_aligned_face, only_center_face, draw_bounding_box):
        if not isinstance(keep_model, KEEPModelPack):
            logger.error(f"Invalid KEEP Model Pack provided. Expected KEEPModelPack, got {type(keep_model)}")
            return (None,)

        try:
            keep_model.load_device()
            cv2_img_single = comfy_image_to_cv2(image[0].unsqueeze(0))

            processor = KEEPFaceProcessor(keep_model)
            processed_cv2_img = processor.process_image(
                cv2_image_orig=cv2_img_single,
                final_upscale_factor=final_upscale_factor,
                has_aligned=has_aligned_face,
                only_center_face=only_center_face,
                draw_box=draw_bounding_box
            )
            output_image_single = cv2_to_comfy_image(processed_cv2_img)
            return (output_image_single,)
        except Exception as e:
            logger.error(f"Error processing single image: {e}")
            traceback.print_exc()
            return (None,)
        finally:
            keep_model.offload()
