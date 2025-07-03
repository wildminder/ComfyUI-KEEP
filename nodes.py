import torch
from comfy import model_management
import traceback

from . import logger # Import the central logger
from .modules.utils import comfy_image_to_cv2, cv2_to_comfy_image, KEEP_MODEL_CONFIGS
from .modules.keep_model_loader import KEEPModelLoader, KEEPModelPack
from .modules.keep_processor import KEEPFaceProcessor

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