import torch
from comfy import model_management
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
                "bg_upscale": ("BOOLEAN", {"default": False}),
                "face_upscale": ("BOOLEAN", {"default": False}),
                "bg_tile_size": ("INT", {"default": 400, "min": 64, "max": 2048, "step": 32}),
            }
        }
    RETURN_TYPES = ("KEEP_MODEL_PACK",)
    RETURN_NAMES = ("keep_model_pack",)
    FUNCTION = "load_model_pack"
    CATEGORY = "ComfyUI-KEEP"

    def load_model_pack(self, model, detection_model, bg_upscale, face_upscale, bg_tile_size):
        loader = get_keep_model_loader()
        model_pack = loader.load_keep_model_pack(
            model_type_str=model, detection_model_str=detection_model,
            use_bg_upsampler=bg_upscale, use_face_upsampler=face_upscale,
            bg_tile_size=bg_tile_size
        )
        return (model_pack,)

class KEEP_FaceUpscaleImageNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "keep_model": ("KEEP_MODEL_PACK",),
                "has_aligned_face": ("BOOLEAN", {"default": False}),
                "only_center_face": ("BOOLEAN", {"default": True}),
                "draw_bounding_box": ("BOOLEAN", {"default": False}),
                "background_upscale_factor": ("INT", {"default": 1, "min":1, "max":4, "step":1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_face_image"
    CATEGORY = "ComfyUI-KEEP"
    def upscale_face_image(self, image: torch.Tensor, keep_model: KEEPModelPack,
                           has_aligned_face, only_center_face, draw_bounding_box,
                           background_upscale_factor):
        if not isinstance(keep_model, KEEPModelPack):
            # print(f"Error: keep_model is not of type KEEPModelPack. Got {type(keep_model)}")
            return (image,) # Return original on error

        # Assuming the batch size B=1 for a single image node.
        #if image.shape[0] > 1:
        #    print("Warning: KEEP_FaceUpscaleImageNode received batch of images, processing only the first one.")

        # Process first image of batch
        cv2_img_single = comfy_image_to_cv2(image[0].unsqueeze(0))

        processor = KEEPFaceProcessor(keep_model)
        processed_cv2_img = processor.process_image(
            cv2_image_orig=cv2_img_single, has_aligned=has_aligned_face,
            only_center_face=only_center_face, draw_box=draw_bounding_box,
            upscale_factor=background_upscale_factor
        )
        output_image_single = cv2_to_comfy_image(processed_cv2_img)
        return (output_image_single,)


class KEEP_ProcessImageSequenceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "keep_model": ("KEEP_MODEL_PACK",),
                "has_aligned_frames": ("BOOLEAN", {"default": False}), # If input frames are 512x512 aligned faces
                "only_center_face": ("BOOLEAN", {"default": True}),    # Used if not has_aligned
                "draw_bounding_box": ("BOOLEAN", {"default": False}),   # Used if not has_aligned
                "background_upscale_factor": ("INT", {"default": 1, "min":1, "max":4, "step":1}),
                "max_clip_length": ("INT", {"default": 20, "min":1, "max":100, "step":1}), # For KEEP network
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_images",)
    FUNCTION = "process_sequence"
    CATEGORY = "ComfyUI-KEEP"

    def process_sequence(self, images: torch.Tensor, keep_model: KEEPModelPack,
                           has_aligned_frames, only_center_face, draw_bounding_box,
                           background_upscale_factor, max_clip_length):
        
        if not isinstance(keep_model, KEEPModelPack):
            # print(f"Error: keep_model is not of type KEEPModelPack. Got {type(keep_model)}")
            # Return original sequence on error
            return (images,)

        processor = KEEPFaceProcessor(keep_model)
        
        try:
            processed_images_tensor = processor.process_image_sequence(
                image_sequence_tensor=images, # Pass the B,H,W,C tensor directly
                has_aligned_frames=has_aligned_frames,
                only_center_face=only_center_face,
                draw_box=draw_bounding_box,
                upscale_factor=background_upscale_factor,
                max_clip_length=max_clip_length
            )
            return (processed_images_tensor,)

        # Return original sequence on error
        except Exception as e:
            # print(f"Error during image sequence processing: {e}")
            import traceback
            traceback.print_exc()
            return (images,)


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