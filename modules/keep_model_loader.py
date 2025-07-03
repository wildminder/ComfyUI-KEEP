import torch
import os
from comfy import model_management
import folder_paths 

from .. import logger # Import from parent __init__.py
from .utils import (
    load_file_from_url_comfy, ARCH_REGISTRY,
    KEEP_MODEL_CONFIGS, FACELIB_MODEL_URLS, FACELIB_DEST_DIR
)

try:
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
except ImportError as e:
    logger.error(f"Critical Import Error in keep_model_loader.py (facelib): {e}")
    class FaceRestoreHelper: pass

class KEEPModelPack:
    def __init__(self, keep_net, face_helper, bg_upscale_model, face_upscale_model, model_type_str):
        self.keep_net = keep_net
        self.face_helper = face_helper
        self.bg_upscale_model = bg_upscale_model
        self.face_upscale_model = face_upscale_model
        self.model_type_str = model_type_str
        self.device = model_management.get_torch_device()
        self.offload_device = model_management.unet_offload_device()

    def load_device(self):
        """Move all models to the main compute device."""
        if self.keep_net is not None:
            self.keep_net.to(self.device)
        # The spandrel model descriptor has a 'model' attribute that needs to be moved
        if self.bg_upscale_model is not None:
            self.bg_upscale_model.model.to(self.device)
        if self.face_upscale_model is not None:
            self.face_upscale_model.model.to(self.device)
        
        if self.face_helper is not None:
            self.face_helper.device = self.device
            if hasattr(self.face_helper, 'face_detector'):
                self.face_helper.face_detector.to(self.device)
            if hasattr(self.face_helper, 'face_parse'):
                self.face_helper.face_parse.to(self.device)

    def offload(self):
        """Move all models to the offload device (CPU)."""
        if self.keep_net is not None:
            self.keep_net.to(self.offload_device)
        if self.bg_upscale_model is not None:
            self.bg_upscale_model.model.to(self.offload_device)
        if self.face_upscale_model is not None:
            self.face_upscale_model.model.to(self.offload_device)
            
        if self.face_helper is not None:
            self.face_helper.device = self.offload_device
            if hasattr(self.face_helper, 'face_detector'):
                self.face_helper.face_detector.to(self.offload_device)
            if hasattr(self.face_helper, 'face_parse'):
                self.face_helper.face_parse.to(self.offload_device)
                
        model_management.soft_empty_cache()

class KEEPModelLoader:
    def __init__(self):
        self.device = model_management.get_torch_device()
        self.offload_device = model_management.unet_offload_device()
        self.loaded_models = {}

    def load_keep_model_pack(self, model_type_str, detection_model_str,
                             bg_upscale_model=None, face_upscale_model=None):
        
        bg_upscaler_present = bg_upscale_model is not None
        face_upscaler_present = face_upscale_model is not None
        cache_key = (model_type_str, detection_model_str, bg_upscaler_present, face_upscaler_present)

        if cache_key in self.loaded_models:
            logger.debug(f"Returning cached base models for {cache_key}")
            cached_pack = self.loaded_models[cache_key]
            model_pack = KEEPModelPack(
                cached_pack.keep_net, 
                cached_pack.face_helper, 
                bg_upscale_model,
                face_upscale_model,
                model_type_str
            )
            return model_pack

        if model_type_str not in KEEP_MODEL_CONFIGS:
            raise ValueError(f"Unknown KEEP model type: {model_type_str}")

        config_entry = KEEP_MODEL_CONFIGS[model_type_str]
        
        keep_model_arch_class = ARCH_REGISTRY.get('KEEP')
        if keep_model_arch_class is None:
            raise RuntimeError("KEEP architecture class not found in ARCH_REGISTRY.")
        
        net = keep_model_arch_class(**config_entry['architecture']).to(self.offload_device)
        
        ckpt_path = load_file_from_url_comfy(
            url=config_entry['url'],
            model_dir_name=config_entry['dest_dir'],
            file_name=os.path.basename(config_entry['url']),
            sha256=config_entry.get('sha256')
        )
        checkpoint = torch.load(ckpt_path, map_location=self.offload_device, weights_only=True)
        
        state_dict_key = 'params_ema' if 'params_ema' in checkpoint else 'params'
        state_dict = checkpoint.get(state_dict_key, checkpoint)

        converted_state_dict = {}
        needs_conversion = any('cross_fuse' in k or 'fuse_convs_dict' in k for k in state_dict.keys())
        if needs_conversion:
            for k, v in state_dict.items():
                new_k = k
                if 'cross_fuse' in k: new_k = k.replace('cross_fuse', 'cfa')
                if 'fuse_convs_dict' in k: new_k = k.replace('fuse_convs_dict', 'cft')
                converted_state_dict[new_k] = v
            state_dict = converted_state_dict
        
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        logger.debug(f"KEEP model '{model_type_str}' loaded onto {self.offload_device}.")

        for fname, (url, sha) in FACELIB_MODEL_URLS.items():
            load_file_from_url_comfy(url=url, model_dir_name=FACELIB_DEST_DIR, file_name=fname, sha256=sha)
        
        facelib_actual_paths = os.path.join(folder_paths.models_dir, FACELIB_DEST_DIR)
        os.makedirs(facelib_actual_paths, exist_ok=True)
        
        try:
            face_helper = FaceRestoreHelper(
                upscale_factor=1, face_size=512, crop_ratio=(1, 1),
                det_model=detection_model_str, save_ext='png', use_parse=True, device=self.offload_device,
                model_rootpath=facelib_actual_paths
            )
        except Exception as e:
            logger.error(f"Error initializing FaceRestoreHelper: {e}")
            raise

        model_pack = KEEPModelPack(net, face_helper, bg_upscale_model, face_upscale_model, model_type_str)

        base_pack_for_cache = KEEPModelPack(net, face_helper, None, None, model_type_str)
        self.loaded_models[cache_key] = base_pack_for_cache
        
        return model_pack