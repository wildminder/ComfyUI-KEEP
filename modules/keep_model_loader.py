import torch
import os
from comfy import model_management
import folder_paths # Import folder_paths for model directory management

from .utils import (
    load_file_from_url_comfy, ARCH_REGISTRY,
    KEEP_MODEL_CONFIGS, REALESRGAN_MODEL_CONFIG, FACELIB_MODEL_URLS, FACELIB_DEST_DIR
)

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer
    import basicsr.archs # Ensure KEEP arch is registered
    # print("basicsr components imported for KEEPModelLoader.")
except ImportError as e:
    print(f"Critical Import Error in keep_model_loader.py (basicsr): {e}")
    class RRDBNet: pass
    class RealESRGANer: pass

try:
    # This now refers to the MODIFIED FaceRestoreHelper in deps/facelib/
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    # print("FaceRestoreHelper imported (hopefully modified version).")
except ImportError as e:
    print(f"Critical Import Error in keep_model_loader.py (facelib): {e}")
    class FaceRestoreHelper: pass


class KEEPModelPack:
    def __init__(self, keep_net, face_helper, bg_upsampler, face_upsampler, device, model_type_str):
        self.keep_net = keep_net
        self.face_helper = face_helper
        self.bg_upsampler = bg_upsampler
        self.face_upsampler = face_upsampler
        self.device = device
        self.model_type_str = model_type_str

class KEEPModelLoader:
    def __init__(self):
        self.device = model_management.get_torch_device()
        self.loaded_models = {}

    def _set_realesrgan(self, tile_size=400):
        cache_key = f"realesrgan_x2_tile{tile_size}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        try:
            use_half = False
            if torch.cuda.is_available():
                no_half_gpu_list = ['1650', '1660']
                if not any(gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list):
                    use_half = True
            
            model_path = load_file_from_url_comfy(
                url=REALESRGAN_MODEL_CONFIG['url'],
                model_dir_name=REALESRGAN_MODEL_CONFIG['dest_dir'],
                file_name=os.path.basename(REALESRGAN_MODEL_CONFIG['url']),
                sha256=REALESRGAN_MODEL_CONFIG.get('sha256')
            )
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            
            upsampler = RealESRGANer(
                scale=2, model_path=model_path, model=model, tile=tile_size,
                tile_pad=40, pre_pad=0, half=use_half, device=self.device
            )
            self.loaded_models[cache_key] = upsampler
            print(f"RealESRGAN model loaded for {cache_key}.")
            return upsampler
        except Exception as e:
            print(f"Error setting up RealESRGAN: {e}")
            return None


    def load_keep_model_pack(self, model_type_str, detection_model_str,
                             use_bg_upsampler, use_face_upsampler, bg_tile_size):
        
        cache_key = (model_type_str, detection_model_str, use_bg_upsampler, use_face_upsampler, bg_tile_size)
        if cache_key in self.loaded_models:
            print(f"Returning cached KEEPModelPack for {model_type_str}")
            return self.loaded_models[cache_key]

        if model_type_str not in KEEP_MODEL_CONFIGS:
            raise ValueError(f"Unknown KEEP model type: {model_type_str}")

        config_entry = KEEP_MODEL_CONFIGS[model_type_str]
        
        keep_model_arch_class = ARCH_REGISTRY.get('KEEP')
        if keep_model_arch_class is None:
            raise RuntimeError("KEEP architecture class not found in ARCH_REGISTRY.")
        
        net = keep_model_arch_class(**config_entry['architecture']).to(self.device)
        
        ckpt_path = load_file_from_url_comfy(
            url=config_entry['url'],
            model_dir_name=config_entry['dest_dir'],
            file_name=os.path.basename(config_entry['url']),
            sha256=config_entry.get('sha256')
        )
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True) # Assuming state_dict only
        
        state_dict_key = 'params_ema' if 'params_ema' in checkpoint else 'params'
        state_dict = checkpoint.get(state_dict_key, checkpoint) # Fallback to raw checkpoint if keys not present

        converted_state_dict = {}
        needs_conversion = any('cross_fuse' in k or 'fuse_convs_dict' in k for k in state_dict.keys())
        if needs_conversion:
            print("Applying key conversion for KEEP model weights...")
            for k, v in state_dict.items():
                new_k = k
                if 'cross_fuse' in k: new_k = k.replace('cross_fuse', 'cfa')
                if 'fuse_convs_dict' in k: new_k = k.replace('fuse_convs_dict', 'cft')
                converted_state_dict[new_k] = v
            state_dict = converted_state_dict
        
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        print(f"KEEP model '{model_type_str}' loaded onto {self.device}.")


        # Download facelib models to ComfyUI's standard directory for face detection models
        for fname, (url, sha) in FACELIB_MODEL_URLS.items():
            load_file_from_url_comfy(url=url, model_dir_name=FACELIB_DEST_DIR, file_name=fname, sha256=sha)
        
        # Get the actual disk path for FACELIB_DEST_DIR category
        #facelib_actual_paths = folder_paths.get_folder_paths(FACELIB_DEST_DIR)
        facelib_actual_paths = os.path.join(folder_paths.models_dir, FACELIB_DEST_DIR)
        if not facelib_actual_paths:
            face_helper_model_root = os.path.join(folder_paths.models_dir, FACELIB_DEST_DIR)
            print(f"Warning: Facelib model dir category '{FACELIB_DEST_DIR}' not in folder_paths. Using default: {face_helper_model_root}")
        else:
            face_helper_model_root = facelib_actual_paths
            os.makedirs(face_helper_model_root, exist_ok=True)
        

        try:
            face_helper = FaceRestoreHelper(
                upscale_factor=1, face_size=512, crop_ratio=(1, 1),
                det_model=detection_model_str, save_ext='png', use_parse=True, device=self.device,
                model_rootpath=face_helper_model_root # Pass the ComfyUI managed path
            )
            # print(f"FaceRestoreHelper initialized with detection model: {detection_model_str} and model_rootpath: {face_helper_model_root}")
        except Exception as e:
            print(f"Error initializing FaceRestoreHelper: {e}")
            # print("Ensure that the vendored FaceRestoreHelper and its dependent model loaders "
            #       "(e.g., init_detection_model, init_parsing_model in facelib) "
            #       "are correctly modified to use the 'model_rootpath' argument or can find models "
            #       f"in '{face_helper_model_root}'.")
            raise

        bg_upsampler = self._set_realesrgan(tile_size=bg_tile_size) if use_bg_upsampler else None
        face_upsampler = self._set_realesrgan(tile_size=bg_tile_size) if use_face_upsampler else None
        
        model_pack = KEEPModelPack(net, face_helper, bg_upsampler, face_upsampler, self.device, model_type_str)
        self.loaded_models[cache_key] = model_pack
        return model_pack