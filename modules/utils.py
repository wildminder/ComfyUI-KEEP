import os
import torch
import numpy as np
import cv2
from PIL import Image
import folder_paths
from comfy.utils import ProgressBar
from urllib.request import urlopen, Request
import hashlib
import sys
from torch.hub import download_url_to_file, get_dir

from .. import logger

# Path to ComfyUI-KEEP/modules/
module_path = os.path.dirname(os.path.abspath(__file__))
deps_path = os.path.join(module_path, 'deps')
if deps_path not in sys.path:
    sys.path.insert(0, deps_path)

# directly import from the vendored basicsr
try:
    from basicsr.utils.registry import ARCH_REGISTRY
    from basicsr.utils.download_util import load_file_from_url
    # print("Successfully imported ARCH_REGISTRY and load_file_from_url from vendored basicsr.")
except ImportError as e:
    print(f"Error importing from vendored basicsr in utils.py: {e}")
    # Fallback ARCH_REGISTRY if needed, though ideally the import should work
    class DummyArchRegistry:
        def __init__(self): self._registry = {}
        def register(self, obj): self._registry[obj.__name__] = obj; return obj
        def get(self, name): return self._registry.get(name)
    ARCH_REGISTRY = DummyArchRegistry()
    # Fallback load_file_from_url if needed
    def load_file_from_url(url, model_dir, progress, file_name): # Simplified
        target_path = os.path.join(model_dir, file_name if file_name else os.path.basename(url))
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Fallback load_file_from_url: {target_path} not found. Please download manually.")
        return target_path

KEEP_MODEL_CONFIGS = {
    'KEEP': {
        'architecture': {
            'img_size': 512, 'emb_dim': 256, 'dim_embd': 512, 'n_head': 8, 'n_layers': 9,
            'codebook_size': 1024, 'cft_list': ['16', '32', '64'], 'kalman_attn_head_dim': 48,
            'num_uncertainty_layers': 3, 'cfa_list': ['16', '32'], 'cfa_nhead': 4, 'cfa_dim': 256, 'cond': 1,
            'nf': 64, # from test_KEEP.yml (network_g) / stage1_VQGAN.yml (network_g)
            'ch_mult': [1, 2, 2, 4, 4, 8], # from stage1_VQGAN.yml (network_g)
            'attn_resolutions': [16], # from stage1_VQGAN.yml (network_g)
            'res_blocks': 2, # from stage1_VQGAN.yml (network_g) -> defaults in VQAutoEncoder
            'quantizer_type': "nearest", # from stage1_VQGAN.yml (network_g -> quantizer) -> used by KEEP
            'beta': 0.25, # VQAutoEncoder default
            'temp_reg_list': ['32'], # From stage3_KEEP.yml (network_g)
        },
        'url': 'https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP-b76feb75.pth',
        'dest_dir': 'keep_models/KEEP'
    },
    'Asian': {
        'architecture': {
            'img_size': 512, 'emb_dim': 256, 'dim_embd': 512, 'n_head': 8, 'n_layers': 9,
            'codebook_size': 1024, 'cft_list': ['32', '64', '128', '256'], 'kalman_attn_head_dim': 48,
            'num_uncertainty_layers': 3, 'cfa_list': ['16', '32'], 'cfa_nhead': 4, 'cfa_dim': 256, 'cond': 1,
            'nf': 64,
            'ch_mult': [1, 2, 2, 4, 4, 8], # Check if Asian model has different VQGAN base
            'attn_resolutions': [16],
            'res_blocks': 2,
            'quantizer_type': "nearest",
            'beta': 0.25,
            'temp_reg_list': [], # Check if Asian model uses temp_reg_list
        },
        'url': 'https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP_Asian-4765ebe0.pth',
        'dest_dir': 'keep_models/KEEP'
    }
}

default_arch_params = {
    'gumbel_straight_through': False, 'gumbel_kl_weight': 1e-8,
    'vqgan_path': None, # Loaded separately if needed by arch, but KEEP loads its own VQGAN weights
    'latent_size': 256, # Default from KEEP arch
    'fix_modules': ['quantize', 'generator'], # Common setting, can be adjusted
    'flownet_path': None, # Will be loaded by ComfyUI node if needed
    'cfa_nlayers': 4, # Default from KEEP arch
    'cross_residual': True, # Default from KEEP arch
    'mask_ratio': 0. # Default from KEEP arch
}

for model_key in KEEP_MODEL_CONFIGS:
    for param_key, default_val in default_arch_params.items():
        if param_key not in KEEP_MODEL_CONFIGS[model_key]['architecture']:
            KEEP_MODEL_CONFIGS[model_key]['architecture'][param_key] = default_val

FACELIB_MODEL_URLS = {
    'detection_Resnet50_Final.pth': ('https://github.com/jnjaby/KEEP/releases/download/v1.0.0/detection_Resnet50_Final.pth', None),
    'detection_mobilenet0.25_Final.pth': ('https://github.com/jnjaby/KEEP/releases/download/v1.0.0/detection_mobilenet0.25_Final.pth', None),
    'yolov5n-face.pth': ('https://github.com/jnjaby/KEEP/releases/download/v1.0.0/yolov5n-face.pth', None),
    'yolov5l-face.pth': ('https://github.com/jnjaby/KEEP/releases/download/v1.0.0/yolov5l-face.pth', None),
    'parsing_parsenet.pth': ('https://github.com/jnjaby/KEEP/releases/download/v1.0.0/parsing_parsenet.pth', None),
}
FACELIB_DEST_DIR = 'facedetection'

def load_file_from_url_comfy(url, model_dir_name, progress=True, file_name=None, sha256=None):
    """
    `model_dir_name` is a category like 'upscale_models', 'facedetection', etc.
    """
    if file_name is None:
        file_name = os.path.basename(url)
        if '?' in file_name:
            file_name = file_name.split('?')[0]

    destination_dir_paths = [os.path.join(folder_paths.models_dir, model_dir_name)]
    
    if not destination_dir_paths:
        # If category not found, default to ComfyUI/models/<model_dir_name>
        destination_dir = os.path.join(folder_paths.models_dir, model_dir_name)
        # print(f"Warning: Model directory type '{model_dir_name}' not in folder_paths. Using default: {destination_dir}")
    else:
        # Use the first path
        destination_dir = destination_dir_paths[0]

    os.makedirs(destination_dir, exist_ok=True)
    cached_file = os.path.join(destination_dir, file_name)

    if os.path.exists(cached_file) and sha256:
        with open(cached_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash == sha256:
            logger.debug(f"File {file_name} already exists and hash matches. Skipping download.")
            return cached_file
        else:
            logger.warning(f"File {file_name} exists but hash mismatch. Redownloading.")
    elif os.path.exists(cached_file) and not sha256:
        logger.debug(f"File {file_name} already exists. Skipping download (no hash check).")
        return cached_file

    logger.info(f'Downloading: "{file_name}" from {url} to {cached_file}')
    
    try:
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    except Exception as e:
        logger.error(f"Error downloading {url} using torch.hub.download_url_to_file: {e}")
        if os.path.exists(cached_file):
            os.remove(cached_file)
        raise

    if sha256:
        with open(cached_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != sha256:
            if os.path.exists(cached_file):
                os.remove(cached_file)
            raise ValueError(f"Downloaded file {file_name} hash mismatch. Expected {sha256}, got {file_hash}.")
    
    return cached_file

def comfy_image_to_cv2(comfy_image: torch.Tensor) -> np.ndarray:
    if comfy_image.ndim == 3: comfy_image = comfy_image.unsqueeze(0)
    img_np = comfy_image.cpu().numpy()
    img_np = (img_np.squeeze(0) * 255).astype(np.uint8)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_cv2

def cv2_to_comfy_image(cv2_image: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype(np.float32) / 255.0
    comfy_image = torch.from_numpy(img_np).unsqueeze(0)
    return comfy_image