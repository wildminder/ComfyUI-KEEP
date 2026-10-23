"""The model configurations of the KEEP pack (P3.2).

The config half of the old ``modules/utils.py``: the architecture configs of
the two KEEP checkpoints and the facelib weight list.  The image half moved
to ``src/keep_image_utils.py``, the download helper to ``src/keep_paths.py``.

P5.1 turns the config table into the ``WEIGHT_MANIFEST`` (the ``dest_folder``
/ ``subdir`` fields are already in place since P3.5) and P5.2 moves the
download helper into ``src/keep_download.py``.
"""

from .keep_logger import logger

# directly import from the vendored basicsr
#
# P2.7: the swallow-and-stub fallback is gone.  A broken vendored dependency
# must fail loudly - the dummy registry hid the missing KEEP architecture and
# the node kept loading without it.
try:  # the package import (the node package is the parent package)
    from ..modules.deps.basicsr.utils.registry import ARCH_REGISTRY
    from ..modules.deps.basicsr.utils.download_util import load_file_from_url
except ImportError:  # the flat import (the repository root on sys.path)
    from modules.deps.basicsr.utils.registry import ARCH_REGISTRY
    from modules.deps.basicsr.utils.download_util import load_file_from_url

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
        'dest_folder': 'keep_models', 'subdir': 'KEEP'
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
        'dest_folder': 'keep_models', 'subdir': 'KEEP'
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
FACELIB_DEST_FOLDER = 'facedetection'
