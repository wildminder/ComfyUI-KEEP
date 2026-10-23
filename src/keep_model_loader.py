"""The model loader of the KEEP pack (P3.2).

The engine that builds a :class:`KEEPModelPack`: the KEEP network, the
facelib face helper and the optional ComfyUI upscalers.  It moved here from
``modules/keep_model_loader.py`` in P3.2.

The ``comfy`` imports are **lazy** (function level): the plan's §3 import
rule wants the pack to import without a ComfyUI server.  The vendored
facelib helper imports ``comfy`` as well, so it is imported lazily too.
"""

import torch
import os

from . import keep_paths
from .keep_logger import logger
from .keep_model_configs import (
    ARCH_REGISTRY, KEEP_MODEL_CONFIGS, FACELIB_MODEL_URLS, FACELIB_DEST_FOLDER
)
from .keep_paths import load_file_from_url_comfy


# P2.7: the swallow-and-stub fallback is gone - a broken vendored dependency
# must fail loudly instead of silently producing an unusable model pack.
def _face_restore_helper():
    """The vendored ``FaceRestoreHelper`` - imported **lazily**.

    The vendored facelib imports ``comfy`` at module level, so the import
    must not run while ``src.keep_model_loader`` is imported.
    """
    try:  # the package import (the node package is the parent package)
        from ..modules.deps.facelib.utils.face_restoration_helper import FaceRestoreHelper
    except ImportError:  # the flat import (the repository root on sys.path)
        from modules.deps.facelib.utils.face_restoration_helper import FaceRestoreHelper
    return FaceRestoreHelper


def __getattr__(name: str):
    """The lazy module attributes (PEP 562).

    ``FaceRestoreHelper`` stays reachable as an attribute of the module (the
    P2.5 / P2.7 tests assert that the vendored class is the one that is
    used), but the import that pulls in ``comfy`` only happens when the
    attribute is really read.
    """
    if name == "FaceRestoreHelper":
        return _face_restore_helper()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class KEEPModelPack:
    def __init__(self, keep_net, face_helper, bg_upscale_model, face_upscale_model, model_type_str):
        from comfy import model_management  # lazy: the pack imports without ComfyUI

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
        from comfy import model_management  # lazy: the pack imports without ComfyUI

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
        from comfy import model_management  # lazy: the pack imports without ComfyUI

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
            folder_name=config_entry['dest_folder'],
            subdir=config_entry['subdir'],
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
            load_file_from_url_comfy(url=url, folder_name=FACELIB_DEST_FOLDER, file_name=fname, sha256=sha)

        # P3.5: the facelib root comes from the registered model folders -
        # the raw ``folder_paths.models_dir`` concatenation is gone.  It is
        # the root the facelib weights really live in (an extra model root
        # included), because facelib builds the weight paths from it.
        facelib_actual_paths = keep_paths.resolve_model_root(
            FACELIB_DEST_FOLDER, FACELIB_MODEL_URLS.keys()
        )

        FaceRestoreHelper = _face_restore_helper()
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