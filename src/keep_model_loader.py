"""The model loader of the KEEP pack (P3.2).

The engine that builds a :class:`KEEPModelPack`: the KEEP network, the
facelib face helper and the optional ComfyUI upscalers.  It moved here from
``modules/keep_model_loader.py`` in P3.2.

Since **P5.3** the weights are resolved through the manifest
(:mod:`src.keep_model_configs`) and the downloader (:mod:`src.keep_download`):
every **required** weight must be on disk before the model is built - a
missing one is downloaded and, when that fails, the error names the file, the
expected path and the manifest URL.  The optional weights (the GMFlow
checkpoint) are deferred to :meth:`KEEPModelPack.ensure_flow`.

Since **P5.5** the optical flow is wired: the GMFlow checkpoint is only
resolved when the loaded KEEP checkpoint does **not** carry the ``flownet.*``
weights (the shipped checkpoint does - P0.4 section 2).  The resolved path is
put into the architecture config **before** the net is built, so the arch
loads the trained weights instead of random ones.

The ``comfy`` imports are **lazy** (function level): the plan's §3 import
rule wants the pack to import without a ComfyUI server.  The vendored
facelib helper imports ``comfy`` as well, so it is imported lazily too.
"""

import torch

from . import keep_download
from . import keep_paths
from .keep_logger import logger
from .keep_model_configs import (
    ARCH_REGISTRY,
    FACELIB_DEST_FOLDER,
    KEEP_MODEL_CONFIGS,
    WEIGHT_MANIFEST,
    WEIGHT_MANIFEST_BY_NAME,
    weight_entry_for,
)

#: The name of the GMFlow manifest entry (the defensive flow weights).
GMFLOW_WEIGHT_NAME = "gmflow_sintel-0c07dcb3.pth"


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


def _build_patchers(keep_net, face_helper, device, offload_device) -> dict:
    """The three patchers of a pack (P4.5).

    * ``keep`` - the KEEP net; it **owns the flownet** as a submodule
      (``keep_arch.py:882``), so the flownet must **not** get a patcher of
      its own: it would be counted twice in the VRAM accounting.
    * ``detector`` - the face detector of the facelib helper.
    * ``parser`` - the face parser of the facelib helper.

    The patchers are built **once** per model build and are shared by every
    pack that uses these models (the loader caches them).
    """
    from comfy.model_patcher import ModelPatcher  # lazy: the pack imports without ComfyUI

    patchers: dict = {}
    if keep_net is not None:
        patchers["keep"] = ModelPatcher(
            keep_net, load_device=device, offload_device=offload_device
        )
    detector = getattr(face_helper, "face_detector", None)
    if detector is not None:
        patchers["detector"] = ModelPatcher(
            detector, load_device=device, offload_device=offload_device
        )
    parser = getattr(face_helper, "face_parse", None)
    if parser is not None:
        patchers["parser"] = ModelPatcher(
            parser, load_device=device, offload_device=offload_device
        )
    return patchers


def _load_flownet_weights(flownet, path: str) -> None:
    """Load the trained GMFlow weights into a flownet (``strict=True``).

    The vendored ``FlowGenerator`` loads the checkpoint itself when it is
    built with a path (``modules/deps/basicsr/archs/gmflow_arch.py:25-28``);
    this helper does the same for a net that was built **without** a path -
    the weights are then loaded with ``strict=True``, so a mismatching file
    fails loudly instead of leaving random weights behind.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model", checkpoint)
    target = getattr(flownet, "model", flownet)
    target.load_state_dict(weights, strict=True)


class KEEPModelPack:
    """The models of the pack and their residency (P4.5).

    The models are wrapped in three ``ModelPatcher`` instances (the KEEP net
    with its flownet, the face detector, the face parser) and ComfyUI
    arbitrates the VRAM: :meth:`load_device` asks
    ``model_management.load_models_gpu`` for the patchers, :meth:`offload`
    gives the memory back.
    """

    def __init__(self, keep_net, face_helper, bg_upscale_model, face_upscale_model,
                 model_type_str, patchers=None, flownet_from_checkpoint=False,
                 flownet_path=None):
        from comfy import model_management  # lazy: the pack imports without ComfyUI

        self.keep_net = keep_net
        self.face_helper = face_helper
        self.bg_upscale_model = bg_upscale_model
        self.face_upscale_model = face_upscale_model
        self.model_type_str = model_type_str
        self.device = model_management.get_torch_device()
        self.offload_device = model_management.unet_offload_device()

        # the flownet is a submodule of the KEEP net - it belongs to the
        # patcher of the net (no second patcher, no double accounting)
        self.flownet = getattr(keep_net, "flownet", None)

        # P5.5: where the flow weights come from.  A KEEP checkpoint that
        # carries the ``flownet.*`` tensors restores them itself; otherwise
        # the GMFlow file is resolved (``flownet_path``).
        self.flownet_from_checkpoint = flownet_from_checkpoint
        self.flownet_path = flownet_path

        if patchers is None:
            patchers = _build_patchers(
                keep_net, face_helper, self.device, self.offload_device
            )
        self.patchers_by_role = dict(patchers)
        self.keep_patcher = self.patchers_by_role.get("keep")
        self.detector_patcher = self.patchers_by_role.get("detector")
        self.parser_patcher = self.patchers_by_role.get("parser")
        self.patchers = list(self.patchers_by_role.values())

        # P4.5 / P5.5: the state of the flow resolution (see ensure_flow)
        self._flow_checked = False
        self._flow_available = False

    def load_device(self, memory_required: int = 0):
        """Move the models to the main compute device.

        ComfyUI arbitrates the VRAM: the patchers are handed to
        ``model_management.load_models_gpu``, which decides what fits.
        """
        from comfy import model_management  # lazy: the pack imports without ComfyUI

        if self.patchers:
            model_management.load_models_gpu(
                self.patchers, memory_required=memory_required
            )
        elif self.keep_net is not None:
            # no patcher (a pack without models) - move the net directly
            self.keep_net.to(self.device)

        # The optional ComfyUI upscalers are not patchers - they are moved
        # directly (they are small and they are not part of the KEEP models).
        if self.bg_upscale_model is not None:
            self.bg_upscale_model.model.to(self.device)
        if self.face_upscale_model is not None:
            self.face_upscale_model.model.to(self.device)

        if self.face_helper is not None:
            self.face_helper.device = self.device

    def offload(self):
        """Move the models back to the offload device and free the VRAM."""
        from comfy import model_management  # lazy: the pack imports without ComfyUI

        for patcher in self.patchers:
            self._unload_patcher(patcher)

        if self.bg_upscale_model is not None:
            self.bg_upscale_model.model.to(self.offload_device)
        if self.face_upscale_model is not None:
            self.face_upscale_model.model.to(self.offload_device)

        if self.face_helper is not None:
            self.face_helper.device = self.offload_device

        model_management.unload_all_models()
        model_management.soft_empty_cache()

    def _unload_patcher(self, patcher):
        """Move the model of a patcher back to its offload device.

        The real ``ModelPatcher`` has **no** ``unload()`` method (ComfyUI
        v0.42.0) - the model is moved to the offload device of the patcher
        and ComfyUI does the rest (``unload_all_models``).
        """
        model = getattr(patcher, "model", None)
        if model is None:
            return
        offload_device = getattr(patcher, "offload_device", None) or self.offload_device
        try:
            model.to(offload_device)
        except Exception as exc:  # noqa: BLE001 - the offload must not raise
            logger.debug(f"KEEP: a model could not be offloaded ({exc})")

    def ensure_flow(self) -> bool:
        """Make sure the optical flow net of the KEEP net has **trained** weights.

        The resolution is **conditional** (P5.5) and **idempotent** - it
        happens at most once per pack:

        * a KEEP checkpoint that carries the ``flownet.*`` tensors restores
          them with ``strict=True`` - there is nothing to resolve and the
          method is a no-op;
        * a checkpoint **without** them needs the GMFlow file: it is resolved
          (downloaded when it is missing) and loaded into the flownet with
          ``strict=True``.

        The image path does not call this hook (the sequence path does); the
        shipped checkpoint carries the flownet anyway, so the image path has
        trained flow weights as well (P0.4 section 1).
        """
        if self._flow_checked:
            return self._flow_available
        self._flow_checked = True

        if self.flownet_from_checkpoint:
            # the checkpoint restored the flownet - nothing to resolve
            self._flow_available = self.flownet is not None
            return self._flow_available

        if self.flownet is None:
            self._flow_available = False
            return False

        # the net was built without the GMFlow weights - resolve them now
        entry = WEIGHT_MANIFEST_BY_NAME[GMFLOW_WEIGHT_NAME]
        path = self.flownet_path or keep_download.ensure_optional(entry)
        if path is None:
            logger.warning(
                f"KEEP: the optical flow weights ({entry.name}) could not be "
                "resolved - the flow net keeps its initial weights. Download "
                f"the file to {keep_download.target_path(entry)} and run again."
            )
            self._flow_available = False
            return False

        if not self.flownet_path:
            # the net was built without them - load the trained weights now
            _load_flownet_weights(self.flownet, path)
        self.flownet_path = path
        self._flow_available = True
        return True


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
                model_type_str,
                # P4.5: the patchers belong to the models - they are built
                # once and shared by every pack that uses these models.
                patchers=cached_pack.patchers_by_role,
                # P5.5: the flow resolution belongs to the models as well
                flownet_from_checkpoint=cached_pack.flownet_from_checkpoint,
                flownet_path=cached_pack.flownet_path,
            )
            return model_pack

        if model_type_str not in KEEP_MODEL_CONFIGS:
            raise ValueError(f"Unknown KEEP model type: {model_type_str}")

        config_entry = KEEP_MODEL_CONFIGS[model_type_str]

        # P5.3: fail fast.  Every weight the pack needs is resolved (and
        # downloaded when it is missing) **before** the model is built.  A
        # missing required weight raises a FileNotFoundError that names the
        # file, the expected path and the manifest URL.
        weight_entry = weight_entry_for(model_type_str)
        ckpt_path = self._resolve_checkpoint(weight_entry)

        facelib_entries = [
            entry for entry in WEIGHT_MANIFEST
            if entry.dest_folder == FACELIB_DEST_FOLDER
        ]
        for entry in facelib_entries:
            keep_download.ensure_required(entry)

        keep_model_arch_class = ARCH_REGISTRY.get('KEEP')
        if keep_model_arch_class is None:
            raise RuntimeError("KEEP architecture class not found in ARCH_REGISTRY.")

        checkpoint = torch.load(ckpt_path, map_location=self.offload_device, weights_only=True)

        state_dict_key = 'params_ema' if 'params_ema' in checkpoint else 'params'
        state_dict = checkpoint.get(state_dict_key, checkpoint)

        # P5.5: does the checkpoint carry the trained flownet weights?  The
        # shipped KEEP checkpoint does (123 ``flownet.*`` tensors - P0.4
        # section 2); the GMFlow file is only resolved when it does not.
        flownet_from_checkpoint = any(
            key.startswith('flownet.') for key in state_dict.keys()
        )

        # the architecture config of this build - a **copy**, so the global
        # config table is never mutated
        architecture = dict(config_entry['architecture'])
        if flownet_from_checkpoint:
            architecture['flownet_path'] = None
            flownet_path = None
        else:
            flownet_path = keep_download.ensure_optional(
                WEIGHT_MANIFEST_BY_NAME[GMFLOW_WEIGHT_NAME]
            )
            architecture['flownet_path'] = flownet_path
            if flownet_path is None:
                logger.warning(
                    "KEEP: the checkpoint does not carry the flownet weights "
                    "and the GMFlow checkpoint could not be resolved - the flow "
                    "net keeps its initial weights."
                )

        net = keep_model_arch_class(**architecture).to(self.offload_device)

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

        # P3.5: the facelib root comes from the registered model folders -
        # the raw ``folder_paths.models_dir`` concatenation is gone.  It is
        # the root the facelib weights really live in (an extra model root
        # included), because facelib builds the weight paths from it.
        facelib_actual_paths = keep_paths.resolve_model_root(
            FACELIB_DEST_FOLDER, [entry.name for entry in facelib_entries]
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

        # P4.5: the patchers are built once (with the base pack) and shared
        # with the returned pack - a second set of patchers for the same
        # models would confuse the VRAM accounting of ComfyUI.
        base_pack_for_cache = KEEPModelPack(
            net, face_helper, None, None, model_type_str,
            flownet_from_checkpoint=flownet_from_checkpoint,
            flownet_path=flownet_path,
        )
        self.loaded_models[cache_key] = base_pack_for_cache

        model_pack = KEEPModelPack(
            net, face_helper, bg_upscale_model, face_upscale_model, model_type_str,
            patchers=base_pack_for_cache.patchers_by_role,
            flownet_from_checkpoint=flownet_from_checkpoint,
            flownet_path=flownet_path,
        )

        return model_pack

    def _resolve_checkpoint(self, weight_entry) -> str:
        """The path of the KEEP checkpoint (downloaded when it is missing).

        A checkpoint that is not published upstream (the Asian one) raises an
        actionable ``FileNotFoundError``: the file, the expected path and the
        dead URL are named, and the user is told to place the file manually.
        """
        path = keep_download.ensure_weight(weight_entry)
        if path is None:
            raise FileNotFoundError(
                f"KEEP: the model needs the weight {weight_entry.name!r}, which "
                "is not published upstream (HTTP 404).\n"
                f"  url:  {weight_entry.url}\n"
                f"  dest: {keep_download.target_path(weight_entry)}\n"
                "Place the file manually at the destination and try again."
            )
        return path