"""The model configurations and the weight manifest of the KEEP pack.

The config half of the old ``modules/utils.py``: the architecture configs of
the two KEEP checkpoints.  The image half lives in ``src/keep_image_utils.py``,
the paths in ``src/keep_paths.py`` and the downloads in ``src/keep_download.py``.

Since **P5.1** the weights are described by the :data:`WEIGHT_MANIFEST`: every
file carries its URL, its sha256 and its on-disk destination.  The manifest is
the **single source of truth** - the model configs and the facelib list are
derived from it, so a URL or a hash is never written down twice.

The sha256 values come from the one time verified download of **P5.0**; the
provenance is recorded in
``.dev/docs/audit/2026-10-22-weight-manifest-sources.md`` (the machine-readable
record is ``tools/weight_hashes.json``).
"""

from __future__ import annotations

from dataclasses import dataclass

# directly import from the vendored basicsr
#
# P2.7: the swallow-and-stub fallback is gone.  A broken vendored dependency
# must fail loudly - the dummy registry hid the missing KEEP architecture and
# the node kept loading without it.
#
# P4.7: the unused ``load_file_from_url`` import is gone - the downloads run
# through ``src/keep_download.py`` (the manifest-driven downloader of P5.2).
try:  # the package import (the node package is the parent package)
    from ..modules.deps.basicsr.utils.registry import ARCH_REGISTRY
except ImportError:  # the flat import (the repository root on sys.path)
    from modules.deps.basicsr.utils.registry import ARCH_REGISTRY


@dataclass(frozen=True)
class WeightEntry:
    """One weight of the manifest.

    ``sha256`` is ``None`` for a file that cannot be verified (the Asian
    checkpoint is not published upstream - P0.4 section 4.1).  Such an entry
    carries ``unverified=True`` and ``available=False``; the downloader skips
    the verification for it and warns loudly.
    """

    name: str
    url: str
    sha256: str | None
    dest_folder: str
    subdir: str
    required: bool
    available: bool = True
    unverified: bool = False
    sha256_prefix: str | None = None
    size: int | None = None

    @property
    def relative_path(self) -> str:
        """The path of the file inside its model folder."""
        return f"{self.subdir}/{self.name}" if self.subdir else self.name


#: The upstream release the weights come from.
KEEP_RELEASE = "v1.0.0"
KEEP_RELEASE_BASE = (
    f"https://github.com/jnjaby/KEEP/releases/download/{KEEP_RELEASE}"
)

#: The eight weights of the pack.
#:
#: The destinations reproduce the **legacy** on-disk layout exactly:
#: ``models/keep_models/KEEP`` (the two KEEP checkpoints), ``models/facedetection``
#: (the facelib detection / parsing weights) and ``models/gmflow`` (the
#: defensive GMFlow checkpoint).
WEIGHT_MANIFEST: tuple[WeightEntry, ...] = (
    WeightEntry(
        name="KEEP-b76feb75.pth",
        url=f"{KEEP_RELEASE_BASE}/KEEP-b76feb75.pth",
        sha256="b76feb75f3ed782f4e91eda85c03018241827bba8693532dc10351bab5052c1a",
        dest_folder="keep_models",
        subdir="KEEP",
        required=True,
        size=634093768,
    ),
    # Not published upstream (HTTP 404 - P0.2 section 3 / P0.4 section 4.1):
    # no hash is obtainable and the entry must not be required (it shares the
    # destination with the default KEEP checkpoint).
    WeightEntry(
        name="KEEP_Asian-4765ebe0.pth",
        url=f"{KEEP_RELEASE_BASE}/KEEP_Asian-4765ebe0.pth",
        sha256=None,
        dest_folder="keep_models",
        subdir="KEEP",
        required=False,
        available=False,
        unverified=True,
        sha256_prefix="4765ebe0",
    ),
    WeightEntry(
        name="detection_Resnet50_Final.pth",
        url=f"{KEEP_RELEASE_BASE}/detection_Resnet50_Final.pth",
        sha256="6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d",
        dest_folder="facedetection",
        subdir="",
        required=True,
        size=109497761,
    ),
    WeightEntry(
        name="detection_mobilenet0.25_Final.pth",
        url=f"{KEEP_RELEASE_BASE}/detection_mobilenet0.25_Final.pth",
        sha256="2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220",
        dest_folder="facedetection",
        subdir="",
        required=True,
        size=1789735,
    ),
    WeightEntry(
        name="yolov5n-face.pth",
        url=f"{KEEP_RELEASE_BASE}/yolov5n-face.pth",
        sha256="d2bbfbe9f36cf1ec345dc69658d7209e5448a676d946f1bf7818ac50d4489357",
        dest_folder="facedetection",
        subdir="",
        required=True,
        size=7145625,
    ),
    WeightEntry(
        name="yolov5l-face.pth",
        url=f"{KEEP_RELEASE_BASE}/yolov5l-face.pth",
        sha256="1ba9d2125fda4d823df5152b9fc2903c59aa76c0d3771e02bcf13a56a282cf96",
        dest_folder="facedetection",
        subdir="",
        required=True,
        size=186973013,
    ),
    WeightEntry(
        name="parsing_parsenet.pth",
        url=f"{KEEP_RELEASE_BASE}/parsing_parsenet.pth",
        sha256="3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2",
        dest_folder="facedetection",
        subdir="",
        required=True,
        size=85331193,
    ),
    # The GMFlow checkpoint is **defensive** (P0.4 section 2): the shipped
    # KEEP checkpoint carries the trained flownet weights.  It is only
    # resolved when the loaded checkpoint does not provide them (P5.5).
    WeightEntry(
        name="gmflow_sintel-0c07dcb3.pth",
        url=f"{KEEP_RELEASE_BASE}/gmflow_sintel-0c07dcb3.pth",
        sha256="0c07dcb35770464f38a5ff4de18c04177b242dc5de8cd2068adf46f3d4fe193a",
        dest_folder="gmflow",
        subdir="",
        required=False,
        size=18768907,
    ),
)

#: The manifest entries by name.
WEIGHT_MANIFEST_BY_NAME: dict[str, WeightEntry] = {
    entry.name: entry for entry in WEIGHT_MANIFEST
}

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
        # the manifest entry of the checkpoint (the weight table is the
        # manifest - P5.1)
        'weight': 'KEEP-b76feb75.pth',
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
        'weight': 'KEEP_Asian-4765ebe0.pth',
    }
}

default_arch_params = {
    'gumbel_straight_through': False, 'gumbel_kl_weight': 1e-8,
    'vqgan_path': None, # Loaded separately if needed by arch, but KEEP loads its own VQGAN weights
    'latent_size': 256, # Default from KEEP arch
    'fix_modules': ['quantize', 'generator'], # Common setting, can be adjusted
    'flownet_path': None, # P5.5: the loader resolves it before the arch is built
    'cfa_nlayers': 4, # Default from KEEP arch
    'cross_residual': True, # Default from KEEP arch
    'mask_ratio': 0., # Default from KEEP arch
}

for model_key in KEEP_MODEL_CONFIGS:
    for param_key, default_val in default_arch_params.items():
        if param_key not in KEEP_MODEL_CONFIGS[model_key]['architecture']:
            KEEP_MODEL_CONFIGS[model_key]['architecture'][param_key] = default_val

# The weight fields of the model configs are **derived** from the manifest -
# a URL or a hash is never written down twice.
for _model_key, _config in KEEP_MODEL_CONFIGS.items():
    _entry = WEIGHT_MANIFEST_BY_NAME[_config['weight']]
    _config['url'] = _entry.url
    _config['dest_folder'] = _entry.dest_folder
    _config['subdir'] = _entry.subdir
    _config['sha256'] = _entry.sha256


def weight_entry_for(model_type: str) -> WeightEntry:
    """The manifest entry of a KEEP model type (``KEEP`` or ``Asian``)."""
    if model_type not in KEEP_MODEL_CONFIGS:
        raise ValueError(f"Unknown KEEP model type: {model_type}")
    return WEIGHT_MANIFEST_BY_NAME[KEEP_MODEL_CONFIGS[model_type]['weight']]


#: The facelib weights (the destination of the facelib helper).
FACELIB_DEST_FOLDER = 'facedetection'

#: The facelib weights - derived from the manifest.
FACELIB_MODEL_URLS = {
    entry.name: (entry.url, entry.sha256)
    for entry in WEIGHT_MANIFEST
    if entry.dest_folder == FACELIB_DEST_FOLDER
}
