"""The conversions between the ComfyUI IMAGE tensors and the cv2 images.

The functions moved here from ``modules/utils.py`` in P3.2 - they are the
image half of the old utility module, the config half lives in
``src/keep_model_configs.py``.

The dependency slimming replaced the channel conversions of cv2 with torch
indexing and added the torch bilinear ``resize_bgr_u8`` for the INTER_LINEAR
resizes of the processor; the equivalences against the cv2 ops are pinned in
``tests/test_image_op_equivalence.py``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def comfy_image_to_cv2(comfy_image: torch.Tensor) -> np.ndarray:
    """A ComfyUI IMAGE tensor ``(B, H, W, C)`` -> a cv2 BGR image."""
    if comfy_image.ndim == 3:
        comfy_image = comfy_image.unsqueeze(0)
    # The scale, the uint8 cast and the RGB -> BGR flip run on the tensor at
    # its device (the cast truncates like the numpy ``astype`` it replaced) -
    # the copy to the host happens once, as uint8.
    img_t = (comfy_image.squeeze(0) * 255).to(torch.uint8)
    return img_t[..., [2, 1, 0]].cpu().numpy()


def cv2_to_comfy_image(cv2_image: np.ndarray) -> torch.Tensor:
    """A cv2 BGR image -> a ComfyUI IMAGE tensor ``(1, H, W, C)``."""
    bgr_t = torch.from_numpy(np.ascontiguousarray(cv2_image))
    rgb_t = bgr_t[..., [2, 1, 0]].to(torch.float32) / 255.0
    return rgb_t.unsqueeze(0)


def resize_bgr_u8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a uint8 BGR image ``(H, W, 3)`` with torch bilinear filtering.

    The torch-native replacement of the INTER_LINEAR resizes of the
    processor: ``size`` keeps the cv2 ``(width, height)`` order, both
    libraries share the half-pixel convention (``align_corners=False``) and
    the measured residual against cv2 is at most 1/255 (the uint8 rounding)
    over the geometry matrix of ``tests/test_image_op_equivalence.py``, with
    an identity resize being exact.
    """
    height, width = size[1], size[0]
    tensor = torch.from_numpy(np.ascontiguousarray(image))
    resized = F.interpolate(
        tensor.permute(2, 0, 1).unsqueeze(0).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    return (
        resized.squeeze(0)
        .permute(1, 2, 0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .contiguous()
        .numpy()
    )
