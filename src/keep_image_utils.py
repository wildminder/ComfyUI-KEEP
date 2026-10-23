"""The conversions between the ComfyUI IMAGE tensors and the cv2 images.

The functions moved here from ``modules/utils.py`` in P3.2 - they are the
image half of the old utility module, the config half lives in
``src/keep_model_configs.py``.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def comfy_image_to_cv2(comfy_image: torch.Tensor) -> np.ndarray:
    """A ComfyUI IMAGE tensor ``(B, H, W, C)`` -> a cv2 BGR image."""
    if comfy_image.ndim == 3:
        comfy_image = comfy_image.unsqueeze(0)
    img_np = comfy_image.cpu().numpy()
    img_np = (img_np.squeeze(0) * 255).astype(np.uint8)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_cv2


def cv2_to_comfy_image(cv2_image: np.ndarray) -> torch.Tensor:
    """A cv2 BGR image -> a ComfyUI IMAGE tensor ``(1, H, W, C)``."""
    img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype(np.float32) / 255.0
    comfy_image = torch.from_numpy(img_np).unsqueeze(0)
    return comfy_image
