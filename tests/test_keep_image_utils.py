"""P3.2 - the image conversions of ``src/keep_image_utils.py``.

The functions moved here from ``modules/utils.py`` in P3.2; the tests assert
the round-trip and the channel order, so a move that changed the conversion
would be caught.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def image_utils():
    return importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")


@pytest.mark.unit
def test_comfy_image_to_cv2_keeps_the_shape_and_the_channel_order(image_utils):
    """An RGB IMAGE tensor becomes a BGR uint8 image."""
    # one red pixel (RGB) inside a 2x2 image
    tensor = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    tensor[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])

    image = image_utils.comfy_image_to_cv2(tensor)

    assert isinstance(image, np.ndarray)
    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    # RGB -> BGR: the red channel moves to the last position
    assert tuple(image[0, 0]) == (0, 0, 255)


@pytest.mark.unit
def test_a_single_image_is_unsqueezed(image_utils):
    """A ``(H, W, C)`` tensor is accepted as well."""
    tensor = torch.full((4, 4, 3), 0.5, dtype=torch.float32)
    image = image_utils.comfy_image_to_cv2(tensor)
    assert image.shape == (4, 4, 3)


@pytest.mark.unit
def test_the_round_trip_preserves_the_pixels(image_utils):
    """cv2 -> comfy -> cv2 keeps the pixels (within the uint8 rounding)."""
    rng = np.random.default_rng(1234)
    source = rng.integers(0, 256, size=(8, 6, 3), dtype=np.uint8)
    # the source is a BGR image
    tensor = image_utils.cv2_to_comfy_image(source)
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (1, 8, 6, 3)
    assert tensor.dtype == torch.float32

    restored = image_utils.comfy_image_to_cv2(tensor)
    assert restored.shape == source.shape
    assert np.abs(restored.astype(int) - source.astype(int)).max() <= 1
