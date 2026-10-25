"""The measured old-vs-new equivalence of the replaced cv2 / scipy ops.

The dependency slimming replaced the cv2 channel conversions, the INTER_LINEAR
resizes and the single scipy smoothing with torch-native implementations
(design §5 of ``.dev/docs/2026-10-25-keep-dependency-slimming.md``).  Every
replacement is pinned here against the library it replaced, with the tolerance
the probes measured - measured, not assumed.  The LANCZOS4 resizes stay on
cv2 (no torch/kornia lanczos exists) and have no replacement to pin.
"""

from __future__ import annotations

import importlib

import cv2
import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def image_utils():
    return importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")


@pytest.fixture(scope="module")
def processor():
    return importlib.import_module("ComfyUI_KEEP.src.keep_processor")


# ---------------------------------------------------------------------------
# The ComfyUI <-> cv2 conversions (rows 1-2 of the design matrix)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_comfy_image_to_cv2_equals_the_old_implementation(image_utils):
    """The tensor path reproduces the numpy scale/cast and the cv2 flip exactly.

    The reference is the complete former implementation: the ``*255`` scale
    and the uint8 cast in numpy (the cast truncates) and the
    ``cv2.cvtColor(COLOR_RGB2BGR)`` flip.
    """
    tensor = torch.rand(
        (1, 16, 24, 3), generator=torch.Generator().manual_seed(7)
    )

    converted = image_utils.comfy_image_to_cv2(tensor)

    img_np = (tensor.numpy().squeeze(0) * 255).astype(np.uint8)
    reference = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    assert converted.dtype == np.uint8
    assert np.array_equal(converted, reference)


@pytest.mark.unit
def test_cv2_to_comfy_image_equals_the_old_implementation(image_utils):
    """The tensor flip and the float32 scale reproduce the cv2 path bit for bit.

    The reference is the complete former implementation: the
    ``cv2.cvtColor(COLOR_BGR2RGB)`` flip and the ``astype(float32) / 255``
    scale in numpy.
    """
    rng = np.random.default_rng(11)
    img_bgr = rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8)

    converted = image_utils.cv2_to_comfy_image(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    reference = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)
    assert converted.dtype == torch.float32
    assert torch.equal(converted, reference)


# ---------------------------------------------------------------------------
# resize_bgr_u8 vs cv2.INTER_LINEAR (rows 3-5 of the design matrix)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resize_bgr_u8_matches_cv2_inter_linear_over_the_geometry_matrix(image_utils):
    """20 geometries, max residual 1/255 - the uint8 rounding the probes measured."""
    rng = np.random.default_rng(42)
    worst = 0
    for height, width in [(512, 512), (480, 640), (333, 500), (700, 700), (513, 383)]:
        image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        for target_h, target_w in [(512, 512), (1024, 1024), (256, 256), (511, 767)]:
            reference = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized = image_utils.resize_bgr_u8(image, (target_w, target_h))
            assert resized.shape == (target_h, target_w, 3)
            assert resized.dtype == np.uint8
            diff = np.abs(reference.astype(int) - resized.astype(int)).max()
            worst = max(worst, int(diff))
    assert worst <= 1, f"the torch bilinear path drifted {worst}/255 from cv2"


@pytest.mark.unit
def test_resize_bgr_u8_identity_is_exact(image_utils):
    """An identity resize copies the pixels (the half-pixel grids coincide)."""
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    assert np.array_equal(image_utils.resize_bgr_u8(image, (512, 512)), image)
    rect = rng.integers(0, 256, size=(333, 500, 3), dtype=np.uint8)
    assert np.array_equal(image_utils.resize_bgr_u8(rect, (500, 333)), rect)


@pytest.mark.unit
def test_resize_bgr_u8_on_a_gradient_ramp(image_utils):
    """The worst case for the rounding - a full-range ramp - stays within 1/255."""
    image = np.tile(
        np.linspace(0, 255, 640, dtype=np.float32).astype(np.uint8)[None, :, None],
        (480, 1, 3),
    )
    reference = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_LINEAR)
    resized = image_utils.resize_bgr_u8(image, (1280, 960))
    assert np.abs(reference.astype(int) - resized.astype(int)).max() <= 1


# ---------------------------------------------------------------------------
# The GRAY2BGR replacement (row 10 of the design matrix)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gray_to_bgr_equals_cv2_and_stays_writable(processor):
    """The torch stack equals cv2.cvtColor(GRAY2BGR) and returns a writable array.

    The array has to stay writable - the stride-0 ``expand`` view is read-only
    and the consumers downstream write into the cv2 images.
    """
    rng = np.random.default_rng(11)
    gray = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)

    converted = processor._gray_to_bgr(gray)

    reference = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert converted.dtype == np.uint8
    assert converted.flags.c_contiguous and converted.flags.writeable
    assert np.array_equal(converted, reference)


# ---------------------------------------------------------------------------
# _gaussian_smooth vs scipy.ndimage.gaussian_filter1d (row 11 of the matrix)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gaussian_smooth_matches_scipy_in_float64(processor):
    """The torch smoothing equals scipy to 1e-10 (float64, the probe ran 4.3e-14)."""
    scipy_gaussian = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(42)
    for num_frames in (3, 5, 20, 51, 200):
        values = rng.normal(size=(num_frames, 10)) * 100.0
        reference = scipy_gaussian.gaussian_filter1d(values, sigma=5, axis=0)
        smoothed = processor._gaussian_smooth(values, sigma=5.0)
        assert smoothed.dtype == np.float64
        diff = np.abs(reference - smoothed).max()
        assert diff <= 1e-10, f"n={num_frames}: the difference is {diff:.3e}"


@pytest.mark.unit
def test_gaussian_smooth_matches_scipy_for_other_sigmas(processor):
    """The kernel formula holds for the sigma neighbourhood of the call site."""
    scipy_gaussian = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(5)
    values = rng.normal(size=(40, 10)) * 100.0
    for sigma in (1.0, 2.5, 5.0):
        reference = scipy_gaussian.gaussian_filter1d(values, sigma=sigma, axis=0)
        smoothed = processor._gaussian_smooth(values, sigma=sigma)
        diff = np.abs(reference - smoothed).max()
        assert diff <= 1e-10, f"sigma={sigma}: the difference is {diff:.3e}"


@pytest.mark.unit
def test_gaussian_smooth_keeps_the_float32_dtype_of_the_landmarks(processor):
    """The landmarks arrive as float32 - the output stays float32, like scipy's.

    Both implementations compute in float64 and cast the final value once, so
    the measured difference on the probe data is 0 (bit-equal); the bound is
    one float32 ulp at the landmark magnitude, in case a value lands on a
    rounding boundary.
    """
    scipy_gaussian = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(9)
    values = (rng.normal(size=(51, 10)) * 512.0).astype(np.float32)

    smoothed = processor._gaussian_smooth(values, sigma=5.0)

    assert smoothed.dtype == np.float32
    reference = scipy_gaussian.gaussian_filter1d(values, sigma=5, axis=0)
    diff = np.abs(reference.astype(np.float64) - smoothed.astype(np.float64)).max()
    assert diff <= 1e-4, f"the difference is {diff:.3e}"
