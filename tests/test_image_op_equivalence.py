"""The measured old-vs-new equivalence of the replaced cv2 / scipy ops.

The dependency slimming replaced the cv2 channel conversions, the INTER_LINEAR
resizes and the single scipy smoothing with torch-native implementations
(design §5 of ``.dev/docs/2026-10-25-keep-dependency-slimming.md``).  Every
replacement is pinned here against the library it replaced, with the tolerance
the probes measured - measured, not assumed.  The background LANCZOS4 resize
is replaced by the owner-accepted GPU Lanczos (the tolerance-accepted route of
2026-10-26) and is pinned against PIL (the filter it implements) and against
cv2 (the accepted drift).
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


# ---------------------------------------------------------------------------
# The GPU Lanczos resample (the owner-accepted replacement of the background
# cv2.resize INTER_LANCZOS4 - the tolerance-accepted route, 2026-10-26)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpu_ops():
    return importlib.import_module("ComfyUI_KEEP.src.keep_gpu_ops")


def _lanczos_corpus(height: int, width: int, seed: int) -> np.ndarray:
    """A gradient + blob + noise frame - smooth content and adversarial content."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    gx, gy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    blob = np.exp(-(((x - width * 0.4) / (width * 0.18)) ** 2 + ((y - height * 0.45) / (height * 0.18)) ** 2))
    rng = np.random.default_rng(seed)
    channel = 255 * (0.35 * gx + 0.35 * gy + 0.3 * blob) + rng.normal(0, 18, (height, width))
    image = np.stack([channel, channel * 0.9, channel * 0.8], -1)
    return np.clip(image, 0, 255).astype(np.uint8)[None, ...]


@pytest.mark.unit
def test_resize_lanczos_u8_matches_pil_lanczos(gpu_ops):
    """The op reproduces PIL's LANCZOS - the filter it implements.

    PIL is the semantic reference (three lobes, the support widened on a
    downscale, the out-of-range taps dropped and renormalized); the measured
    residual is the float32 coefficient and accumulation rounding.
    """
    from PIL import Image

    frames = np.concatenate([_lanczos_corpus(240, 320, 7), _lanczos_corpus(240, 320, 11)])
    for out_w, out_h in ((480, 360), (160, 120)):
        out = gpu_ops.resize_lanczos_u8(frames, (out_w, out_h), torch.device("cpu")).numpy()
        reference = np.stack(
            [
                np.asarray(
                    Image.fromarray(frame).resize((out_w, out_h), resample=Image.Resampling.LANCZOS)
                )
                for frame in frames
            ]
        )
        diff = np.abs(out.astype(np.int16) - reference.astype(np.int16))
        assert diff.mean() <= 0.3, f"mean {diff.mean():.3f} at {out_w}x{out_h}"
        assert diff.max() <= 12, f"max {diff.max()} at {out_w}x{out_h}"


@pytest.mark.unit
def test_resize_lanczos_u8_documents_the_accepted_cv2_drift(gpu_ops):
    """The drift against cv2's INTER_LANCZOS4 - measured and accepted.

    The owner chose the tolerance-accepted GPU resample (2026-10-26) over a
    bit-exact port: the filter is PIL's three-lobe Lanczos, and against cv2's
    fixed 8-tap ``INTER_LANCZOS4`` the measured envelope is mean 0.85 / max 8
    on an upscale and mean 7.9 / max 54 on a downscale (cv2 does not widen
    the kernel on a downscale - the difference is its aliasing).  The bounds
    below are that envelope, not a permission to grow.
    """
    frames = np.concatenate([_lanczos_corpus(240, 320, 7), _lanczos_corpus(240, 320, 11)])
    up = gpu_ops.resize_lanczos_u8(frames, (480, 360), torch.device("cpu")).numpy()
    up_ref = np.stack(
        [cv2.resize(f, (480, 360), interpolation=cv2.INTER_LANCZOS4) for f in frames]
    )
    diff = np.abs(up.astype(np.int16) - up_ref.astype(np.int16))
    assert diff.mean() <= 1.0, f"upscale mean {diff.mean():.3f}"
    assert diff.max() <= 8, f"upscale max {diff.max()}"

    down = gpu_ops.resize_lanczos_u8(frames, (160, 120), torch.device("cpu")).numpy()
    down_ref = np.stack(
        [cv2.resize(f, (160, 120), interpolation=cv2.INTER_LANCZOS4) for f in frames]
    )
    diff = np.abs(down.astype(np.int16) - down_ref.astype(np.int16))
    assert diff.mean() <= 8.5, f"downscale mean {diff.mean():.3f}"
    assert diff.max() <= 54, f"downscale max {diff.max()}"


@pytest.mark.unit
def test_resize_lanczos_u8_chunking_is_transparent(gpu_ops, monkeypatch):
    """The chunked device path is the per-frame path, exactly."""
    frames = np.concatenate([_lanczos_corpus(240, 320, 7), _lanczos_corpus(240, 320, 11)])
    whole = gpu_ops.resize_lanczos_u8(frames, (480, 360), torch.device("cpu"))
    monkeypatch.setattr(gpu_ops, "LANCZOS_CHUNK_PIXELS", 240 * 320)
    per_frame = gpu_ops.resize_lanczos_u8(frames, (480, 360), torch.device("cpu"))
    assert torch.equal(whole, per_frame)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the GPU pin needs a CUDA device")
def test_resize_lanczos_u8_on_the_gpu_matches_the_cpu_result(gpu_ops):
    """The device result is the CPU result within the float32 reduction order."""
    frames = _lanczos_corpus(240, 320, 7)
    cpu = gpu_ops.resize_lanczos_u8(frames, (480, 360), torch.device("cpu"))
    gpu = gpu_ops.resize_lanczos_u8(frames, (480, 360), torch.device("cuda"))
    diff = (cpu.float() - gpu.float().cpu()).abs().max()
    assert diff <= 4, f"the GPU result drifted {diff:.1f} levels"
