"""The batched GPU ops of the sequence pipeline redesign.

The ops replace the per-frame cv2 / numpy loops of
``process_image_sequence`` with one batched torch op each (design section 7
of ``.dev/docs/2026-10-25-keep-gpu-video-pipeline.md``); the per-op
equivalences against the vendored references are **measured** in
``tests/test_gpu_batch_equivalence.py``.

The module is pure torch / cv2 / numpy: it imports without a ComfyUI server
(the plan's section 3 import rule) and leaves the chunking drivers (the KEEP
chunks, the ParseNet batch size, the VRAM-aware staging) to its callers.

The equivalence strategy per op:

- the affine matrices come from the **same vendored cv2 call**
  (``cv2.estimateAffinePartial2D(..., LMEDS)``), so the capture is identity
  by construction and the equivalence work moves to the warps;
- the warps use ``affine_grid`` + ``grid_sample`` (``align_corners=False``,
  the half-pixel convention cv2 shares).  cv2 quantizes the sampling
  coordinate to 1/32 px (``INTER_TAB_SIZE``) while torch samples in float32,
  so a per-pixel residual of up to ``local gradient / 64`` against the cv2
  output is expected - the tests measure the residual on smooth and on
  adversarial content and pin the measured bound;
- a non-zero ``border_value`` is composited through a ones-map sample (the
  in-bounds weight of the same grid), which reproduces cv2's per-tap
  ``BORDER_CONSTANT`` blending instead of the black border a zero-padded
  ``grid_sample`` alone would give;
- the ``tensor2img`` batch is the vendored elementwise chain on the stacked
  tensor (numpy and torch both round half-to-even) - bit-exact;
- the parse postprocess keeps the exact ``cv2.getGaussianKernel`` kernel and
  the ``BORDER_REFLECT101`` extension of ``cv2.GaussianBlur`` (torch pad
  ``mode="reflect"``), as a two-pass separable ``conv2d``.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

#: The gray ``BORDER_CONSTANT`` value of the vendored face crop
#: (``face_restoration_helper.py:326-328``), in BGR order.
CROP_BORDER_BGR = (135, 133, 132)

#: The parse class -> mask value table of the vendored paste
#: (``face_restoration_helper.py:465``).
PARSE_MASK_COLORMAP = (
    0, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 0, 255, 0, 0, 0,
)

#: The zeroed border width of the parse mask
#: (``face_restoration_helper.py:469-471``).
PARSE_MASK_BORDER = 10


# ---------------------------------------------------------------------------
# Affine capture (design section 7.3) - the same vendored cv2 call
# ---------------------------------------------------------------------------

def estimate_affines(
    landmarks: Sequence[np.ndarray], face_template: np.ndarray
) -> list[np.ndarray | None]:
    """The per-frame crop affines of ``align_warp_face``, computed once.

    The same vendored call as ``face_restoration_helper.py:314`` -
    ``cv2.estimateAffinePartial2D(landmark, face_template, method=cv2.LMEDS)``
    - so the matrices are identical to the ones the vendored helper computes
    for the same landmarks (identity by construction, pinned in the
    equivalence tests).  A ``None`` entry (the LMEDS failure of the vendored
    ``:315-319`` branch, e.g. degenerate landmarks) is preserved; the callers
    map those frames to their zero-face / bg-only handling exactly like the
    processor does today.
    """
    return [
        cv2.estimateAffinePartial2D(np.asarray(landmark), face_template, method=cv2.LMEDS)[0]
        for landmark in landmarks
    ]


def inverse_affines(
    affine_matrices: Sequence[np.ndarray | None], upscale_factor: float = 1.0
) -> list[np.ndarray | None]:
    """The inverse warps of ``get_inverse_affine`` (``face_restoration_helper.py:341-343``).

    ``cv2.invertAffineTransform`` first, then the multiply with the upscale
    factor - the same order as the vendored method, so the float result is
    identical.  ``None`` entries pass through, like the vendored ``:338-340``.
    """
    return [
        None if matrix is None else cv2.invertAffineTransform(matrix) * upscale_factor
        for matrix in affine_matrices
    ]


# ---------------------------------------------------------------------------
# Batched warps (design sections 7.4 and 7.7)
# ---------------------------------------------------------------------------

def _thetas_from(
    matrices: Sequence[np.ndarray],
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Affine matrices -> the normalized-space thetas of ``F.affine_grid``.

    ``cv2.warpAffine(img, M, dsize)`` (without ``WARP_INVERSE_MAP``) samples
    ``dst(x) = src(inv(M) @ x)`` - the probe of the design measured this
    form.  ``F.affine_grid`` wants that map in the normalized coordinates of
    ``align_corners=False``, where a pixel coordinate ``p`` on an axis of
    size ``S`` is ``n = (2p + 1 - S) / S``.  Substituting the pixel maps of
    both sides into ``p_src = A @ p_dst + t`` gives, per output axis, the
    theta row ``[a_x * D_w / S, a_y * D_h / S, (a_x * (D_w - 1) + a_y * (D_h
    - 1) + 2 * t - (S - 1)) / S]`` with ``D`` the output and ``S`` the source
    size of the axis.

    cv2 samples on a 1/32 px fixed-point grid; replicating that quantization
    in the grid was measured (``.dev/tmp/probe_fixedpoint6.py``) and made the
    residual **worse** (the crop drifted 12/255 instead of 8/255), so the
    grid stays in the float form of cv2's own coordinates.
    """
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    thetas = np.zeros((len(matrices), 2, 3), dtype=np.float64)
    for i, matrix in enumerate(matrices):
        if matrix is None:
            raise ValueError(
                "warp_affine_batch: a None affine matrix - filter the "
                "no-face frames to their zero / bg-only handling first"
            )
        inv = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float64))
        ax, ay, t = inv[:, 0], inv[:, 1], inv[:, 2]
        thetas[i, 0, 0] = ax[0] * dst_w / src_w
        thetas[i, 0, 1] = ay[0] * dst_h / src_w
        thetas[i, 0, 2] = (ax[0] * (dst_w - 1) + ay[0] * (dst_h - 1) + 2.0 * t[0] - (src_w - 1)) / src_w
        thetas[i, 1, 0] = ax[1] * dst_w / src_h
        thetas[i, 1, 1] = ay[1] * dst_h / src_h
        thetas[i, 1, 2] = (ax[1] * (dst_w - 1) + ay[1] * (dst_h - 1) + 2.0 * t[1] - (src_h - 1)) / src_h
    return torch.from_numpy(thetas).to(device=device, dtype=torch.float32)


def warp_affine_batch(
    images: torch.Tensor,
    matrices: Sequence[np.ndarray],
    out_size: tuple[int, int],
    border_value: float | Sequence[float] | None = None,
) -> torch.Tensor:
    """The batched ``cv2.warpAffine`` (bilinear) over a stack of images.

    Args:
        images: ``(N, C, H, W)`` float tensor on the compute device, in the
            0..255 value scale of the cv2 images (an integer dtype is cast to
            float32; ``grid_sample`` needs floats).
        matrices: one 2x3 affine matrix per image - the **same** objects
            that would be handed to ``cv2.warpAffine(img, M, dsize)`` (the
            forward map; cv2 inverts internally and so does this op).
        out_size: ``(width, height)``, the cv2 ``dsize`` order.
        border_value: ``None`` (or 0) for the cv2 default black border;
            otherwise a scalar or a per-channel sequence - the crop's gray
            ``(135, 133, 132)`` (BGR).  A non-zero border is composited via a
            ones-map sample with the identical grid, so the boundary ring
            blends the border value per tap exactly like cv2's
            ``BORDER_CONSTANT`` (a zero-padded ``grid_sample`` alone would
            lose the in-bounds fraction of the ring).

    Returns:
        ``(N, C, out_h, out_w)`` float32 - not quantized; the callers that
        replace a uint8 cv2 warp quantize with ``round / clamp / to(uint8)``
        (see :func:`crop_faces_batch`).
    """
    if images.ndim != 4:
        raise ValueError(f"warp_affine_batch expects (N, C, H, W), got {tuple(images.shape)}")
    if len(matrices) != images.shape[0]:
        raise ValueError(
            f"warp_affine_batch: {len(matrices)} matrices for {images.shape[0]} images"
        )
    src_h, src_w = int(images.shape[-2]), int(images.shape[-1])
    out_w, out_h = (int(out_size[0]), int(out_size[1]))
    source = images if images.dtype == torch.float32 else images.float()

    thetas = _thetas_from(matrices, (src_h, src_w), (out_h, out_w), source.device)
    grid = F.affine_grid(
        thetas, (source.shape[0], source.shape[1], out_h, out_w), align_corners=False
    )
    sampled = F.grid_sample(
        source, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    if border_value is None:
        return sampled

    border = torch.as_tensor(border_value, dtype=source.dtype, device=source.device)
    if border.numel() not in (1, source.shape[1]):
        raise ValueError(
            f"border_value must be a scalar or one value per channel "
            f"({source.shape[1]}), got {border.numel()}"
        )
    # A zero-padded sample already carries the in-bounds fraction of the
    # interpolation weights; the ones-map sample with the identical grid
    # yields exactly that weight, so the missing part is border * (1 - w) -
    # the same per-tap blend cv2 performs against its border value.
    weight = F.grid_sample(
        torch.ones(
            (source.shape[0], 1, src_h, src_w), device=source.device, dtype=source.dtype
        ),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled + border.reshape(1, -1, 1, 1) * (1.0 - weight)


def crop_faces_batch(
    frames_bchw: torch.Tensor,
    affines: Sequence[np.ndarray],
    face_size: tuple[int, int] = (512, 512),
) -> torch.Tensor:
    """The batched crop of ``align_warp_face`` (design section 7.4).

    One :func:`warp_affine_batch` with the gray border of the vendored crop
    (``face_restoration_helper.py:326-328``), requantized to the uint8 grid
    today's ``cv2.warpAffine`` produces - the KEEP input must see the same
    quantization.  Chain :func:`ComfyUI_KEEP.src.keep_image_utils.bgr_u8_batch_to_normalized`
    where the per-frame ``img2tensor`` + normalize used to run.
    """
    warped = warp_affine_batch(frames_bchw, affines, face_size, border_value=CROP_BORDER_BGR)
    return warped.round().clamp_(0, 255).to(torch.uint8)


# ---------------------------------------------------------------------------
# tensor2img batch (design section 7.6)
# ---------------------------------------------------------------------------

def tensor2img_u8_batch(tensor: torch.Tensor, min_max: tuple[float, float] = (-1, 1)) -> torch.Tensor:
    """The vendored ``tensor2img`` elementwise chain on an RGB stack, on device.

    ``(N, 3, H, W)`` float -> ``(N, 3, H, W)`` uint8 BGR on the input's
    device - the quantized face stack the batched paste warp consumes
    (today's paste warps the uint8 ``tensor2img`` output, so the
    quantization has to happen before the warp).  The chain is the vendored
    ``modules/deps/basicsr/utils/img_util.py:66-93``: clamp -> rescale ->
    *255 -> round (half-to-even in torch as in numpy) -> uint8 -> RGB->BGR.
    """
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(
            f"tensor2img_u8_batch expects (N, 3, H, W) RGB, got {tuple(tensor.shape)}"
        )
    low, high = min_max
    t = tensor.detach().float().clamp(low, high)
    t = (t - low) / (high - low)
    t = (t * 255.0).round().to(torch.uint8)
    return t.permute(0, 2, 3, 1)[..., [2, 1, 0]].contiguous()


def tensor2img_batch(
    tensor: torch.Tensor, min_max: tuple[float, float] = (-1, 1)
) -> np.ndarray:
    """The batched vendored ``tensor2img`` (design section 7.6).

    The math of :func:`tensor2img_u8_batch` with a single host transfer at
    the end - the per-frame loop of the processor made one per frame.
    Every step is IEEE elementwise and ``round`` is half-to-even in both
    numpy and torch, so the output is bit-exact against the per-frame
    vendored call (pinned in the equivalence tests).

    Returns:
        ``(N, H, W, 3)`` uint8 BGR.
    """
    return tensor2img_u8_batch(tensor, min_max).cpu().numpy()


# ---------------------------------------------------------------------------
# Parse postprocess (design section 7.7, step 4)
# ---------------------------------------------------------------------------

def parse_colormap_batch(out_mask: torch.Tensor) -> torch.Tensor:
    """The raw parse masks of the vendored assignment loop, batched.

    ``(N, 19, H, W)`` parse logits -> ``(N, 1, H, W)`` float32 with the
    :data:`PARSE_MASK_COLORMAP` values - one ``argmax`` over the class axis
    and one gather of the colormap, the same values the per-class
    ``parse_mask_raw[out_parse == map_idx] = color_val`` loop of
    ``face_restoration_helper.py:463-466`` writes.
    """
    classes = out_mask.argmax(dim=1)
    colormap = torch.as_tensor(PARSE_MASK_COLORMAP, dtype=torch.float32, device=out_mask.device)
    return colormap[classes].unsqueeze(1)


def gaussian_blur_2d_reflect101(masks: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
    """The batched ``cv2.GaussianBlur`` for single-channel float32 masks.

    Two separable ``conv2d`` passes with the **exact**
    ``cv2.getGaussianKernel`` kernel (``CV_32F`` - the type cv2 builds for a
    float32 image) and the torch ``mode="reflect"`` padding, which mirrors
    without repeating the edge value: the same extension as cv2's default
    ``BORDER_REFLECT101``.  ``ksize`` is odd and the radius must stay below
    the spatial size (the parse masks blur at the 512 face size).
    """
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(f"gaussian_blur_2d_reflect101 needs an odd ksize >= 3, got {ksize}")
    if masks.dtype != torch.float32:
        raise ValueError(f"gaussian_blur_2d_reflect101 needs float32, got {masks.dtype}")
    kernel = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F).astype(np.float32).reshape(-1)
    radius = ksize // 2
    weight_h = torch.from_numpy(kernel).view(1, 1, 1, ksize).to(masks.device)
    weight_v = weight_h.view(1, 1, ksize, 1)
    blurred = F.pad(masks, (radius, radius, 0, 0), mode="reflect")
    blurred = F.conv2d(blurred, weight_h)
    blurred = F.pad(blurred, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(blurred, weight_v)


def parse_soft_masks(out_mask: torch.Tensor) -> torch.Tensor:
    """The soft parse masks of the vendored paste, batched (design section 7.7).

    ``(N, 19, H, W)`` parse logits -> ``(N, 1, H, W)`` float32 in [0, 1] -
    the batched form of ``face_restoration_helper.py:463-472``: the colormap
    assignment (:func:`parse_colormap_batch`), two Gaussian blurs
    (101, 101, sigma 11) via :func:`gaussian_blur_2d_reflect101`, the
    :data:`PARSE_MASK_BORDER` zeroing and the /255 scale.
    """
    masks = parse_colormap_batch(out_mask)
    masks = gaussian_blur_2d_reflect101(masks, 101, 11.0)
    masks = gaussian_blur_2d_reflect101(masks, 101, 11.0)
    border = PARSE_MASK_BORDER
    masks[:, :, :border, :] = 0
    masks[:, :, -border:, :] = 0
    masks[:, :, :, :border] = 0
    masks[:, :, :, -border:] = 0
    return masks / 255.0
