"""The batched paste-back of the KEEP sequence pipeline (design section 7.7).

The paste stage of ``process_image_sequence`` is the largest redesignable
stage of the profile (16.3% of the GPU time, 31.1% of the CPU time - design
section 5.4): the vendored ``paste_faces_to_input_image``
(``modules/deps/facelib/utils/face_restoration_helper.py:355-528``) runs,
**per frame**, a ParseNet forward, two Gaussian blurs, two ``cv2.warpAffine``
calls and the blend.

This module runs the default branch of that stage on the device:

- the ParseNet forward runs in **chunks** - the VRAM-aware cap of the
  design (the KJNodes pattern of design section 6),
- the parse mask (the colormap assignment, the two blurs and the border
  zeroing) is the batched op of :mod:`.keep_gpu_ops`,
- the inverse warp of the faces and the warp of the parse mask use the
  batched grid machinery of :mod:`.keep_gpu_ops`,
- the blend is one tensor op,
- the optional background upscaler runs as **one** ``tiled_scale`` call over
  the batch instead of one call per frame.

The **non-default branches stay with the vendored paste** (design section
7.7, step 6): ``draw_box``, a face upsampler, the grayscale frames and a
face helper without a parse net.  The vendored call path is not deleted -
it is the fallback of the processor.

The equivalence of every op against the vendored reference is **measured**
in ``tests/test_gpu_paste_equivalence.py`` (the per-op bounds are the ones
t1 measured, the end-to-end bound of the paste is measured here).

The module is pure torch / cv2 / numpy: it imports without a ComfyUI server
(the plan's section 3 import rule) - the ``comfy.utils.tiled_scale`` import
of the background upscaler is lazy.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch

from .keep_gpu_ops import parse_soft_masks, warp_affine_batch

#: The size the ParseNet works on (``face_restoration_helper.py:452`` -
#: ``cv2.resize(current_restored_face, (512, 512))``).
PARSE_FACE_SIZE = 512

#: The upper bound of the ParseNet chunk (design section 7.7, step 3): the
#: probe measured a VRAM cliff at 16 samples on a 16 GB card (7.0 s at 16
#: against 1.3 s at 4), so the cap stays inside the measured sweet spot.
PARSE_CHUNK_MAX = 8

#: The VRAM one ParseNet sample needs at 512x512 - the estimate behind the
#: ``get_free_memory`` sizing of the chunk (the KJNodes pattern).
PARSE_SAMPLE_BYTES = 1 << 30

#: The share of the free VRAM the paste staging may use.
STAGING_MEMORY_SHARE = 0.5

#: The hard cap of the paste staging, whatever the free VRAM says.
STAGING_BYTES = 2 << 30

#: The upper bound of the paste frame chunk (the staging of one chunk).
PASTE_CHUNK_MAX = 16

#: The gray threshold of the vendored ``is_gray``
#: (``modules/deps/facelib/utils/misc.py:146``).
GRAY_THRESHOLD = 10

#: The bytes per pixel the paste staging holds per frame: the float32
#: background (12), the float32 warped face (12), the float32 warped mask
#: (4) and the uint8 output (3).
STAGING_BYTES_PER_PIXEL = 31


# ---------------------------------------------------------------------------
# The VRAM-aware chunk sizes (design section 7.7, step 3)
# ---------------------------------------------------------------------------

def free_memory_bytes(device: torch.device) -> int:
    """The free memory of the device in bytes (0 when there is no host).

    The ComfyUI helper behind the KJNodes pattern (design section 6):
    ``model_management.get_free_memory(device, torch_free_too=True)`` counts
    the memory the torch cache can give back, so the chunk sizes follow the
    real headroom of the card.  The import is lazy (the pack imports without
    a ComfyUI server) and a host without ComfyUI falls back to the
    conservative defaults of the chunk functions.
    """
    try:
        from comfy import model_management
    except ImportError:
        return 0
    try:
        return int(model_management.get_free_memory(device, torch_free_too=True))
    except Exception:  # pragma: no cover - the host decides
        return 0


def parse_chunk_size(device: torch.device, maximum: int = PARSE_CHUNK_MAX) -> int:
    """The number of frames per ParseNet forward (design section 7.7, step 3).

    The KJNodes pattern (``nodes/image_nodes.py:5051-5057``): the chunk is
    the number of samples that fit into the free VRAM, capped at the
    measured sweet spot of 4-8.  Without a ComfyUI host the cap is used.
    """
    free = free_memory_bytes(device)
    if free <= 0:
        return maximum
    fits = int(free * STAGING_MEMORY_SHARE / PARSE_SAMPLE_BYTES)
    return max(1, min(maximum, fits))


def paste_chunk_size(
    height: int, width: int, device: torch.device, maximum: int = PASTE_CHUNK_MAX
) -> int:
    """The number of frames per paste chunk (the VRAM budget of the staging).

    The staging holds, per frame, the float32 background, the warped face,
    the warped mask and the uint8 output - about 31 bytes per pixel of the
    target canvas.
    """
    per_frame = max(1, int(height) * int(width) * STAGING_BYTES_PER_PIXEL)
    budget = STAGING_BYTES
    free = free_memory_bytes(device)
    if free > 0:
        budget = min(budget, int(free * STAGING_MEMORY_SHARE))
    return max(1, min(maximum, budget // per_frame))


# ---------------------------------------------------------------------------
# The gray flags of the frames
# ---------------------------------------------------------------------------

def is_gray_batch(frames_bgr_u8: np.ndarray) -> np.ndarray:
    """The per-frame ``is_gray`` flags (``facelib/utils/misc.py:146-160``).

    The vendored helper measures the variance of the three channel
    differences of every frame - through PIL, so one round trip per frame.
    This form computes the very same three variances with numpy, frame by
    frame (the memory of one frame at a time): the flags are **bit-exact**
    against the vendored call, because the arrays, their dtype and their
    shape are the same.

    Args:
        frames_bgr_u8: the ``(N, H, W, 3)`` uint8 BGR frames.

    Returns:
        ``(N,)`` bool - the flag of every frame.
    """
    frames = np.asarray(frames_bgr_u8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"is_gray_batch expects (N, H, W, 3), got {tuple(frames.shape)}"
        )
    flags = np.zeros(frames.shape[0], dtype=bool)
    for index in range(frames.shape[0]):
        frame = frames[index]
        blue = frame[..., 0].astype(np.int16)
        green = frame[..., 1].astype(np.int16)
        red = frame[..., 2].astype(np.int16)
        diff_sum = (
            (blue - green).var() + (green - red).var() + (red - blue).var()
        ) / 3.0
        flags[index] = diff_sum <= GRAY_THRESHOLD
    return flags


# ---------------------------------------------------------------------------
# The background of the paste (design section 7.7, step 1)
# ---------------------------------------------------------------------------

def _tiled_scale_batch(
    model, frames_bgr_u8: np.ndarray, device: torch.device
) -> np.ndarray:
    """The batched ``tiled_scale`` of the background frames.

    ``comfy.utils.tiled_scale`` loops over the images of the batch
    (``comfy/utils.py:1205-1206``) and processes every image on its own, so
    a batched call is **pixel-identical** to N per-frame calls (the model
    still sees B=1 tiles - the win is the removed per-frame round trips).
    ``output_device`` keeps the result on the compute device.

    The value scale is the one of the per-frame ``_run_upscaler``: the RGB
    float in 0..1 in, the clamped float out, quantized with the
    **truncating** uint8 cast of ``comfy_image_to_cv2`` and flipped back to
    BGR.
    """
    from comfy.utils import tiled_scale  # lazy: the pack imports without ComfyUI

    rgb = frames_bgr_u8[..., [2, 1, 0]].astype(np.float32) / 255.0
    samples = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).to(device)
    scaled = tiled_scale(
        samples,
        lambda a: model.model(a),
        tile_x=512,
        tile_y=512,
        overlap=64,
        upscale_amount=model.scale,
        output_device=device,
    )
    scaled = torch.clamp(scaled, min=0, max=1.0)
    quantized = (scaled * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
    return quantized[..., [2, 1, 0]].contiguous().cpu().numpy()


def batched_bg_frames(
    frames_bgr_u8: np.ndarray,
    upscale_factor: float,
    bg_upscale_model,
    device: torch.device,
    canvas_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """The background of the paste, at the target size of the paste.

    Without a background model (the production default) the background is
    the frame itself; with one it is the **one** batched ``tiled_scale``
    call of :func:`_tiled_scale_batch`.  The LANCZOS4 resize to the target
    size stays on cv2 (there is no torch lanczos - the documented precedent
    of ``tests/test_image_op_equivalence.py``) and runs only when the shape
    mismatches, exactly like the per-frame path of the processor.

    Args:
        canvas_hw: the ``(height, width)`` of the paste canvas - the size of
            the image the vendored ``read_image`` produced (the frame
            upscaled for ``min(h, w) < 512``).  The vendored paste resizes
            the background to that canvas
            (``face_restoration_helper.py:365-370``); ``None`` keeps the
            raw frame size (the frames of 512 px and above, where the two
            sizes are the same).

    Returns:
        ``(N, h_up, w_up, 3)`` uint8 BGR.
    """
    frames = np.asarray(frames_bgr_u8)
    height, width = frames.shape[1:3]
    target_h, target_w = int(height * upscale_factor), int(width * upscale_factor)
    if bg_upscale_model is None:
        background = frames
    else:
        background = _tiled_scale_batch(bg_upscale_model, frames, device)
    if background.shape[1] != target_h or background.shape[2] != target_w:
        background = np.stack(
            [
                cv2.resize(
                    background[index], (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
                )
                for index in range(background.shape[0])
            ]
        )
    if canvas_hw is not None:
        canvas_h, canvas_w = canvas_hw
        canvas_target_h = int(canvas_h * upscale_factor)
        canvas_target_w = int(canvas_w * upscale_factor)
        if background.shape[1] != canvas_target_h or background.shape[2] != canvas_target_w:
            background = np.stack(
                [
                    cv2.resize(
                        background[index],
                        (canvas_target_w, canvas_target_h),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    for index in range(background.shape[0])
                ]
            )
    return background


# ---------------------------------------------------------------------------
# The batched paste (design section 7.7, steps 2-5)
# ---------------------------------------------------------------------------

def parse_soft_masks_batched(
    parse_net,
    faces_bchw: torch.Tensor,
    device: torch.device,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """The soft parse masks of the vendored paste, batched (design 7.7/3-4).

    The input of the ParseNet is the vendored one
    (``face_restoration_helper.py:452-462``): the restored face resized to
    512x512, the ``img2tensor`` BGR -> RGB flip, the float32 ``/255`` and
    the ``(t - 0.5) / 0.5`` normalize.  The postprocess is the batched
    :func:`~.keep_gpu_ops.parse_soft_masks` (the colormap assignment, the
    two Gaussian blurs, the border zeroing and the ``/255``).

    Args:
        parse_net: the vendored ParseNet - its forward returns the
            ``(N, 19, H, W)`` logits as its **first** value.
        faces_bchw: the ``(N, 3, 512, 512)`` uint8 **BGR** restored faces.
        device: the compute device.
        chunk_size: the frames per forward; ``None`` sizes it from the free
            VRAM (:func:`parse_chunk_size`).

    Returns:
        ``(N, 1, 512, 512)`` float32 in [0, 1].
    """
    if faces_bchw.ndim != 4 or faces_bchw.shape[1] != 3:
        raise ValueError(
            f"parse_soft_masks_batched expects (N, 3, H, W), got {tuple(faces_bchw.shape)}"
        )
    num_frames = int(faces_bchw.shape[0])
    if chunk_size is None:
        chunk_size = parse_chunk_size(device)
    chunk_size = max(1, int(chunk_size))
    masks = []
    for start in range(0, num_frames, chunk_size):
        stop = min(start + chunk_size, num_frames)
        # the vendored input chain: the float32 /255, the BGR -> RGB flip
        # and the (t - 0.5) / 0.5 normalize
        chunk = faces_bchw[start:stop].to(torch.float32).div(255.0)
        chunk = chunk[:, [2, 1, 0]]
        chunk = chunk.sub_(0.5).div_(0.5)
        with torch.no_grad():
            logits = parse_net(chunk)[0]
        masks.append(parse_soft_masks(logits))
    if not masks:
        return torch.zeros(
            (0, 1, faces_bchw.shape[2], faces_bchw.shape[3]),
            dtype=torch.float32,
            device=faces_bchw.device,
        )
    return torch.cat(masks, dim=0)


def _quantize_and_clip(blended: torch.Tensor) -> torch.Tensor:
    """The quantizing tail of the vendored paste (``:497-507``).

    The vendored code picks its clip by the maximum of the blended image:
    a 0..255 image is clipped, an image above 255 is rescaled and an image
    below 1.001 is treated as a 0..1 float and multiplied by 255.  The
    blend of two uint8 images can only reach the first and the third case;
    the second one is kept for the fidelity of the branch.
    """
    maximum = blended.amax(dim=(1, 2, 3), keepdim=True)
    rescale = torch.where(
        maximum > 255.001, 255.0 / maximum.clamp(min=1e-12), torch.ones_like(maximum)
    )
    factor = torch.where(maximum > 1.001, rescale, torch.full_like(maximum, 255.0))
    return (blended * factor).clamp_(0, 255)


def paste_faces_batch(
    faces_bgr: np.ndarray,
    inverse_affines: Sequence[np.ndarray],
    background_bgr: np.ndarray,
    device: torch.device,
    parse_net,
    parse_chunk: int | None = None,
    frame_chunk: int | None = None,
) -> list[np.ndarray]:
    """The batched paste-back of the restored faces (design section 7.7).

    The batched form of the vendored paste
    (``face_restoration_helper.py:406, 452-478, 495``):

    1. the ParseNet forward in chunks and the batched parse mask,
    2. the inverse warp of the restored faces - quantized to the uint8 grid
       the vendored ``cv2.warpAffine`` produces (it warps uint8 -> uint8),
    3. the warp of the parse mask (float32, ``INTER_LINEAR`` semantics),
    4. the blend ``mask * face + (1 - mask) * background`` and the
       quantizing tail of the vendored code.

    Args:
        faces_bgr: the ``(N, 512, 512, 3)`` uint8 BGR restored faces.
        inverse_affines: the ``N`` inverse warps of the paste - the
            ``cv2.invertAffineTransform(affine) * upscale_factor`` of the
            vendored ``get_inverse_affine`` (:func:`.keep_gpu_ops.inverse_affines`).
        background_bgr: the ``(N, h_up, w_up, 3)`` uint8 BGR background, at
            the target size of the paste (:func:`batched_bg_frames`).
        device: the compute device.
        parse_net: the vendored ParseNet.
        parse_chunk: the frames per ParseNet forward (``None`` sizes it
            from the free VRAM).
        frame_chunk: the frames per paste chunk (``None`` sizes it from the
            free VRAM).

    Returns:
        The ``N`` pasted frames as uint8 BGR arrays.
    """
    faces = np.asarray(faces_bgr)
    background = np.asarray(background_bgr)
    num_frames = int(faces.shape[0])
    if faces.ndim != 4 or faces.shape[1:3] != (PARSE_FACE_SIZE, PARSE_FACE_SIZE):
        # the vendored paste resizes the face to the ParseNet size and the
        # parse mask back to the face size - the batched path keeps the
        # production size (512x512) and leaves the other sizes to the
        # vendored paste.
        raise ValueError(
            f"paste_faces_batch expects (N, {PARSE_FACE_SIZE}, {PARSE_FACE_SIZE}, 3) "
            f"faces, got {tuple(faces.shape)}"
        )
    if len(inverse_affines) != num_frames:
        raise ValueError(
            f"paste_faces_batch: {len(inverse_affines)} inverse affines for "
            f"{num_frames} faces"
        )
    if background.shape[0] != num_frames:
        raise ValueError(
            f"paste_faces_batch: {background.shape[0]} background frames for "
            f"{num_frames} faces"
        )
    height, width = background.shape[1:3]
    out_size = (width, height)  # the cv2 dsize order
    if frame_chunk is None:
        frame_chunk = paste_chunk_size(height, width, device)

    pasted: list[np.ndarray] = []
    for start in range(0, num_frames, max(1, int(frame_chunk))):
        stop = min(start + max(1, int(frame_chunk)), num_frames)
        pasted.extend(
            _paste_chunk(
                faces[start:stop],
                list(inverse_affines[start:stop]),
                background[start:stop],
                device,
                parse_net,
                out_size,
                parse_chunk,
            )
        )
    return pasted


def _paste_chunk(
    faces: np.ndarray,
    inverse_affines: list[np.ndarray],
    background: np.ndarray,
    device: torch.device,
    parse_net,
    out_size: tuple[int, int],
    parse_chunk: int | None,
) -> list[np.ndarray]:
    """One chunk of the batched paste (the steps of :func:`paste_faces_batch`)."""
    faces_u8 = torch.from_numpy(np.ascontiguousarray(faces)).to(device)
    faces_bchw = faces_u8.permute(0, 3, 1, 2)  # (N, 3, 512, 512) BGR

    # 1. the parse mask of the faces
    masks = parse_soft_masks_batched(parse_net, faces_bchw, device, parse_chunk)

    # 2. the inverse warp of the faces - it stays on cv2 (tolerance decision
    # D2): the warp is a uint8-bound op and the batched ``grid_sample`` of
    # the device drifts up to 8/255 against cv2's fixed-point sampling on
    # high-frequency content.  The vendored warp is bit-exact.
    warped_face = np.stack(
        [
            cv2.warpAffine(faces[index], inverse_affines[index], out_size)
            for index in range(faces.shape[0])
        ]
    )
    warped_face = (
        torch.from_numpy(warped_face).to(device).permute(0, 3, 1, 2).to(torch.float32)
    )

    # 3. the warp of the parse mask (float32, INTER_LINEAR) - the float32
    # path of cv2 samples exactly on the 1/32 px grid of the device warp
    warped_mask = warp_affine_batch(masks, inverse_affines, out_size)

    # 4. the blend
    background_t = torch.from_numpy(np.ascontiguousarray(background)).to(device)
    background_t = background_t.permute(0, 3, 1, 2).to(torch.float32)
    blended = warped_mask * warped_face + (1.0 - warped_mask) * background_t
    blended = _quantize_and_clip(blended).round_().to(torch.uint8)

    return list(blended.permute(0, 2, 3, 1).contiguous().cpu().numpy())
