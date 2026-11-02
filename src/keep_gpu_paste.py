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
  batched grid machinery of :mod:`.keep_gpu_ops` (the full-canvas
  reference chunk) - since the owner's fix 4 (2026-10-31) the ROI chunks
  warp each face straight into its canvas ROI instead (``F.affine_grid``
  / ``F.grid_sample`` off :func:`_compute_roi_theta`),
- the blend is one tensor op,
- the optional background upscaler runs as batched ``tiled_scale`` calls
  over the VRAM-aware frame groups of :func:`bg_chunk_size` instead of one
  call per frame - 2026-11-01: one call over the whole batch materialised
  the x4 intermediate of the whole clip (~0.37 GiB a 720p frame) and OOMed
  a 16 GB card on the owner's 48-frame clip before the first paste,
- the **gray flags** of the frames run on the device as well - the float64
  variances of the three channel differences, with the numpy computation as
  the fallback of the borderline frames (the paste-stage CPU optimization
  round, section 4.2).

The **non-default branches stay with the vendored paste** (design section
7.7, step 6): a face upsampler and a face helper without a parse net.
The vendored call path is not deleted - it is the fallback of the
processor.  Since the owner's fix 1.2 (2026-10-30) the grayscale frames
ride the batched core, and since the owner's fix 4 (2026-10-31)
``draw_box`` does too: the box is drawn on the device canvas / chunk
buffer by the processor
(:meth:`~.keep_processor.KEEPFaceProcessor._draw_boxes_on_canvas`) - the
host branch of the paste draws no box (a CPU run with ``draw_box`` keeps
no boxes on the batched frames; the fix's disclosed limitation).

The **multi-face paste core** (the cancel + multi-face round, t3; the
owner's fix 4, 2026-10-31) keeps the ``face_frame`` map of
:func:`paste_faces_batch` (face ``i`` composites into canvas row
``face_frame[i]``; ``None`` keeps the one-face-per-row identity): the
faces, the affines and the ROIs are **face-indexed**, the background, the
canvas and the output stay **frame-indexed**.  Both ROI chunks
(:func:`_paste_chunk_roi` and :func:`_paste_chunk_faces`) warp each face
straight into its canvas ROI - one ``F.affine_grid`` / ``F.grid_sample``
pair per face off :func:`_compute_roi_theta` - and blend ROI-local; the
multi-face form only re-bases the target row through the map.

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
import torch.nn.functional as F

from . import keep_interrupt
from .keep_gpu_ops import parse_soft_masks, resize_lanczos_u8, warp_affine_batch
from .keep_image_utils import bgr_u8_batch_to_comfy

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

#: The band around the gray threshold in which the **device** result is not
#: trusted: the frame falls back to the numpy computation.  The device
#: reduction order differs from numpy's pairwise summation, so the two
#: values can differ in the last bits - a frame whose result sits on the
#: threshold must be decided by the numpy value, or a flag could flip.  The
#: measured drift of the device variance against the numpy one is below
#: ``1e-9`` on a 720p frame (the float64 sums of ~1e6 terms), so the band
#: is three orders of magnitude above it.
GRAY_BORDERLINE_EPSILON = 1e-6

#: The pixels per chunk of the device gray flags - the VRAM budget of the
#: float64 temporaries (eight bytes per pixel and per difference).
GRAY_CHUNK_PIXELS = 4 << 20

#: The bytes per pixel the paste staging holds per frame: the float32
#: background (12), the float32 warped face (12), the float32 warped mask
#: (4) and the uint8 output (3).
STAGING_BYTES_PER_PIXEL = 31

#: The bytes per **scaled** pixel the background upscaler's ``tiled_scale``
#: stage holds at its peak: two float32 stacks of the scaled size - the
#: ``torch.empty`` output buffer of the installed ``tiled_scale``
#: (``comfy/utils.py:1203``, no dtype argument) and the out-of-place
#: clamp / ``* 255.0`` chain of :func:`_tiled_scale_batch` - plus the
#: uint8 quantized stack (``2 * 12 + 3 = 27``).  The bg frame groups of
#: :func:`batched_bg_frames` size from this, **not** from
#: :func:`paste_chunk_size`: its 31 B/px canvas budget would still allow
#: ~5.9 GiB of x4 intermediate per 720p group (~0.37 GiB a frame here).
BG_SCALED_PIXEL_BYTES = 27

#: The fixed-point grid of cv2's ``warpAffine`` (``imgwarp.cpp``): the
#: sampling coordinate is quantised to 1/32 px (``INTER_TAB_SIZE``) and the
#: matrix is pre-scaled to 1/1024 units (``AB_BITS``).
CV_WARP_TAB_SIZE = 32
CV_WARP_AB_SCALE = 1024
CV_WARP_ROUND_DELTA = CV_WARP_AB_SCALE // CV_WARP_TAB_SIZE // 2

#: The margin of the ROI around the transformed face footprint (1 px - it
#: absorbs the float rounding of the bounding box).
ROI_MARGIN = 1

#: The transfer counters of the paste - the round's evidence for
#: "ROI-sized, not canvas-sized".  ``h2d_bytes`` and ``d2h_bytes`` count the
#: ROI background upload and the blended ROI download; the ParseNet inputs
#: are counted apart (``face_h2d_bytes``) - they are not ROI-sized by
#: nature.  The counters describe the **last** paste call.
#: ``roi_staged_bytes`` is the device-output claim: with the output stack on
#: the device nothing travels, so the ROI-sized staging is recorded as the
#: bytes that **would** have travelled (the same numbers ``h2d_bytes`` /
#: ``d2h_bytes`` hold in the host-output mode).
#: The device canvas budget of the device-output mode: a batch of
#: ``N * H * W * 3`` bytes above it falls back to the host assembly (a
#: 200-frame 1080p batch is 622 MB).  The budget, not a rule of thumb,
#: decides.
PASTE_DEVICE_STACK_MAX_BYTES = 1 << 30

#: The share of the free VRAM the device-output canvas may use.  The full
#: mode holds the background AND the canvas on the device (about 2x the
#: checked bytes) plus the convert-out transient, so a quarter of free VRAM
#: keeps the engaged footprint inside the headroom.  Owner-tunable on CUDA.
PASTE_DEVICE_STACK_SHARE = 0.25

#: The key of :func:`_batched_paste`'s mapping that carries the device
#: canvas and its frame indices in the device-output mode.  The host mode
#: never sets it, so the ``{frame index: frame}`` contract is unchanged.
DEVICE_CANVAS_KEY = "__device_canvas__"

#: The key of :func:`_batched_paste`'s mapping that carries the converted
#: IMAGE rows of the chunked device-output mode - the
#: ``(N_eligible, canvas_h, canvas_w, 3)`` float32 host tensor of
#: :func:`paste_faces_batch_chunked`.  The full mode sets
#: :data:`DEVICE_CANVAS_KEY` (its rows are still one uint8 canvas), the
#: host mode sets neither, so the ``{frame index: frame}`` contract is
#: unchanged.
DEVICE_OUTPUT_CHUNKS_KEY = "__device_output_chunks__"

PASTE_TRANSFER_STATS: dict[str, int] = {
    "frames": 0,
    "h2d_bytes": 0,
    "d2h_bytes": 0,
    "face_h2d_bytes": 0,
    "canvas_bytes": 0,
    "roi_frames": 0,
    "fallback_frames": 0,
    "roi_staged_bytes": 0,
}


def reset_paste_transfer_stats() -> None:
    """Reset the transfer counters (they describe the last paste call)."""
    for key in PASTE_TRANSFER_STATS:
        PASTE_TRANSFER_STATS[key] = 0


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


def bg_chunk_size(height: int, width: int, scale: float, device: torch.device) -> int:
    """The number of frames per background-upscale group (the x4 budget).

    Sized from the **x4 intermediate**, not from the canvas: a 720p frame
    at the production x4 holds ~0.37 GiB of it (two float32 stacks of the
    scaled size plus the uint8 one, :data:`BG_SCALED_PIXEL_BYTES`), so
    :func:`paste_chunk_size`'s 31 B/px canvas budget would still allow
    ~5.9 GiB of x4 intermediate per group.  The budget is the staging one
    (:data:`STAGING_BYTES`, the free-VRAM share of
    :data:`STAGING_MEMORY_SHARE`); a host without ComfyUI falls back to
    the cap, which keeps the small batches of the pins and the tests in
    one group - the pre-grouping behaviour.  2026-11-01: the whole-clip
    ``tiled_scale`` call materialised the x4 intermediate of the entire
    clip - 17.8 GiB for the owner's 48 720p frames - and OOMed a 16 GB
    card before the first paste.
    """
    scaled_pixels = round(scale * int(height)) * round(scale * int(width))
    per_frame = BG_SCALED_PIXEL_BYTES * scaled_pixels
    budget = STAGING_BYTES
    free = free_memory_bytes(device)
    if free > 0:
        budget = min(budget, int(free * STAGING_MEMORY_SHARE))
    return max(1, budget // per_frame)


# ---------------------------------------------------------------------------
# The device-output assembly (the round of the GPU-resident output)
# ---------------------------------------------------------------------------

def device_canvas_budget(device: torch.device) -> int:
    """The VRAM budget of the device-output canvas (the design's D1).

    The mirror of :func:`paste_chunk_size`: the share of the free VRAM
    (:data:`PASTE_DEVICE_STACK_SHARE`) under the hard cap
    (:data:`PASTE_DEVICE_STACK_MAX_BYTES`), with the cap itself when the
    free memory is unavailable (a host without ComfyUI).  The cap keeps
    the full mode's engagement exactly as tight as the F1 one was - the
    share only degrades the budget gracefully on a small-free card.

    Note: with a ComfyUI host ``free_memory_bytes`` returns the **host**
    RAM for a CPU device (``comfy/model_management.py:1816-1818``), so a
    CPU budget is the cap - the pins patch ``free_memory_bytes`` (the
    shipped chunk-size pattern) for the determinism of smaller budgets.
    """
    free = free_memory_bytes(device)
    if free <= 0:
        return PASTE_DEVICE_STACK_MAX_BYTES
    return min(PASTE_DEVICE_STACK_MAX_BYTES, int(free * PASTE_DEVICE_STACK_SHARE))


def device_output_mode(
    device: torch.device, num_frames: int, canvas_h: int, canvas_w: int
) -> str:
    """The output-assembly mode of one paste call: ``full`` | ``chunked`` | ``host``.

    Fix 2.2 (2026-10-30, the owner's ``keep-fix3.txt``): a CUDA device
    never gets ``"host"`` - the free-VRAM chunk gate of the former body is
    gone and the selector returns ``"full"`` while the whole eligible
    batch's canvas fits the budget (:func:`device_canvas_budget`) and
    ``"chunked"`` otherwise.  A genuinely starved card now enters the
    chunked mode with the group floored at one frame
    (:func:`paste_chunk_size`) instead of the host assembly - the OOM
    handler stays the net (the owner-authorized trade).

    ``"full"`` - the F1 device canvas of the whole eligible batch
    (:func:`paste_faces_batch` with ``device_output=True``); ``"chunked"``
    - the same device assembly over frame groups, per group convert-out
    downloaded into a preallocated host buffer
    (:func:`paste_faces_batch_chunked`); ``"host"`` - the processor's host
    assembly (a CPU device).

    The chunk operand is :func:`paste_chunk_size` of the canvas - the same
    value the full mode derives internally - so the chunked mode's group
    boundaries coincide with the full mode's internal chunk boundaries and
    both modes run the same batch compositions (the CUDA batch
    sensitivity of the mask warp and of the Lanczos resize rides on that).

    The module stays torch / cv2 / numpy only.  The callers used to log the
    disengagement (a ``"host"`` verdict on a CUDA device) at their call
    sites - the owner's fix 4 (2026-10-31) replaced the batched-paste
    bodies wholesale and dropped that log line with them.
    """
    if torch.device(device).type != "cuda":
        return "host"
    canvas_bytes = int(canvas_h) * int(canvas_w) * 3
    budget = device_canvas_budget(device)
    if int(num_frames) * canvas_bytes <= budget:
        return "full"
    return "chunked"


def paste_canvas_hw(
    background_hw: tuple[int, int],
    upscale_factor: float,
    canvas_hw: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """The ``(height, width)`` the paste assembles at (the single source).

    The canvas is the background at the target size of the paste - the
    frame size times the upscale factor - widened to the canvas the
    vendored ``read_image`` produced when that is larger (the frame
    upscaled for ``min(h, w) < 512``; the vendored paste resizes the
    background to it, ``face_restoration_helper.py:365-370``).  In
    production the ``canvas_hw`` product is never smaller than the frame
    one (``read_image`` only upscales), so the ``max`` is defensive.

    Returns:
        ``(canvas_h, canvas_w)`` - the shape of
        :func:`paste_faces_batch`'s device-output canvas and of
        :func:`paste_faces_batch_chunked`'s IMAGE rows.
    """
    target_h = int(background_hw[0] * upscale_factor)
    target_w = int(background_hw[1] * upscale_factor)
    if canvas_hw is None:
        return target_h, target_w
    return (
        max(target_h, int(canvas_hw[0] * upscale_factor)),
        max(target_w, int(canvas_hw[1] * upscale_factor)),
    )


def comfy_convert_out(
    canvas: torch.Tensor,
    out: torch.Tensor | None = None,
    budget: int | None = None,
) -> torch.Tensor:
    """A uint8 BGR canvas -> a ComfyUI IMAGE batch, in budget-sized sub-chunks.

    The convert-out of the device-output modes: the same
    :func:`~.keep_image_utils.bgr_u8_batch_to_comfy` op chain - the RGB
    gather, the float32 cast and the ``/255`` - runs where the canvas
    lives and the float32 result downloads (the final IMAGE tensor the
    ComfyUI contract forces).

    The finding behind the sub-chunks: the chain is out-of-place and
    peaks at about **9x the canvas bytes** of device transient per frame
    (the gather 1x, the float cast 4x, the ``/255`` 4x) - at the full
    mode's own 1 GiB engagement limit that is a ~9 GiB transient, so the
    budget would not bound what it gates.  The conversion is elementwise
    - measured batch-invariant on CPU (the design's D3/D4) - so the
    sub-chunk of ``max(1, budget // (9 * frame_bytes))`` frames bounds
    the transient by the budget on both tails without changing a pixel.
    The float32 download equals the size of the final IMAGE tensor
    anyway; the uint8-download-convert-on-host alternative was rejected -
    it revives the host float math the round removes (the design's D4).

    Args:
        canvas: the ``(N, H, W, 3)`` uint8 BGR tensor (any device).
        out: the preallocated ``(N, H, W, 3)`` float32 **host** tensor the
            sub-chunk results are written into - ``None`` allocates it.
        budget: the device bytes the transient may peak at; ``None`` (and
            values at or below zero) converts the whole batch in one call.

    Returns:
        The ``(N, H, W, 3)`` float32 CPU IMAGE batch (``out``).
    """
    num_frames, height, width = (int(value) for value in canvas.shape[:3])
    if out is None:
        out = torch.empty(
            (num_frames, height, width, int(canvas.shape[3])), dtype=torch.float32
        )
    if budget is None or int(budget) <= 0:
        sub_chunk = max(1, num_frames)
    else:
        sub_chunk = max(1, int(budget) // (9 * height * width * int(canvas.shape[3])))
    for start in range(0, num_frames, sub_chunk):
        stop = min(start + sub_chunk, num_frames)
        out[start:stop] = bgr_u8_batch_to_comfy(canvas[start:stop])
    return out


def paste_faces_batch_chunked(
    faces_bgr: np.ndarray,
    inverse_affines: Sequence[np.ndarray],
    background_bgr: np.ndarray,
    upscale_factor: float,
    bg_upscale_model,
    device: torch.device,
    canvas_hw: tuple[int, int] | None,
    parse_net,
    frame_chunk: int | None = None,
    parse_chunk: int | None = None,
    face_frame: Sequence[int] | None = None,
    num_frames: int | None = None,
    stage_bar=None,
) -> torch.Tensor:
    """The device-output paste of the whole eligible batch, in frame groups.

    The chunked mode of :func:`device_output_mode`: the F1 device
    assembly (:func:`paste_faces_batch` with ``device_output=True``) over
    ``frame_chunk``-sized groups, with the background produced **per
    group** (a whole-batch device background would itself be canvas-sized
    and defeat the chunking) and the per-group convert-out
    (:func:`comfy_convert_out`) downloaded into one preallocated host
    float32 buffer.  The device footprint is one group's (raw upload +
    Lanczos out + canvas + the convert transient <= the budget),
    independent of the clip length - the point: the full mode's
    ``N * H * W * 3`` canvas disengages past the budget and revives the
    host assembly at 16x pixels.

    The group size defaults to :func:`paste_chunk_size` of the canvas -
    the same value the full mode derives internally, so the full mode's
    internal chunk boundaries and the groups coincide and both modes run
    the same batch compositions.  On CPU every op is measured
    batch-invariant (the design's D3), so the modes are bit-identical for
    any group size; on CUDA the boundary alignment is the structural
    guarantee, with the Lanczos background resize's batch sensitivity
    owner-gated - the one op that still sees the whole batch in the full
    mode and groups in this one.

    The vendored-fallback frames are not this function's business: only
    the eligible rows are converted; the processor writes the fallback
    rows host-side (one stack, one conversion) into its own buffer.

    Args:
        faces_bgr: the ``(N, 512, 512, 3)`` uint8 BGR restored faces of
            the eligible frames.
        inverse_affines: the ``N`` inverse warps of the paste - the
            ``cv2.invertAffineTransform(affine) * upscale_factor`` of the
            vendored ``get_inverse_affine``
            (:func:`.keep_gpu_ops.inverse_affines`).
        background_bgr: the ``(N, h, w, 3)`` uint8 BGR **raw** background
            frames (the processor's ``background_stacked``) - the resize
            to the canvas happens per group.
        upscale_factor: the ``final_upscale_factor`` of the paste.
        bg_upscale_model: the optional background upscaler (``None`` in
            the production default).
        device: the compute device.
        canvas_hw: the ``(height, width)`` of the vendored ``read_image``
            canvas (:func:`paste_canvas_hw`); ``None`` keeps the frame
            target.
        parse_net: the vendored ParseNet.
        frame_chunk: the frames per group (``None`` sizes it from the
            free VRAM - :func:`paste_chunk_size` of the canvas).
        parse_chunk: the frames per ParseNet forward (``None`` sizes it
            from the free VRAM).
        face_frame: the canvas row each face composites into - ``None``
            (the default) keeps the identity contract of the shipped path,
            face ``i`` -> row ``i``.  With the map the faces and the
            affines are **face-indexed** while the background and the
            returned IMAGE rows stay **frame-indexed** (``num_frames``
            rows), and the groups are built **frame-wise** - a group never
            splits a frame's faces (the t3 contract; the fix-4 chunk
            composites per face, the grouping stays).
        num_frames: the number of canvas rows - ``None`` derives it (the
            background rows with the map, the face count without).  When
            given it has to match the background stack.
        stage_bar: the optional paste-stage progress bar - ticked with the
            row count of every completed group, after its convert-out
            (fix 2.3, 2026-10-30; the owner's fix 4, 2026-10-31, extends
            the tick to the multi-face groups - the multiface stage's
            loop ticks only its fallback rows, so the batched rows tick
            here).  ``None`` (the default) ticks nothing.

    Returns:
        The ``(N, canvas_h, canvas_w, 3)`` float32 **CPU** tensor - the
        ComfyUI IMAGE rows of the eligible canvas rows.
    """
    faces = np.asarray(faces_bgr)
    background = np.asarray(background_bgr)
    num_faces = int(faces.shape[0])
    if faces.ndim != 4 or faces.shape[1:3] != (PARSE_FACE_SIZE, PARSE_FACE_SIZE):
        raise ValueError(
            f"paste_faces_batch_chunked expects (N, {PARSE_FACE_SIZE}, "
            f"{PARSE_FACE_SIZE}, 3) faces, got {tuple(faces.shape)}"
        )
    if len(inverse_affines) != num_faces:
        raise ValueError(
            f"paste_faces_batch_chunked: {len(inverse_affines)} inverse "
            f"affines for {num_faces} faces"
        )
    rows: list[int] | None = None
    if face_frame is None:
        if num_frames is not None and int(num_frames) != num_faces:
            raise ValueError(
                f"paste_faces_batch_chunked: num_frames={int(num_frames)} "
                f"without a face_frame map pins one row per face ({num_faces})"
            )
        num_rows = num_faces
    else:
        rows = [int(value) for value in face_frame]
        if len(rows) != num_faces:
            raise ValueError(
                f"paste_faces_batch_chunked: {len(rows)} face_frame entries "
                f"for {num_faces} faces"
            )
        num_rows = (
            int(num_frames) if num_frames is not None else int(background.shape[0])
        )
    if background.ndim != 4 or background.shape[0] != num_rows:
        if rows is None:
            raise ValueError(
                f"paste_faces_batch_chunked: {tuple(background.shape)} background "
                f"frames for {num_rows} faces"
            )
        raise ValueError(
            f"paste_faces_batch_chunked: {tuple(background.shape)} background "
            f"frames for {num_rows} canvas rows ({num_faces} faces)"
        )
    if rows is not None and any(not 0 <= row < num_rows for row in rows):
        raise ValueError(
            "paste_faces_batch_chunked: face_frame names a canvas row outside "
            f"the background stack of {num_rows} rows"
        )
    canvas_h, canvas_w = paste_canvas_hw(
        background.shape[1:3], upscale_factor, canvas_hw
    )
    if frame_chunk is None:
        frame_chunk = paste_chunk_size(canvas_h, canvas_w, device)
    frame_chunk = max(1, int(frame_chunk))
    budget = device_canvas_budget(device)
    out = torch.empty((num_rows, canvas_h, canvas_w, 3), dtype=torch.float32)
    reset_paste_transfer_stats()
    totals: dict[str, int] = {key: 0 for key in PASTE_TRANSFER_STATS}
    if rows is None:
        for start in range(0, num_rows, frame_chunk):
            # The cancel poll (C1): ComfyUI's Cancel button raises here - a
            # no-op outside ComfyUI (``src/keep_interrupt.py``).
            keep_interrupt.check_interrupted()
            stop = min(start + frame_chunk, num_rows)
            group_background = batched_bg_frames(
                background[start:stop], upscale_factor, bg_upscale_model,
                device, canvas_hw=canvas_hw, return_device=True,
            )
            canvas = paste_faces_batch(
                faces[start:stop],
                list(inverse_affines[start:stop]),
                group_background,
                device,
                parse_net,
                parse_chunk=parse_chunk,
                frame_chunk=frame_chunk,
                device_output=True,
            )
            # the group's paste call resets the counters - the chunked run
            # sums the additive ones; ``frames`` and ``canvas_bytes`` are
            # per-call values the tail sets for the whole helper call.
            for key, value in PASTE_TRANSFER_STATS.items():
                if key not in ("frames", "canvas_bytes"):
                    totals[key] += value
            comfy_convert_out(canvas, out=out[start:stop], budget=budget)
            # the paste bar (fix 2.3, 2026-10-30): the group's rows are
            # done - converted out into the host buffer.  The multi-face
            # groups below tick the same way (the owner's fix 4,
            # 2026-10-31): the stage's loop ticks only its fallback rows.
            if stage_bar is not None:
                stage_bar.tick(stop - start)
    else:
        # The multi-face groups (the cancel + multi-face round, t3):
        # frame-wise - a group never splits a frame's faces (the t3
        # contract; the owner's fix 4 composites per face, the grouping
        # stays).  The rows of the group carry the canvas; the faces ride
        # the map, re-based to the group's first row.
        row_faces: list[list[int]] = [[] for _ in range(num_rows)]
        for face_index, row in enumerate(rows):
            row_faces[row].append(face_index)
        for start in range(0, num_rows, frame_chunk):
            keep_interrupt.check_interrupted()
            stop = min(start + frame_chunk, num_rows)
            face_indices = [
                index for row in range(start, stop) for index in row_faces[row]
            ]
            group_background = batched_bg_frames(
                background[start:stop], upscale_factor, bg_upscale_model,
                device, canvas_hw=canvas_hw, return_device=True,
            )
            canvas = paste_faces_batch(
                faces[face_indices],
                [inverse_affines[index] for index in face_indices],
                group_background,
                device,
                parse_net,
                parse_chunk=parse_chunk,
                frame_chunk=frame_chunk,
                device_output=True,
                face_frame=[rows[index] - start for index in face_indices],
            )
            for key, value in PASTE_TRANSFER_STATS.items():
                if key not in ("frames", "canvas_bytes"):
                    totals[key] += value
            comfy_convert_out(canvas, out=out[start:stop], budget=budget)
            # the paste bar (fix 2.3, extended to the multi-face groups by
            # the owner's fix 4, 2026-10-31): the group's rows are done -
            # converted out into the host buffer.
            if stage_bar is not None:
                stage_bar.tick(stop - start)
    totals["frames"] = num_rows
    totals["canvas_bytes"] = canvas_h * canvas_w * 3
    PASTE_TRANSFER_STATS.clear()
    PASTE_TRANSFER_STATS.update(totals)
    return out


# ---------------------------------------------------------------------------
# The gray flags of the frames
# ---------------------------------------------------------------------------

def _is_gray_diff_sum_numpy(frame_bgr_u8: np.ndarray) -> float:
    """The variance sum of one frame (the value the vendored ``is_gray`` compares).

    The vendored helper (``facelib/utils/misc.py:146-160``) measures the
    variance of the three channel differences and compares their mean with
    the threshold.  This is that value, computed with numpy: the int16
    differences and the float64 ``.var()`` are the ones of the vendored
    call (PIL only changes the way the channels are read).

    Args:
        frame_bgr_u8: the ``(H, W, 3)`` uint8 BGR frame.

    Returns:
        The variance sum (the float64 value behind the flag).
    """
    frame = np.asarray(frame_bgr_u8)
    blue = frame[..., 0].astype(np.int16)
    green = frame[..., 1].astype(np.int16)
    red = frame[..., 2].astype(np.int16)
    return float(
        (blue - green).var() + (green - red).var() + (red - blue).var()
    ) / 3.0


def _is_gray_batch_numpy(frames_bgr_u8: np.ndarray) -> np.ndarray:
    """The gray flags of the frames, computed on the host (numpy).

    Args:
        frames_bgr_u8: the ``(N, H, W, 3)`` uint8 BGR frames.

    Returns:
        ``(N,)`` bool - the flag of every frame.
    """
    frames = np.asarray(frames_bgr_u8)
    flags = np.zeros(frames.shape[0], dtype=bool)
    for index in range(frames.shape[0]):
        flags[index] = _is_gray_diff_sum_numpy(frames[index]) <= GRAY_THRESHOLD
    return flags


def _is_gray_diff_sums_device(
    frames_bgr_u8: np.ndarray, device: torch.device
) -> np.ndarray:
    """The variance sum of every frame, computed on the device (float64).

    The three channel differences are the vendored int16 ones (the channel
    values are 0..255, so the difference never overflows) and their
    variances are accumulated in **float64**, like the numpy ``.var()`` of
    the host path.  The device reduction order differs from numpy's pairwise
    summation, so the two values can differ in the last bits (the measured
    drift is below ``1e-9`` on a 720p frame).

    Args:
        frames_bgr_u8: the ``(N, H, W, 3)`` uint8 BGR frames.
        device: the CUDA device.

    Returns:
        ``(N,)`` float64 - the variance sum of every frame.
    """
    frames = np.asarray(frames_bgr_u8)
    num_frames = int(frames.shape[0])
    if num_frames == 0:
        return np.zeros(0, dtype=np.float64)
    host = np.ascontiguousarray(frames)
    if not host.flags.writeable:
        host = host.copy()
    stack = torch.from_numpy(host).to(device)  # the (N, H, W, 3) uint8 frames
    pixels = max(1, int(frames.shape[1]) * int(frames.shape[2]))
    chunk = max(1, GRAY_CHUNK_PIXELS // pixels)
    diff_sums = torch.empty(num_frames, dtype=torch.float64, device=device)
    for start in range(0, num_frames, chunk):
        stop = min(start + chunk, num_frames)
        block = stack[start:stop].permute(0, 3, 1, 2)  # the (n, 3, H, W) uint8
        variances = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            difference = (
                block[:, first].to(torch.int16) - block[:, second].to(torch.int16)
            )
            variances.append(
                difference.to(torch.float64).var(dim=(1, 2), correction=0)
            )
            del difference
        diff_sums[start:stop] = (
            variances[0] + variances[1] + variances[2]
        ) / 3.0
    values = diff_sums.cpu().numpy()
    del stack, diff_sums
    return values


def _is_gray_batch_device(
    frames_bgr_u8: np.ndarray, device: torch.device
) -> np.ndarray:
    """The gray flags of the frames, computed on the device (float64).

    The variances run on the device (:func:`_is_gray_diff_sums_device`); a
    frame whose result lands within :data:`GRAY_BORDERLINE_EPSILON` of the
    threshold falls back to the numpy computation, and only the numpy value
    decides it.  The flags are therefore **bit-exact** against the host path
    - a tolerance would be a flag that can flip.

    Args:
        frames_bgr_u8: the ``(N, H, W, 3)`` uint8 BGR frames.
        device: the CUDA device.

    Returns:
        ``(N,)`` bool - the flag of every frame.
    """
    frames = np.asarray(frames_bgr_u8)
    values = _is_gray_diff_sums_device(frames, device)
    flags = values <= GRAY_THRESHOLD
    # the borderline frames - the numpy value decides them.
    borderline = np.flatnonzero(
        np.abs(values - GRAY_THRESHOLD) <= GRAY_BORDERLINE_EPSILON
    )
    for index in borderline:
        flags[index] = _is_gray_diff_sum_numpy(frames[index]) <= GRAY_THRESHOLD
    return flags


def is_gray_batch(
    frames_bgr_u8: np.ndarray, device: torch.device | None = None
) -> np.ndarray:
    """The per-frame ``is_gray`` flags (``facelib/utils/misc.py:146-160``).

    The vendored helper measures the variance of the three channel
    differences of every frame - through PIL, so one round trip per frame.
    This form computes the very same three variances, frame by frame (the
    memory of one frame at a time): the flags are **bit-exact** against the
    vendored call, because the arrays, their dtype and their shape are the
    same.

    On a CUDA device the variances run **on the device** (the host pass was
    the largest host-side item of the paste stage - 41.02 ms of process CPU
    per frame, the paste-stage CPU optimization round, section 4.2).  The
    device computes the variances in float64 and falls back to the numpy
    computation for the frames whose result lands on the threshold, so the
    flags are the flags of the host path.

    Args:
        frames_bgr_u8: the ``(N, H, W, 3)`` uint8 BGR frames.
        device: the compute device - a CUDA device runs the variances on
            the device, a CPU device (and ``None``) keeps the host
            computation.

    Returns:
        ``(N,)`` bool - the flag of every frame.
    """
    frames = np.asarray(frames_bgr_u8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"is_gray_batch expects (N, H, W, 3), got {tuple(frames.shape)}"
        )
    if device is not None and torch.device(device).type == "cuda":
        return _is_gray_batch_device(frames, torch.device(device))
    return _is_gray_batch_numpy(frames)


# ---------------------------------------------------------------------------
# The background of the paste (design section 7.7, step 1)
# ---------------------------------------------------------------------------

def _tiled_scale_batch(
    model, frames_bgr_u8: np.ndarray, device: torch.device
) -> torch.Tensor:
    """The batched ``tiled_scale`` of the background frames.

    ``comfy.utils.tiled_scale`` loops over the images of the batch
    (``comfy/utils.py:1205-1206``) and processes every image on its own, so
    a batched call is **pixel-identical** to N per-frame calls (the model
    still sees B=1 tiles - the win is the removed per-frame round trips).
    ``output_device`` keeps the result on the compute device.

    The value scale is the one of the per-frame ``_run_upscaler``: the RGB
    float in 0..1 in, the clamped float out, quantized with the
    **truncating** uint8 cast of ``comfy_image_to_cv2`` and flipped back to
    BGR.  The result stays **on the device** - the resize of
    :func:`batched_bg_frames` (and with it the only download of the
    background) happens after it.
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
    return quantized[..., [2, 1, 0]].contiguous()


def _resize_bg_frames(
    background: np.ndarray | torch.Tensor,
    target_hw: tuple[int, int],
    device: torch.device,
    return_device: bool,
) -> np.ndarray | torch.Tensor:
    """One resize stage of the background (the frame target or the canvas).

    The device Lanczos (:func:`resize_lanczos_u8`) runs when the result
    stays on the device (a CUDA device, or the ``return_device`` staging -
    on the CPU it is bit-exact, ``tests/test_image_op_equivalence.py:292-299``);
    a host background keeps the per-frame cv2 loop (the einsum resample is
    slower than cv2 there).  The resize is per-frame independent, so the
    group loop of :func:`batched_bg_frames` is pixel-neutral by
    construction (the CUDA batch composition stays the design's disclosed
    owner-gated residual).
    """
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    on_device = device.type == "cuda"
    if background.shape[1] == target_h and background.shape[2] == target_w:
        return background
    if on_device or return_device:
        return resize_lanczos_u8(background, (target_w, target_h), device)
    return np.stack(
        [
            cv2.resize(
                background[index], (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
            )
            for index in range(background.shape[0])
        ]
    )


def batched_bg_frames(
    frames_bgr_u8: np.ndarray,
    upscale_factor: float,
    bg_upscale_model,
    device: torch.device,
    canvas_hw: tuple[int, int] | None = None,
    return_device: bool = False,
) -> np.ndarray | torch.Tensor:
    """The background of the paste, at the target size of the paste.

    Without a background model (the production default) the background is
    the frame itself; with one it is the batched ``tiled_scale`` calls of
    :func:`_tiled_scale_batch` over the VRAM-aware frame groups of
    :func:`bg_chunk_size` - each group is resized to its target **before
    the next group starts**, so the x4 intermediate dies with its group
    instead of materialising for the whole clip (2026-11-01: ~0.37 GiB a
    720p frame - two float32 stacks of the 4x size plus the uint8 one -
    OOMed a 16 GB card on the owner's 48-frame clip before the first
    paste; the model work is unchanged, ``tiled_scale`` still processes
    every image on its own).  The resize to the target size runs **on the
    device** (:func:`resize_lanczos_u8` - the owner-accepted GPU Lanczos,
    the tolerance-accepted route) when the device is CUDA; a CPU device
    keeps the per-frame cv2 loop (the einsum resample is slower than cv2
    there).  The background is downloaded once, after the last resize.

    Args:
        canvas_hw: the ``(height, width)`` of the paste canvas - the size of
            the image the vendored ``read_image`` produced (the frame
            upscaled for ``min(h, w) < 512``).  The vendored paste resizes
            the background to that canvas
            (``face_restoration_helper.py:365-370``); ``None`` keeps the
            raw frame size (the frames of 512 px and above, where the two
            sizes are the same).

    Returns:
        ``(N, h_up, w_up, 3)`` uint8 BGR - on the host, or on the device
        when ``return_device`` is set (the numpy frames are uploaded once;
        with a background model the batched ``tiled_scale`` result is
        already on the device).
    """
    frames = np.asarray(frames_bgr_u8)
    height, width = frames.shape[1:3]
    target_h, target_w = int(height * upscale_factor), int(width * upscale_factor)
    on_device = device.type == "cuda"
    if bg_upscale_model is None:
        background: np.ndarray | torch.Tensor = (
            torch.from_numpy(np.ascontiguousarray(frames)).to(device)
            if return_device else frames
        )
        background = _resize_bg_frames(
            background, (target_h, target_w), device, return_device
        )
    else:
        # The bg model runs in VRAM-aware frame groups (2026-11-01): one
        # ``tiled_scale`` call per group instead of one over the whole
        # batch, every group resized to its target before the next group
        # starts - the x4 intermediate dies with its group instead of
        # materialising for the whole clip.  The model work is unchanged
        # (``tiled_scale`` still processes every image on its own).
        frame_chunk = bg_chunk_size(height, width, bg_upscale_model.scale, device)
        groups: list[np.ndarray | torch.Tensor] = []
        for start in range(0, frames.shape[0], frame_chunk):
            # The cancel poll (C1) of the grouped loops - a no-op outside
            # ComfyUI (``src/keep_interrupt.py``).
            keep_interrupt.check_interrupted()
            stop = min(start + frame_chunk, frames.shape[0])
            group = _tiled_scale_batch(bg_upscale_model, frames[start:stop], device)
            if not on_device and not return_device:
                group = group.numpy()
            groups.append(
                _resize_bg_frames(group, (target_h, target_w), device, return_device)
            )
        if len(groups) == 1:
            background = groups[0]
        elif isinstance(groups[0], torch.Tensor):
            background = torch.cat(groups, dim=0)
        else:
            background = np.concatenate(groups, axis=0)
    if canvas_hw is not None:
        canvas_h, canvas_w = canvas_hw
        canvas_target_h = int(canvas_h * upscale_factor)
        canvas_target_w = int(canvas_w * upscale_factor)
        background = _resize_bg_frames(
            background, (canvas_target_h, canvas_target_w), device, return_device
        )
    if isinstance(background, torch.Tensor) and not return_device:
        background = background.cpu().numpy()
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


def _quantize_and_clip(
    blended: torch.Tensor, maximum: torch.Tensor | None = None
) -> torch.Tensor:
    """The quantizing tail of the vendored paste (``:497-507``).

    The vendored code picks its clip by the maximum of the blended image:
    a 0..255 image is clipped, an image above 255 is rescaled and an image
    below 1.001 is treated as a 0..1 float and multiplied by 255.  The
    blend of two uint8 images can only reach the first and the third case;
    the second one is kept for the fidelity of the branch.

    Args:
        blended: the blended image - ``(N, C, H, W)`` of the full canvas or
            ``(C, H, W)`` of one ROI.
        maximum: the frame-global maximum of the blend, when it is known
            already (the ROI paste computes it over the whole frame, not
            over its ROI).  ``None`` measures it on the given tensor.
    """
    if maximum is None:
        maximum = (
            blended.amax(dim=(1, 2, 3), keepdim=True)
            if blended.ndim == 4
            else blended.amax()
        )
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
    use_roi: bool = True,
    device_output: bool = False,
    face_frame: Sequence[int] | None = None,
    num_frames: int | None = None,
    stage_bar=None,
) -> list[np.ndarray] | torch.Tensor:
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
        device_output: keep the output stack on the device - return the
            ``(N, h_up, w_up, 3)`` uint8 **tensor** instead of the ``N``
            host arrays.  Nothing is downloaded or assembled on the host;
            the bytes that would have travelled are counted in
            ``roi_staged_bytes``.  Default ``False`` keeps the host output
            of the shipped pins byte-for-byte.
        face_frame: the canvas row each face composites into - ``None``
            (the default) keeps the identity contract of the shipped path,
            face ``i`` -> row ``i``, and the body below runs unchanged.
            With the map the faces, the affines and the ROIs are
            **face-indexed** while the background, the canvas and the
            output stay **frame-indexed** (``num_frames`` rows): every
            face runs the per-face ROI tail of :func:`_paste_chunk_faces`
            into its row (the owner's fix 4, 2026-10-31 - the t3
            sequential per-row accumulator is replaced by it, so
            overlapping ROIs no longer see an earlier face's blend).
            The multi-face chunk is the ROI one (``use_roi`` has to stay
            set); the chunk boundaries stay frame-wise, so a group never
            splits a frame's faces.
        num_frames: the number of canvas rows - ``None`` derives it (the
            background rows with the map, the face count without).  When
            given it has to match the background stack.
        stage_bar: the optional paste-stage progress bar - ticked with the
            row count of every completed chunk, after the chunk body (the
            spec's bar threading realized as the signature the spec never
            spelled out; 2026-10-30).  The owner's fix 4 (2026-10-31)
            extends the tick to the multi-face loop - the multiface
            stage's own loop ticks only its fallback rows, so the batched
            rows tick here.  ``None`` (the default) ticks nothing.

    Returns:
        The pasted frames as uint8 BGR arrays - one per canvas row - or
        one uint8 BGR device tensor when ``device_output`` is set.
    """
    faces = np.asarray(faces_bgr)
    background = (
        background_bgr if torch.is_tensor(background_bgr)
        else np.asarray(background_bgr)
    )
    num_faces = int(faces.shape[0])
    if faces.ndim != 4 or faces.shape[1:3] != (PARSE_FACE_SIZE, PARSE_FACE_SIZE):
        # the vendored paste resizes the face to the ParseNet size and the
        # parse mask back to the face size - the batched path keeps the
        # production size (512x512) and leaves the other sizes to the
        # vendored paste.
        raise ValueError(
            f"paste_faces_batch expects (N, {PARSE_FACE_SIZE}, {PARSE_FACE_SIZE}, 3) "
            f"faces, got {tuple(faces.shape)}"
        )
    if len(inverse_affines) != num_faces:
        raise ValueError(
            f"paste_faces_batch: {len(inverse_affines)} inverse affines for "
            f"{num_faces} faces"
        )
    rows: list[int] | None = None
    if face_frame is None:
        if num_frames is not None and int(num_frames) != num_faces:
            raise ValueError(
                f"paste_faces_batch: num_frames={int(num_frames)} without a "
                f"face_frame map pins one row per face ({num_faces})"
            )
        num_rows = num_faces
    else:
        rows = [int(value) for value in face_frame]
        if len(rows) != num_faces:
            raise ValueError(
                f"paste_faces_batch: {len(rows)} face_frame entries for "
                f"{num_faces} faces"
            )
        if not use_roi:
            raise ValueError(
                "paste_faces_batch: the multi-face paste runs on the ROI "
                "chunk - the full-canvas chunk has no multi-face form"
            )
        num_rows = (
            int(num_frames) if num_frames is not None else int(background.shape[0])
        )
    if background.shape[0] != num_rows:
        if rows is None:
            raise ValueError(
                f"paste_faces_batch: {background.shape[0]} background frames for "
                f"{num_rows} faces"
            )
        raise ValueError(
            f"paste_faces_batch: {background.shape[0]} background frames for "
            f"{num_rows} canvas rows ({num_faces} faces)"
        )
    if rows is not None and any(not 0 <= row < num_rows for row in rows):
        raise ValueError(
            "paste_faces_batch: face_frame names a canvas row outside the "
            f"background stack of {num_rows} rows"
        )
    height, width = background.shape[1:3]
    out_size = (width, height)  # the cv2 dsize order
    if frame_chunk is None:
        frame_chunk = paste_chunk_size(height, width, device)
    reset_paste_transfer_stats()
    PASTE_TRANSFER_STATS["frames"] = num_rows

    # the device-output mode: one canvas, seeded from the background in
    # chunk-sized indexed copies; the ROI chunks scatter into it.  The
    # non-ROI chunk (``_paste_chunk``) assembles on the host and keeps the
    # list return, so the mode is the ROI path's.
    if device_output and use_roi:
        background_device = (
            torch.from_numpy(np.ascontiguousarray(background)).to(device)
            if not torch.is_tensor(background) else background
        )
        canvas = torch.empty(
            (num_rows, height, width, 3), dtype=torch.uint8, device=device
        )
    else:
        device_output = False
        background_device = background
    pasted: list[np.ndarray] = []
    if rows is None:
        for start in range(0, num_rows, max(1, int(frame_chunk))):
            keep_interrupt.check_interrupted()
            stop = min(start + max(1, int(frame_chunk)), num_rows)
            chunk = _paste_chunk_roi if use_roi else _paste_chunk
            if device_output:
                canvas[start:stop] = background_device[start:stop].to(device)
                chunk(
                    faces[start:stop],
                    list(inverse_affines[start:stop]),
                    background_device[start:stop],
                    device,
                    parse_net,
                    out_size,
                    parse_chunk,
                    canvas=canvas,
                    frame_offset=start,
                )
            else:
                pasted.extend(
                    chunk(
                        faces[start:stop],
                        list(inverse_affines[start:stop]),
                        background[start:stop],
                        device,
                        parse_net,
                        out_size,
                        parse_chunk,
                    )
                )
            # the paste bar (the spec's fix-1.2 threading, 2026-10-30): the
            # chunk's rows are pasted - host rows appended, canvas rows
            # scattered.  The processor's loop ticks only its fallback rows,
            # so the eligible rows tick here, per completed chunk.
            if stage_bar is not None:
                stage_bar.tick(stop - start)
        return canvas if device_output else pasted

    # The multi-face mode (the cancel + multi-face round, t3): frame-wise
    # groups under the same canvas-byte budget - a group never splits a
    # frame's faces (the t3 contract; the owner's fix 4 composites per
    # face, the grouping stays).  The rows of the group carry the canvas;
    # the faces ride the map, re-based to the group's first row.
    row_faces: list[list[int]] = [[] for _ in range(num_rows)]
    for face_index, row in enumerate(rows):
        row_faces[row].append(face_index)
    for start in range(0, num_rows, max(1, int(frame_chunk))):
        keep_interrupt.check_interrupted()
        stop = min(start + max(1, int(frame_chunk)), num_rows)
        face_indices = [
            index for row in range(start, stop) for index in row_faces[row]
        ]
        if device_output:
            canvas[start:stop] = background_device[start:stop].to(device)
            _paste_chunk_faces(
                faces[face_indices],
                [inverse_affines[index] for index in face_indices],
                [rows[index] - start for index in face_indices],
                background_device[start:stop],
                device,
                parse_net,
                out_size,
                parse_chunk,
                canvas=canvas,
                frame_offset=start,
            )
        else:
            pasted.extend(
                _paste_chunk_faces(
                    faces[face_indices],
                    [inverse_affines[index] for index in face_indices],
                    [rows[index] - start for index in face_indices],
                    background[start:stop],
                    device,
                    parse_net,
                    out_size,
                    parse_chunk,
                )
            )
        # the paste bar (fix 2.3, extended to the multi-face loop by the
        # owner's fix 4, 2026-10-31): the group's rows are pasted - canvas
        # rows scattered, host rows appended.  The multiface stage's loop
        # ticks only its fallback rows.
        if stage_bar is not None:
            stage_bar.tick(stop - start)
    return canvas if device_output else pasted


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


# ---------------------------------------------------------------------------
# The ROI paste (the round of the paste-stage CPU optimization)
# ---------------------------------------------------------------------------

def opencv_warp_inverse(matrix: np.ndarray) -> np.ndarray:
    """The sampling matrix of ``cv2.warpAffine`` (its own double inversion).

    ``cv::warpAffine`` inverts the matrix itself (``imgwarp.cpp``) and its
    operation order differs from ``cv2.invertAffineTransform`` in the last
    bits - and those last bits are exactly where the knife-edge pixels of
    the 1/32 grid live.  The device warp therefore uses **this** matrix.
    """
    m = np.asarray(matrix, dtype=np.float64).reshape(6)
    det = m[0] * m[4] - m[1] * m[3]
    det = 1.0 / det if det != 0 else 0.0
    a11 = m[4] * det
    a22 = m[0] * det
    n0, n1 = a11, m[1] * -det
    n3, n4 = m[3] * -det, a22
    b1 = -n0 * m[2] - n1 * m[5]
    b2 = -n3 * m[2] - n4 * m[5]
    return np.array([[n0, n1, b1], [n3, n4, b2]], dtype=np.float64)


def warp_fixed_point_terms(
    matrix: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The per-column / per-row 1/1024-unit terms of cv2's warpAffine.

    ``cv::warpAffine`` pre-scales the matrix to 1/1024 units
    (``saturate_cast<int>`` - a ``cvRound``, i.e. round-half-to-even) and
    splits the coordinate into a per-column and a per-row part; the row
    part carries the ``+16`` half-step bias that turns the blockline's
    arithmetic shift into a rounding to the 1/32 grid.
    """
    inv = opencv_warp_inverse(matrix)
    xs = np.arange(int(width), dtype=np.float64)
    ys = np.arange(int(height), dtype=np.float64)
    col_x = np.rint(inv[0, 0] * xs * CV_WARP_AB_SCALE)
    col_y = np.rint(inv[1, 0] * xs * CV_WARP_AB_SCALE)
    row_x = np.rint((inv[0, 1] * ys + inv[0, 2]) * CV_WARP_AB_SCALE) + CV_WARP_ROUND_DELTA
    row_y = np.rint((inv[1, 1] * ys + inv[1, 2]) * CV_WARP_AB_SCALE) + CV_WARP_ROUND_DELTA
    return (
        col_x.astype(np.int64),
        col_y.astype(np.int64),
        row_x.astype(np.int64),
        row_y.astype(np.int64),
    )


def face_roi_box(
    inverse_affine: np.ndarray, canvas_hw: tuple[int, int]
) -> tuple[int, int, int, int]:
    """The ROI of the transformed face (the bbox of its source footprint).

    The ROI covers the transformed source footprint ``[-1, 512]`` - the
    ``-1`` covers the bilinear neighbour cv2 taps at the source's border,
    so the ROI holds every pixel the full-canvas warp can give a non-zero
    value.  The parse mask's support (``[9, 503]``) lies inside it, so the
    blend outside the ROI is the untouched background.

    Returns:
        ``(x0, y0, x1, y1)`` - the clipped box in canvas coordinates.
    """
    # the paste's "inverse affine" already maps the face onto the canvas
    # (it is the inverse of the crop affine) - ``cv2.warpAffine`` samples
    # the face with its own inverse.
    forward = np.asarray(inverse_affine, dtype=np.float64)
    height, width = int(canvas_hw[0]), int(canvas_hw[1])
    xs: list[float] = []
    ys: list[float] = []
    for source_x, source_y in ((-1.0, -1.0), (-1.0, 512.0), (512.0, -1.0), (512.0, 512.0)):
        xs.append(forward[0, 0] * source_x + forward[0, 1] * source_y + forward[0, 2])
        ys.append(forward[1, 0] * source_x + forward[1, 1] * source_y + forward[1, 2])
    x0 = max(0, int(np.floor(min(xs))) - ROI_MARGIN)
    y0 = max(0, int(np.floor(min(ys))) - ROI_MARGIN)
    x1 = min(width, int(np.ceil(max(xs))) + ROI_MARGIN)
    y1 = min(height, int(np.ceil(max(ys))) + ROI_MARGIN)
    return x0, y0, max(x0, x1), max(y0, y1)


def warp_face_roi(
    face_u8: torch.Tensor,
    terms: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    roi: tuple[int, int, int, int],
) -> torch.Tensor:
    """The ROI face warp - cv2's fixed-point bilinear, in int32 on the device.

    The warp reproduces ``cv2.warpAffine`` **bit-exactly**: the 1/32
    sampling grid, the 5-bit weights, cv2's ``(+512) >> 10`` rounding and
    its ``BORDER_CONSTANT`` rule (an out-of-bounds tap contributes 0 and
    keeps its weight).  The coordinates are built the way cv2 builds them:
    the matrix is pre-scaled to 1/1024 units, the row term carries the
    ``+16`` half-step bias and the blockline shifts the 1/1024 value down
    to the 1/32 grid with an arithmetic shift.

    Args:
        face_u8: the ``(C, H, W)`` uint8 source face.
        terms: the 1/1024-unit terms of the warp
            (:func:`warp_fixed_point_terms`).
        roi: the ``(x0, y0, x1, y1)`` destination box (absolute canvas
            coordinates).

    Returns:
        ``(C, roi_h, roi_w)`` uint8.
    """
    x0, y0, x1, y1 = (int(value) for value in roi)
    col_x, col_y, row_x, row_y = terms
    device = face_u8.device
    channels, src_h, src_w = (int(value) for value in face_u8.shape)
    roi_h, roi_w = y1 - y0, x1 - x0

    # the four term uploads happen once, at the top - everything below is
    # pure device arithmetic (the upload-count pin freezes this shape)
    col_x_t = torch.from_numpy(col_x[x0:x1].astype(np.int32)).to(device)
    col_y_t = torch.from_numpy(col_y[x0:x1].astype(np.int32)).to(device)
    row_x_t = torch.from_numpy(row_x[y0:y1].astype(np.int32)).to(device)
    row_y_t = torch.from_numpy(row_y[y0:y1].astype(np.int32)).to(device)

    x_full = row_x_t[:, None] + col_x_t[None, :]
    y_full = row_y_t[:, None] + col_y_t[None, :]
    x32 = torch.div(x_full, CV_WARP_TAB_SIZE, rounding_mode="floor")
    y32 = torch.div(y_full, CV_WARP_TAB_SIZE, rounding_mode="floor")
    sx0 = torch.div(x32, CV_WARP_TAB_SIZE, rounding_mode="floor")
    sy0 = torch.div(y32, CV_WARP_TAB_SIZE, rounding_mode="floor")
    fx = x32 - CV_WARP_TAB_SIZE * sx0
    fy = y32 - CV_WARP_TAB_SIZE * sy0

    # `.reshape`, not `.view`: on the production input's strides both are a
    # zero-copy view, and `.reshape` also adapts to a differently-strided
    # view where the H/W merge is not stride-legal
    source = face_u8.reshape(channels, src_h * src_w)

    def tap(offset_x: int, offset_y: int) -> torch.Tensor:
        """One tap of the bilinear (0 outside the source, like cv2)."""
        sx = sx0 + offset_x
        sy = sy0 + offset_y
        inside = (sx >= 0) & (sx < src_w) & (sy >= 0) & (sy < src_h)
        flat = (sy.clamp(0, src_h - 1) * src_w + sx.clamp(0, src_w - 1)).reshape(-1)
        # the int32 cast stays on the ROI-sized gather - an up-front
        # ``source.int()`` would materialize a whole-face int32 copy
        values = source[:, flat].to(torch.int32).reshape(channels, roi_h, roi_w)
        return values * inside.unsqueeze(0).to(torch.int32)

    w00 = ((CV_WARP_TAB_SIZE - fx) * (CV_WARP_TAB_SIZE - fy)).unsqueeze(0)
    w01 = (fx * (CV_WARP_TAB_SIZE - fy)).unsqueeze(0)
    w10 = ((CV_WARP_TAB_SIZE - fx) * fy).unsqueeze(0)
    w11 = (fx * fy).unsqueeze(0)
    total = (
        w00 * tap(0, 0)
        + w01 * tap(1, 0)
        + w10 * tap(0, 1)
        + w11 * tap(1, 1)
        + 512
    ) >> 10
    return total.clamp_(0, 255).to(torch.uint8)


def _background_max_outside(frame: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    """The maximum of the background outside the ROI (the four bands).

    The blend outside the ROI is the background itself, so the frame-global
    maximum of the vendored tail is the maximum of the ROI blend and this
    value.  It is only needed when the ROI blend is near black - the rare
    branch of the vendored tail.
    """
    x0, y0, x1, y1 = roi
    height, width = frame.shape[:2]
    maximum = 0.0
    if y0 > 0:
        maximum = max(maximum, float(frame[:y0].max()))
    if y1 < height:
        maximum = max(maximum, float(frame[y1:].max()))
    if x0 > 0:
        maximum = max(maximum, float(frame[y0:y1, :x0].max()))
    if x1 < width:
        maximum = max(maximum, float(frame[y0:y1, x1:].max()))
    return maximum


def _paste_frame_full_canvas(
    face: np.ndarray,
    inverse_affine: np.ndarray,
    warped_mask: torch.Tensor,
    background: np.ndarray,
    device: torch.device,
    out_size: tuple[int, int],
    return_device: bool = False,
) -> np.ndarray | torch.Tensor:
    """The full-canvas paste of one frame (the near-black fallback).

    The vendored tail multiplies the **whole canvas** by 255 when the
    blended frame is near black (``face_restoration_helper.py:504-505``) -
    a frame-global operation the ROI cannot reproduce.  Such a frame falls
    back to the full-canvas path, which is bit-exact by construction.

    Args:
        warped_mask: the **warped** parse mask of the frame - the full
            canvas ``(H, W)``, from the batched warp of the chunk.
        return_device: keep the ``(H, W, 3)`` uint8 result on the device
            (the device-output canvas mode) instead of downloading it.
    """
    warped_face = cv2.warpAffine(face, inverse_affine, out_size)
    warped_face_t = (
        torch.from_numpy(np.ascontiguousarray(warped_face))
        .to(device)
        .permute(2, 0, 1)
        .to(torch.float32)
    )
    background_t = (
        torch.from_numpy(np.ascontiguousarray(background))
        .to(device)
        .permute(2, 0, 1)
        .to(torch.float32)
    )
    blended = warped_mask * warped_face_t + (1.0 - warped_mask) * background_t
    pasted = _quantize_and_clip(blended).round_().to(torch.uint8)
    if return_device:
        return pasted.permute(1, 2, 0).contiguous()
    return pasted.permute(1, 2, 0).contiguous().cpu().numpy()


def _paste_frame_full_canvas_multi(
    faces: Sequence[np.ndarray],
    inverse_affines: Sequence[np.ndarray],
    warped_masks: Sequence[torch.Tensor],
    background: np.ndarray,
    device: torch.device,
    out_size: tuple[int, int],
    return_device: bool = False,
) -> np.ndarray | torch.Tensor:
    """The full-canvas paste of one row's faces (the near-black fallback).

    The multi-face form of :func:`_paste_frame_full_canvas`: a row whose
    **whole** frame is near black takes the vendored tail's full-canvas
    fallback, which re-blends **all** of the row's faces sequentially at
    the full canvas - the same accumulation the vendored paste runs
    (``face_restoration_helper.py:495``) - and applies the vendored x255
    tail (``:497-507``) once, over the final composite.  With one face the
    loop degenerates to the exact expression of
    :func:`_paste_frame_full_canvas`.

    Args:
        faces: the row's uint8 BGR restored faces, in detection order.
        inverse_affines: the row's paste warps (face -> canvas).
        warped_masks: the row's **full-canvas** warped parse masks - the
            ``(H, W)`` floats of the batched warp of the chunk.
        background: the row's ``(H, W, 3)`` uint8 BGR background.
        return_device: keep the ``(H, W, 3)`` uint8 result on the device
            (the device-output canvas mode) instead of downloading it.
    """
    background_t = (
        torch.from_numpy(np.ascontiguousarray(background))
        .to(device)
        .permute(2, 0, 1)
        .to(torch.float32)
    )
    composite = background_t
    for face, inverse_affine, warped_mask in zip(faces, inverse_affines, warped_masks):
        warped_face = cv2.warpAffine(face, inverse_affine, out_size)
        warped_face_t = (
            torch.from_numpy(np.ascontiguousarray(warped_face))
            .to(device)
            .permute(2, 0, 1)
            .to(torch.float32)
        )
        composite = warped_mask * warped_face_t + (1.0 - warped_mask) * composite
    pasted = _quantize_and_clip(composite).round_().to(torch.uint8)
    if return_device:
        return pasted.permute(1, 2, 0).contiguous()
    return pasted.permute(1, 2, 0).contiguous().cpu().numpy()


def _compute_roi_theta(
    inverse_affine: np.ndarray,
    roi: tuple[int, int, int, int],
    src_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """The ``affine_grid`` theta that samples one face's canvas ROI.

    The owner's fix 4 (2026-10-31): the ROI paste warps each restored face
    straight into its canvas ROI - no full-canvas intermediate.  The theta
    composes the paste's inverse affine (face -> canvas) with the ROI
    window: ``cv2.invertAffineTransform`` gives the forward affine (canvas
    -> face), the ROI's top-left maps to the grid's ``[-1, 1]`` corners and
    the scale normalizes the ROI extent against the ``512 x 512`` source
    face (the hardcoded ``src_hw`` is the production face size).

    Returns:
        The ``(1, 2, 3)`` float32 theta of :func:`torch.nn.functional.affine_grid`.
    """
    x0, y0, x1, y1 = roi
    roi_w, roi_h = max(1, x1 - x0), max(1, y1 - y0)
    src_h, src_w = src_hw

    inv = cv2.invertAffineTransform(np.asarray(inverse_affine, dtype=np.float64))
    ax, ay, t = inv[:, 0], inv[:, 1], inv[:, 2]

    T0 = ax[0] * x0 + ay[0] * y0 + t[0]
    T1 = ax[1] * x0 + ay[1] * y0 + t[1]

    theta = np.zeros((1, 2, 3), dtype=np.float32)
    theta[0, 0, 0] = ax[0] * roi_w / src_w
    theta[0, 0, 1] = ay[0] * roi_h / src_w
    theta[0, 0, 2] = (ax[0] * (roi_w - 1) + ay[0] * (roi_h - 1) + 2.0 * T0 - (src_w - 1)) / src_w

    theta[0, 1, 0] = ax[1] * roi_w / src_h
    theta[0, 1, 1] = ay[1] * roi_h / src_h
    theta[0, 1, 2] = (ax[1] * (roi_w - 1) + ay[1] * (roi_h - 1) + 2.0 * T1 - (src_h - 1)) / src_h

    return torch.from_numpy(theta).to(device=device)


def _paste_chunk_roi(
    faces: np.ndarray,
    inverse_affines: list[np.ndarray],
    background: np.ndarray | torch.Tensor,
    device: torch.device,
    parse_net,
    out_size: tuple[int, int],
    parse_chunk: int | None,
    canvas: torch.Tensor | None = None,
    frame_offset: int = 0,
) -> list[np.ndarray] | None:
    """One chunk of the ROI paste, per-face (the owner's fix 4, 2026-10-31).

    The chunk warps each restored face **straight into its canvas ROI** -
    one :func:`_compute_roi_theta` theta, one ``F.affine_grid`` grid and
    one ``F.grid_sample`` pair (the face and its parse mask) per face -
    then blends ROI-local: ``mask * face + (1 - mask) * background`` over
    the ROI patch of the canvas and scatters the quantized patch back.
    The fix-2.1 full-canvas batched warp (:func:`~.keep_gpu_ops.warp_affine_batch`
    over the whole canvas) is replaced by it; ``parse_soft_masks_batched``
    stays the one batched op of the chunk.

    The pixel change is owner-authorized (the owner's ``keep-fix4.txt``):
    the ROI-local ``grid_sample`` replaces both the cv2 fixed-point warp and
    the fix-2.1 full-canvas one, and the blend quantizes with the plain
    ``round_().clamp_(0, 255)`` tail - the vendored near-black 255-factor
    scaling (:func:`_quantize_and_clip`) is gone from this path (the
    helper itself stays live for :func:`_paste_chunk` and the full-canvas
    fallbacks).  Outside a warped face's ROI the canvas keeps the seeded
    background bytes exactly.

    Transfer counters (the design's D3): ``face_h2d_bytes`` counts the
    face upload and ``roi_frames`` counts the chunk's faces; the ROI patch
    staging keys (``h2d_bytes``, ``d2h_bytes``, ``roi_staged_bytes``,
    ``fallback_frames``, ``canvas_bytes``) stay 0 from here.

    With a ``canvas`` (the device-output mode) the quantized ROI patches
    scatter into ``canvas[frame_offset + index]`` and the return stays
    ``None``; without it the chunk clones the background, pastes and
    returns the pasted rows as host arrays.
    """
    num_faces = len(faces)
    faces_u8 = torch.from_numpy(np.ascontiguousarray(faces)).to(device, non_blocking=True)
    PASTE_TRANSFER_STATS["face_h2d_bytes"] += faces_u8.numel() * faces_u8.element_size()
    faces_bchw = faces_u8.permute(0, 3, 1, 2)  # (N, 3, 512, 512) BGR

    # 1. the parse mask of the faces
    masks = parse_soft_masks_batched(parse_net, faces_bchw, device, parse_chunk)

    out_w, out_h = out_size
    if canvas is not None:
        # the device-output mode: the canvas rows are already seeded with
        # the background - the ROI patches blend in place, no download
        target_canvas = canvas
        canvas_offset = frame_offset
    else:
        bg_t = background if torch.is_tensor(background) else torch.from_numpy(np.ascontiguousarray(background)).to(device)
        target_canvas = bg_t.clone()
        canvas_offset = 0

    # 2. per face: the ROI warp of the face and of its parse mask, the
    # ROI-local blend and the scatter back into the canvas
    for index in range(num_faces):
        inv_aff = inverse_affines[index]
        if inv_aff is None:
            continue

        x0, y0, x1, y1 = face_roi_box(inv_aff, (out_h, out_w))
        roi_w, roi_h = x1 - x0, y1 - y0
        if roi_w <= 0 or roi_h <= 0:
            continue

        theta = _compute_roi_theta(inv_aff, (x0, y0, x1, y1), (512, 512), device)
        grid = F.affine_grid(theta, (1, 3, roi_h, roi_w), align_corners=False)

        face_roi = F.grid_sample(faces_bchw[index : index + 1].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        mask_roi = F.grid_sample(masks[index : index + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        frame_idx = canvas_offset + index
        bg_patch = target_canvas[frame_idx, y0:y1, x0:x1].permute(2, 0, 1).unsqueeze(0).float()
        blended = mask_roi * face_roi + (1.0 - mask_roi) * bg_patch
        target_canvas[frame_idx, y0:y1, x0:x1] = blended.squeeze(0).permute(1, 2, 0).round_().clamp_(0, 255).to(torch.uint8)

    PASTE_TRANSFER_STATS["roi_frames"] += num_faces
    if canvas is not None:
        return None
    return list(target_canvas.cpu().numpy())


def _paste_chunk_faces(
    faces: np.ndarray,
    inverse_affines: list[np.ndarray],
    face_rows: Sequence[int],
    background: np.ndarray | torch.Tensor,
    device: torch.device,
    parse_net,
    out_size: tuple[int, int],
    parse_chunk: int | None,
    canvas: torch.Tensor | None = None,
    frame_offset: int = 0,
) -> list[np.ndarray]:
    """One frame-wise chunk of the multi-face ROI paste (the owner's fix 4, 2026-10-31).

    The multi-face form of :func:`_paste_chunk_roi`: the faces, the
    affines and the parse masks are **face-indexed** (the ParseNet forward
    is one batched call over the chunk's faces), the background, the
    canvas and the output stay **frame-indexed** - ``face_rows[i]`` is the
    chunk-local row face ``i`` composites into.  Every face runs the same
    per-face ROI tail as the single-face chunk - one
    :func:`_compute_roi_theta` theta, one ``F.affine_grid`` grid, one
    ``F.grid_sample`` pair, the ROI-local blend and the scatter - only the
    target row is re-based through ``face_rows``.

    The pixel change is owner-authorized (the owner's ``keep-fix4.txt``):
    the t3 sequential per-row accumulator (each face blended over the
    row's running composite, the vendored op order) is replaced by the
    per-face ROI blend against the **background** patch - overlapping ROIs
    no longer see an earlier face's blend, the later face's scatter wins
    per pixel - and the quantizing tail is the plain
    ``round_().clamp_(0, 255)`` (the near-black full-canvas fallback of
    the old body is gone with it; :func:`warp_face_roi` and
    :func:`_paste_frame_full_canvas_multi` lose their production call
    site and stay live for their op-level pins).

    Args:
        faces: the ``(F, 512, 512, 3)`` uint8 BGR restored faces of the
            chunk (face-indexed).
        face_rows: the ``F`` chunk-local canvas rows the faces composite
            into, in face order.
        background: the ``(num_rows, H, W, 3)`` uint8 BGR background rows
            of the chunk (frame-indexed; a device tensor in the
            device-output mode).
        canvas: the device-output canvas - the blended ROI patches
            scatter into ``canvas[frame_offset + row]`` and the return
            stays empty (the caller only extends the host-mode return).

    Returns:
        The pasted rows as uint8 BGR arrays, in row order (the host mode;
        an empty list in the device-output mode).
    """
    num_rows = int(background.shape[0])
    if int(faces.shape[0]) == 0:
        # the defensive empty chunk (the eligibility of the processor gives
        # every row at least one face): the rows pass through as their
        # background - the canvas rows are already seeded with it.
        if canvas is None:
            return [background[row].copy() for row in range(num_rows)]
        return []

    faces_u8 = torch.from_numpy(np.ascontiguousarray(faces)).to(device, non_blocking=True)
    PASTE_TRANSFER_STATS["face_h2d_bytes"] += faces_u8.numel() * faces_u8.element_size()
    faces_bchw = faces_u8.permute(0, 3, 1, 2)  # (F, 3, 512, 512) BGR

    # 1. the parse mask of the faces
    masks = parse_soft_masks_batched(parse_net, faces_bchw, device, parse_chunk)

    out_w, out_h = out_size
    if canvas is not None:
        # the device-output mode: the canvas rows are already seeded with
        # the background - the ROI patches blend in place, no download
        target_canvas = canvas
        canvas_offset = frame_offset
    else:
        bg_t = background if torch.is_tensor(background) else torch.from_numpy(np.ascontiguousarray(background)).to(device)
        target_canvas = bg_t.clone()
        canvas_offset = 0

    # 2. per face: the ROI warp of the face and of its parse mask, the
    # ROI-local blend and the scatter into the face's canvas row
    for face_idx, row in enumerate(face_rows):
        inv_aff = inverse_affines[face_idx]
        if inv_aff is None:
            continue

        x0, y0, x1, y1 = face_roi_box(inv_aff, (out_h, out_w))
        roi_w, roi_h = x1 - x0, y1 - y0
        if roi_w <= 0 or roi_h <= 0:
            continue

        theta = _compute_roi_theta(inv_aff, (x0, y0, x1, y1), (512, 512), device)
        grid = F.affine_grid(theta, (1, 3, roi_h, roi_w), align_corners=False)

        face_roi = F.grid_sample(faces_bchw[face_idx : face_idx + 1].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        mask_roi = F.grid_sample(masks[face_idx : face_idx + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        frame_idx = canvas_offset + int(row)
        bg_patch = target_canvas[frame_idx, y0:y1, x0:x1].permute(2, 0, 1).unsqueeze(0).float()
        blended = mask_roi * face_roi + (1.0 - mask_roi) * bg_patch
        target_canvas[frame_idx, y0:y1, x0:x1] = blended.squeeze(0).permute(1, 2, 0).round_().clamp_(0, 255).to(torch.uint8)

    PASTE_TRANSFER_STATS["roi_frames"] += num_rows
    if canvas is not None:
        return []
    return list(target_canvas.cpu().numpy())
