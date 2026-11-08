"""The face processor of the KEEP pack (P3.2).

The engine that detects, aligns, restores and pastes the faces of an image
or an image sequence.  It moved here from ``modules/keep_processor.py`` in
P3.2.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from collections.abc import Sequence

from .keep_console import capture_console_to_logger
from .keep_image_utils import (
    BgrFrameSource,
    bgr_u8_batch_to_comfy,
    bgr_u8_batch_to_normalized,
    comfy_batch_to_bgr_u8,
    comfy_image_to_cv2,
    cv2_to_comfy_image,
    resize_bgr_u8,
)
from .keep_gpu_ops import (
    crop_faces_batch,
    estimate_affines,
    inverse_affines,
    tensor2img_batch,
)
from . import keep_gpu_paste as paste_mod
from . import keep_interrupt
from .keep_gpu_paste import (
    PARSE_FACE_SIZE,
    _call_spandrel,
    _spandrel_supports_fp16,
    batched_bg_frames,
    is_gray_batch,
    paste_faces_batch,
)
from .keep_logger import logger
from .keep_progress import ProgressReporter
from .keep_stage_progress import StageBar

# The vendored basicsr / facelib helpers - imported through the pack-private
# namespace (Round 21, :mod:`src.keep_vendored`): the old dual-mode pattern
# (package-relative first, flat fallback second) bound the shared top-level
# ``modules`` name, so on a ``sys.path`` where another root ships a
# ``modules/`` package, first-importer-wins decided which tree the pack
# binds.  Under ``_keep_vendor`` the trees are instantiated once per process,
# anchored to the pack's own files, in every import mode.  A broken vendored
# dependency still fails loudly (P2.7).
#
# P2.1: the vendored ``basicsr.utils.video_util`` module is deleted (it ran
# a pip install of ``ffmpeg-python`` at import time).  Its video helpers are
# not importable any more and the dead video path of the processor that used
# them was removed with it (P4.7).  The image / image-sequence paths are
# unaffected.
from .keep_vendored import vendor

_img_util = vendor("basicsr.utils.img_util")
img2tensor = _img_util.img2tensor
tensor2img = _img_util.tensor2img
is_gray = vendor("facelib.utils.misc").is_gray


def _soft_empty_cache() -> bool:
    """Release the cached VRAM through ComfyUI (P4.6).

    ``torch.cuda.empty_cache()`` is **unconditional**: it releases the cached
    blocks of every device, it fights with the VRAM arbitration of ComfyUI and
    it crashes on a CPU-only machine (the B7 breakage of the plan).
    ``model_management.soft_empty_cache()`` only releases what ComfyUI does
    not need any more - and it is a no-op without a GPU.

    The import is **lazy** (function level): the pack must import without a
    ComfyUI server (the plan's §3 import rule).
    """
    try:
        from comfy import model_management  # lazy: src/ imports without ComfyUI
    except ImportError as exc:  # pragma: no cover - the nodes run inside ComfyUI
        logger.debug(f"the VRAM cache could not be released ({exc})")
        return False
    model_management.soft_empty_cache()
    return True


def interpolate_sequence(sequence):
    interpolated_sequence = np.copy(sequence)
    missing_indices = np.isnan(sequence)
    if np.any(missing_indices):
        valid_indices = ~missing_indices
        x = np.arange(len(sequence))
        interpolated_sequence[missing_indices] = np.interp(
            x[missing_indices], x[valid_indices], sequence[valid_indices])
    return interpolated_sequence


def _group_by_shape(frames: dict[int, np.ndarray],
                    indices) -> dict[tuple[int, int], list[int]]:
    """Group ``indices`` by the ``(height, width)`` of their detection image.

    The batched crop warps one source-shape group per call: all frames of a
    clip share ``(h, w)``, and the vendored ``read_image`` upscale maps the
    sub-512 frames to one second shape, so in practice exactly one group
    comes back.  The groups keep the insertion order of the first
    occurrence, and every member list keeps the order of ``indices``.
    """
    groups: dict[tuple[int, int], list[int]] = {}
    for index in indices:
        groups.setdefault(tuple(frames[index].shape[:2]), []).append(index)
    return groups


def _gaussian_smooth(values: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Smooth a ``(frames, values)`` array along the frame axis.

    The torch-native replacement of the pack's former ``gaussian_filter1d``
    smoothing (the temporal landmark filter): the kernel is the reference
    implementation's own formula (``exp(-x**2 / (2*sigma**2))`` normalized
    by its sum, with the radius ``int(truncate * sigma + 0.5)`` of the
    ``truncate=4.0`` default), the boundary extension is symmetric (the
    reference ``mode="reflect"``) and the correlation runs as a depthwise
    ``conv1d`` - one channel per value column.

    The whole chain runs in **float32** (round 6, Opt 3), same decision as
    :func:`bgr_u8_batch_to_normalized`: the float64 positions / kernel /
    signal doubled the transient bytes of the whole-clip ``(frames, 10)``
    landmark array for a one-ulp-at-the-landmark-magnitude difference.  The
    kernel still sums to one, so the landmarks move by at most
    ``2**-23 * magnitude`` (~1e-4 px at the 512 landmark scale) - orders
    below the warp tolerance the crops already carry, which is why the
    sequence golden fixtures were regenerated rather than re-toleranced in
    e5.  The output keeps the dtype of the input: float32 landmarks in,
    float32 landmarks out (the ``conv1d`` result is cast back exactly once);
    the measured difference against the scipy (float64) reference is
    ~1e-5 (``tests/test_image_op_equivalence.py``).

    There is **no** ``device`` parameter on purpose: the sole call site is a
    ``(frames, 10)`` host array, and a CPU float32 ``conv1d`` is already the
    cheaper of the two.
    """
    source = np.asarray(values)
    radius = int(4.0 * sigma + 0.5)
    positions = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(positions**2) / (2.0 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    # ``mode="reflect"`` of the reference == numpy ``mode="symmetric"``: the
    # edges are mirrored with the edge value repeated.
    padded = np.pad(source, ((radius, radius), (0, 0)), mode="symmetric")
    signal = torch.from_numpy(padded).to(torch.float32).T.unsqueeze(0)
    weight = kernel.repeat(signal.shape[1], 1, 1)
    smoothed = F.conv1d(signal, weight, groups=signal.shape[1])
    result = smoothed.squeeze(0).T.contiguous().numpy()
    return result.astype(source.dtype, copy=False)


def _gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    """A 2D image -> a 3-channel BGR image (cv2 GRAY2BGR, torch-native).

    ``torch.stack`` materializes the channels (contiguous and writable) - a
    stride-0 ``expand`` view would be read-only for the consumers downstream.
    """
    gray_t = torch.from_numpy(np.ascontiguousarray(gray))
    return torch.stack((gray_t, gray_t, gray_t), dim=-1).numpy()


def _core_background(background):
    """The background the paste core consumes: a lazy source or a stack.

    The chunked core (:func:`~.keep_gpu_paste.paste_faces_batch_chunked`)
    takes a :class:`~.keep_image_utils.BgrFrameSource` (the round-17 lazy
    path - the core converts the group's rows on demand) or a numpy stack
    (the materialized path).  A provider of the numpy side is unwrapped
    here - its array is already there, so nothing is copied.
    """
    if isinstance(background, BgrFrameSource):
        return background
    return background.materialize()


def _flush_fallback_group(final_batch,
                          group_rows: list[int],
                          output_frames_cv2: list,
                          fallback_indices: list[int]) -> None:
    """Round 19 (FIX 2): one consumed gray group's fallback frames -> the buffer.

    The chunks_mode tail used to stack **all** fallback frames of the clip
    and convert the whole stack in one ``bgr_u8_batch_to_comfy`` call - the
    chain peaks at about 10x the stack bytes (the stack 1x, the RGB gather
    1x, the float32 cast 4x, the ``/255`` 4x), so the peak scaled with the
    fallback count.  The fallback loop already consumes the frames per gray
    group (``_fallback_gray_groups`` - at most ``max_clip_length`` rows per
    group); this flush converts **that group** at the moment the loop
    advances past it, so at most one group's frames, its stack and its
    conversion transient are ever alive - independent of the fallback
    count.  The conversion is the same elementwise chain, so a row's bytes
    cannot move (the conversion is batch-invariant, pinned round 19).

    The write is the **indexed write** (``final_batch[rows] = value`` - the
    round-18 ``index_put_`` pattern of ``comfy_convert_out``'s ``out_rows``
    seam): in place, into the named rows.  The read-side ``final_batch[rows]``
    would be advanced indexing - a copy - and the rows would be lost.

    Every fallback row appends exactly one frame (and its row number) to
    ``output_frames_cv2`` / ``fallback_indices``, in clip order, and a gray
    group is a consecutive chunk of the fallback rows - so when the loop
    advances past a group, that group's frames are the **last**
    ``len(group_rows)`` entries of the lists.  The flush takes exactly those,
    writes them into the group's rows and trims them off the lists - the
    converted frames are freed and only the pending group stays collected.
    The lists keep the one-frame-per-fallback-row alignment the other
    output tails (the device canvas and the host) read.

    Args:
        final_batch: the whole-clip float32 IMAGE buffer (the eligible rows
            are already in it - the tail writes only the fallback rows).
        group_rows: the consumed group's row numbers, in clip order.
        output_frames_cv2: the collected fallback frames (the group's are
            the last ``len(group_rows)``); trimmed by the flush.
        fallback_indices: the row numbers of the collected frames; trimmed
            in step (the alignment of the other tails is kept).
    """
    count = len(group_rows)
    if not count:
        # the defensive empty group (no fallback row at all): nothing
        # collected, nothing to write.
        return
    pending = len(output_frames_cv2) - count
    if pending < 0 or len(fallback_indices) != len(output_frames_cv2):
        raise RuntimeError(
            f"the fallback bookkeeping drifted: {len(output_frames_cv2)} "
            f"frames for {len(fallback_indices)} indices "
            f"(the group holds {count} rows)")
    final_batch[group_rows] = bgr_u8_batch_to_comfy(
        np.stack(output_frames_cv2[pending:]))
    del output_frames_cv2[pending:]
    del fallback_indices[pending:]


class KEEPFaceProcessor:
    def __init__(self, model_pack):
        self.keep_net = model_pack.keep_net
        self.face_helper = model_pack.face_helper
        self.bg_upscale_model = model_pack.bg_upscale_model
        self.face_upscale_model = model_pack.face_upscale_model
        self.device = model_pack.device
        self.model_type_str = model_pack.model_type_str

    def _run_upscaler(self, model, cv2_image):
        """Helper to run a comfy-native upscaler on a cv2 image.

        keep-fix5 (2026-11-03): the forward runs in FP16 autocast on a CUDA
        device (the CPU gate exercises the FP32 path), and ``output_device``
        keeps the scaled float on the compute device - the former body
        downloaded the scaled intermediate to the host before the clamp.

        keep-fix5-implement2 (2026-11-03): the model call goes through
        :func:`~.keep_gpu_paste._call_spandrel` (the spandrel descriptor's
        ``size_requirements`` padding; the plan's Fix 1 diagnosis is
        ComfyUI-KEEP-wide) and the autocast is gated on
        :func:`~.keep_gpu_paste._spandrel_supports_fp16` (``supports_half``).
        No ``_spandrel_on_device`` wrap here - the plan is silent on the
        single-image leg and the pack loader owns the model's lifecycle
        (it pre-moves at load and offloads at run end).
        """
        from comfy.utils import tiled_scale  # lazy: the pack imports without ComfyUI

        if model is None:
            return cv2_image

        img_tensor = cv2_to_comfy_image(cv2_image).to(self.device)
        img_tensor_bchw = img_tensor.movedim(-1, -3)
        # keep-fix5-implement2 (2026-11-03): the FP16 autocast is gated on
        # the descriptor's supports_half (False on a non-CUDA device), and
        # the scaled float stays on the compute device.
        use_autocast = _spandrel_supports_fp16(model, self.device)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_autocast):
            s = tiled_scale(
                img_tensor_bchw,
                lambda a: _call_spandrel(model, a),
                tile_x=512,
                tile_y=512,
                overlap=64,
                upscale_amount=model.scale,
                output_device=self.device,
            )
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return comfy_image_to_cv2(s)

    def _draw_boxes_on_canvas(
        self,
        canvas: torch.Tensor,
        affines: Sequence[np.ndarray | None],
        canvas_hw: tuple[int, int],
        face_rows: Sequence[int] | None = None,
    ) -> None:
        """Draw the 2-px green face boxes onto the paste canvas, in place.

        The owner's fix 4 (2026-10-31): ``draw_box`` no longer disqualifies
        the batched paste - the box is drawn on the device canvas / chunk
        buffer after the paste, one box per face at the face's
        :func:`~.keep_gpu_paste.face_roi_box` (the same box geometry the
        vendored paste's border masks cover).

        Disclosed limitations of the fix: the boxes land on the
        **device-output** canvases only (the ``"full"`` uint8 canvas and
        the ``"chunked"`` float32 buffer) - the host branch draws none, so
        a CPU run with ``draw_box`` loses the boxes on the batched frames.
        On the float32 chunk buffer the uint8 color casts on assignment
        (the box lands at ``255.0`` in the 0..1 IMAGE scale; the ComfyUI
        consumers clamp after the ``*255`` round trip, so the rendered box
        is the same green).

        Args:
            canvas: the ``(N, H, W, 3)`` canvas tensor (uint8 or float32),
                modified in place.
            affines: the paste's inverse affines, face-indexed - ``None``
                skips the face (no box).
            canvas_hw: the ``(height, width)`` of the canvas.
            face_rows: the canvas row of every face - ``None`` keeps the
                one-face-per-row identity (face ``i`` -> row ``i``).
        """
        color = torch.tensor([0, 255, 0], dtype=torch.uint8, device=canvas.device)
        h, w = canvas_hw
        for idx, aff in enumerate(affines):
            if aff is None:
                continue
            row = idx if face_rows is None else face_rows[idx]
            x0, y0, x1, y1 = paste_mod.face_roi_box(aff, (h, w))
            if x1 <= x0 or y1 <= y0:
                continue
            canvas[row, y0 : min(y0 + 2, h), x0:x1] = color
            canvas[row, max(0, y1 - 2) : y1, x0:x1] = color
            canvas[row, y0:y1, x0 : min(x0 + 2, w)] = color
            canvas[row, y0:y1, max(0, x1 - 2) : x1] = color

    def _fallback_gray_groups(self,
                              frame_source: BgrFrameSource,
                              fallback_rows: list[int],
                              gray_flags: np.ndarray,
                              gray_device,
                              group_size: int):
        """The fallback rows' frames, in groups - with their gray flags.

        Round 17 (the final stage's RAM): the fallback rows are converted
        **once**, in groups of ``group_size`` rows, and the gray flags of a
        group are computed **with** the group - the whole-clip uint8 stack
        and the whole-clip gray pass are gone.  Only one group is alive at
        a time.

        The flags themselves do not move: ``is_gray_batch`` decides **per
        frame**, so a group is the same set of decisions as the whole clip
        (the device path's borderline band, ``GRAY_BORDERLINE_EPSILON``,
        keeps the device and the numpy values in agreement).  The eligible
        rows' flags are never computed - they were never read (the batched
        rows do not use them), so they stay ``False``.

        Args:
            frame_source: the lazy source of the clip's frames.
            fallback_rows: the rows the batched paste did not take - in
                clip order.
            gray_flags: the flag array of the clip - the group's flags are
                written into it.
            gray_device: the device of the gray pass - ``None`` keeps the
                host (numpy) computation.
            group_size: the rows per group (the processor's
                ``max_clip_length``).

        Yields:
            ``(rows, frames)`` - the group's row numbers and their uint8
            BGR frames ``(k, H, W, 3)``.  With no fallback row at all the
            one **empty** group is yielded: the gray pass then runs on the
            empty batch, so the processor's device contract holds in every
            configuration.
        """
        size = max(1, int(group_size))
        rows = [int(row) for row in fallback_rows]
        if not rows:
            # the defensive empty group: the gray pass runs once, on the
            # empty batch (a pinned shape), so the processor's device
            # contract holds even when every row is batched.  There is no
            # row to write a flag for.
            frames = frame_source.group([])
            is_gray_batch(frames, device=gray_device)
            yield [], frames
            return
        for start in range(0, len(rows), size):
            group = rows[start:start + size]
            frames = frame_source.group(group)
            gray_flags[group] = is_gray_batch(frames, device=gray_device)
            yield group, frames

    def _batched_paste(self,
                       input_frames_bgr: np.ndarray | BgrFrameSource,
                       restored_faces: np.ndarray,
                       affine_matrices: list,
                       avg_landmarks_sequence,
                       final_upscale_factor: float,
                       draw_box: bool,
                       gray_flags: np.ndarray,
                       canvas_hw: tuple[int, int] | None = None,
                       stage_bar=None) -> dict:
        """The batched paste of the default branch (design section 7.7).

        The frames the batched paste can handle - a valid affine, a
        restored face of the ParseNet size and an available parse net -
        are pasted on the device: the chunked ParseNet forward, the
        batched parse mask, the batched inverse warp and the batched
        blend.  Since the owner's fix 1.2 (2026-10-30) a grayscale frame
        is batch-eligible - the device blend handles it with the same
        tensor logic (``gray_flags`` stays a parameter; the fallback loop
        still reads it, the eligibility no longer does).  Since keep-fix5
        (2026-11-03) a configured face upsampler is batch-eligible too -
        the ``return {}`` abort is deleted; the upscaled faces ride the
        ROI chunks (:func:`upscale_faces_batch_gpu`).

        Since the owner's fix 4 (2026-10-31) ``draw_box`` no longer
        disqualifies the batch: in the device-output modes the boxes are
        drawn on the canvas / chunk buffer after the paste
        (:meth:`_draw_boxes_on_canvas` - the host branch draws none, so a
        CPU run with ``draw_box`` keeps no boxes on the batched frames;
        see the method's disclosed limitations).

        Every other frame keeps the vendored ``paste_faces_to_input_image``
        (design section 7.7, step 6): the face upsampler, the frames
        without a face and a face helper without a parse net.

        The background arrives as a
        :class:`~.keep_image_utils.BgrFrameSource` (the round-17 lazy path
        - the eligible rows are a **view** of the clip and the core
        converts the group's rows on demand) or as a numpy stack (the
        direct-call path of the tests and the tools).

        Args:
            stage_bar: the optional paste-stage progress bar - forwarded
                into the paste core, which ticks the eligible rows per
                completed chunk/group (the fallback rows are ticked by the
                processor's loop).

        Returns:
            ``{frame index: the pasted frame}`` - the frames that are not
            in the mapping take the vendored path.  A device-output mode
            replaces the mapped frames with one sentinel key:
            :data:`~.keep_gpu_paste.DEVICE_CANVAS_KEY` (the ``"full"``
            mode's one uint8 canvas - the mapped indices name its rows,
            in order) or
            :data:`~.keep_gpu_paste.DEVICE_OUTPUT_CHUNKS_KEY` (the
            ``"chunked"`` mode's float32 IMAGE rows - round 18: the
            **whole-clip** buffer the twin allocates and hands to the
            core, so the eligible rows land in their final rows and the
            mapped index is the buffer's own row).
        """
        parse_net = getattr(self.face_helper, "face_parse", None)
        if parse_net is None or not getattr(self.face_helper, "use_parse", False):
            return {}
        if avg_landmarks_sequence is None:
            return {}

        eligible = []
        for i in range(len(restored_faces)):
            if i >= len(affine_matrices) or affine_matrices[i] is None:
                continue
            if i >= len(avg_landmarks_sequence) or np.isnan(avg_landmarks_sequence[i]).any():
                continue
            if restored_faces[i].shape[:2] != (PARSE_FACE_SIZE, PARSE_FACE_SIZE):
                continue
            eligible.append(i)
        if not eligible:
            return {}

        # The inverse warps of the paste (the vendored ``get_inverse_affine``:
        # ``cv2.invertAffineTransform(affine) * upscale_factor``) and the
        # background of the batch - with a bg model the background runs one
        # frame at a time (:func:`_tiled_scale_and_resize_batch`, keep-fix5
        # 2026-11-03; the whole-clip x4 intermediate of 2026-11-01 OOMed
        # the card before the first paste).
        inverses = inverse_affines(
            [affine_matrices[i] for i in eligible], upscale_factor=final_upscale_factor)
        # The pass-through gates (round 4, D-R2.3.2): the paste core is
        # read-only on both stacks (it slices and uploads), so the identity
        # pass-through is bitwise the gather.  The faces pass through when
        # the batch covers every restored face in order; the background only
        # when every frame is eligible - the core enforces
        # ``background.shape[0] == num_rows`` (``num_rows = len(eligible)``
        # here), so a clip with a face-less or gray frame keeps the frame
        # gather.
        if eligible == list(range(len(restored_faces))):
            faces_bgr = restored_faces
        else:
            # Round 19 (FIX 1): the mixed case builds no gather copy - the
            # lazy selected-rows source serves the core's groups (a view
            # when the group is contiguous in stack space, a group-sized
            # copy otherwise), so the eligible-sized np.stack of the old
            # branch (~611 MB on the owner's mixed clip) is never alive.
            # The pass-through gate above keeps handing the raw stack
            # object for the all-eligible case (round 4, D-R2.3.2).
            faces_bgr = paste_mod.LazyFaceSource(restored_faces).select(eligible)
        # The background of the batch (round 17): the **provider** of the
        # rows - the lazy source (the paste stage's path) or the numpy stack
        # (the direct-call path of the tests and the tools).  The eligible
        # rows are a **view** of the source: the gather copy of the
        # eligible rows is gone, and the source converts the rows per group
        # inside the core.  The numpy path keeps its gather (the array is
        # already materialized there).
        background_provider = paste_mod.frame_provider(input_frames_bgr)
        # Round 18: the clip's row count - the row count of the final
        # buffer.  It is read **before** the select (the select narrows
        # the provider to the eligible rows), and it is the frame count
        # the eligible rows are indexed by: the restored-face stack is
        # frame-indexed (``num_input_frames`` rows - one row per input
        # frame), so an eligible index is a frame index.
        num_input_frames = int(background_provider.shape[0])
        if eligible != list(range(num_input_frames)):
            background_provider = background_provider.select(eligible)

        canvas_h, canvas_w = paste_mod.paste_canvas_hw(
            background_provider.shape[1:3], final_upscale_factor, canvas_hw)
        mode = paste_mod.device_output_mode(self.device, len(eligible), canvas_h, canvas_w)
        # keep-fix5 (2026-11-03): a background model never takes the full
        # mode - the whole-video device canvas would block before the
        # first model frame; the chunked groups keep the progress alive.
        if self.bg_upscale_model is not None and mode == "full":
            mode = "chunked"

        if mode == "full":
            background = batched_bg_frames(
                background_provider.materialize(), final_upscale_factor,
                self.bg_upscale_model,
                self.device, canvas_hw=canvas_hw, return_device=True)
            canvas = paste_faces_batch(
                faces_bgr=faces_bgr,
                inverse_affines=inverses,
                background_bgr=background,
                device=self.device,
                parse_net=parse_net,
                face_upscale_model=self.face_upscale_model,
                device_output=True,
                stage_bar=stage_bar)
            if draw_box:
                self._draw_boxes_on_canvas(canvas, inverses, (canvas_h, canvas_w))
            return {**dict(zip(eligible, list(canvas))), paste_mod.DEVICE_CANVAS_KEY: canvas}

        if mode == "chunked":
            # Round 18: the final whole-clip float32 buffer is allocated
            # **before** the paste and handed to the core - the eligible
            # rows land straight in it (the core writes them by index).
            # The core's own buffer would be a second whole-clip float32
            # buffer alive at the same time (the defect this round
            # removes).
            final_batch = torch.empty(
                (num_input_frames, canvas_h, canvas_w, 3), dtype=torch.float32)
            chunks = paste_mod.paste_faces_batch_chunked(
                faces_bgr=faces_bgr,
                inverse_affines=inverses,
                background_bgr=_core_background(background_provider),
                upscale_factor=final_upscale_factor,
                bg_upscale_model=self.bg_upscale_model,
                device=self.device,
                canvas_hw=canvas_hw,
                parse_net=parse_net,
                stage_bar=stage_bar,
                face_upscale_model=self.face_upscale_model,
                out=final_batch,
                # the core's canvas row r is the r-th eligible frame, so
                # the eligible rows land in their final rows.
                out_rows=eligible)
            if draw_box:
                # the boxes land on the **final** rows (the float32 IMAGE
                # rows) - the eligible-space row of a face is its final
                # row ``eligible[face]``.
                self._draw_boxes_on_canvas(
                    chunks, inverses, (canvas_h, canvas_w),
                    face_rows=list(eligible))
            # the mapped values are **views** into the final buffer (basic
            # integer indexing) - the tail drops them into a set, so no
            # second copy of the rows is ever alive.
            return {**{frame: chunks[frame] for frame in eligible},
                    paste_mod.DEVICE_OUTPUT_CHUNKS_KEY: chunks}

        background = batched_bg_frames(
            background_provider.materialize(), final_upscale_factor,
            self.bg_upscale_model,
            self.device, canvas_hw=canvas_hw)
        pasted = paste_faces_batch(
            faces_bgr=faces_bgr,
            inverse_affines=inverses,
            background_bgr=background,
            device=self.device,
            parse_net=parse_net,
            face_upscale_model=self.face_upscale_model,
            stage_bar=stage_bar)
        return dict(zip(eligible, pasted))

    def _batched_paste_multiface(self,
                                 input_frames_bgr: np.ndarray | BgrFrameSource,
                                 restored_faces: np.ndarray,
                                 face_affines: list,
                                 face_frames: list[int],
                                 face_landmarks: list,
                                 final_upscale_factor: float,
                                 draw_box: bool,
                                 gray_flags: np.ndarray,
                                 canvas_hw: tuple[int, int] | None = None,
                                 stage_bar=None) -> dict:
        """The batched paste of the multi-face branch (design section 3.4).

        The sibling of :meth:`_batched_paste` for the frames with
        **several** faces: the canvas rows, the background and the output
        stay **frame-indexed** while the faces, the inverses, the masks and
        the ROIs are **face-indexed**, joined by the ``face_frame`` map of
        ``src/keep_gpu_paste.py`` (the t3 paste core).

        The eligibility is **per frame, all-or-nothing**: a frame is
        device-batched iff it has at least one face and **every** face is
        eligible (an affine, no NaN landmarks, a restored face of the
        ParseNet size) - a frame with any ineligible face goes entirely to
        the vendored fallback with all its faces (mixing device ROIs and a
        vendored composite in one frame would need a float accurate
        hand-off).  A ``None`` affine or NaN landmarks never reach here in
        practice - the crop stage of the multi-face branch drops those
        faces (``_process_sequence_multiface``) - the checks are the
        contract of the design, kept like the sibling's.  Since the
        owner's fix 4 (2026-10-31) a grayscale frame is batch-eligible
        here too - the gray skip is gone from the eligibility
        (``gray_flags`` stays a parameter; the fallback loop still reads
        it), matching the sibling's fix 1.2.

        Since the owner's fix 4 (2026-10-31) ``draw_box`` no longer
        disqualifies the batch: in the device-output modes the boxes are
        drawn on the canvas / chunk buffer after the paste
        (:meth:`_draw_boxes_on_canvas` with the ``face_frame`` map - the
        host branch draws none, so a CPU run with ``draw_box`` keeps no
        boxes on the batched frames; see the method's disclosed
        limitations).

        Args:
            stage_bar: the optional paste-stage progress bar - forwarded
                into the paste core, which ticks the eligible rows per
                completed chunk/group (the fallback rows are ticked by the
                processor's loop; the owner's fix 4, 2026-10-31).

        The background arrives as a
        :class:`~.keep_image_utils.BgrFrameSource` (the round-17 lazy path
        - the eligible **frames** are a view of the clip and the core
        converts the group's rows on demand) or as a numpy stack (the
        direct-call path of the tests and the tools).

        Returns:
            The ``{frame index: ...}`` mapping of the **eligible** frames -
            the frames that are not in it take the vendored fallback.  A
            device-output mode replaces the mapped frames with the sentinel
            keys, exactly like :meth:`_batched_paste`: the canvas is
            allocated over the eligible **frame** count
            (``device_output_mode`` counts canvas rows) and the faces ride
            the map; the chunk buffer (round 18) is the **whole-clip**
            float32 buffer - the eligible frames land in it by index.

            Since keep-fix5 (2026-11-03) a configured face upsampler is
            batch-eligible here too - the ``return {}`` abort is deleted,
            the upscaled faces ride the ROI chunks
            (:func:`upscale_faces_batch_gpu`).
        """
        parse_net = getattr(self.face_helper, "face_parse", None)
        if parse_net is None or not getattr(self.face_helper, "use_parse", False):
            return {}

        # The faces of every frame, in detection order (design section 3.2:
        # the entries are ordered by (frame index, detection index), so a
        # frame's faces are contiguous and every row owns whole faces).
        faces_by_frame: dict[int, list[int]] = {}
        for face_index, frame in enumerate(face_frames):
            faces_by_frame.setdefault(int(frame), []).append(face_index)

        eligible = []
        for frame, face_indices in faces_by_frame.items():
            if any(
                face_affines[k] is None
                or np.isnan(face_landmarks[k]).any()
                or restored_faces[k].shape[:2] != (PARSE_FACE_SIZE, PARSE_FACE_SIZE)
                for k in face_indices
            ):
                continue
            eligible.append(frame)
        if not eligible:
            return {}

        # The canvas rows are the eligible frames; the faces and the inverses
        # ride the map, a row after a row.
        batch_faces = [k for frame in eligible for k in faces_by_frame[frame]]
        face_frame = [
            row for row, frame in enumerate(eligible)
            for _ in faces_by_frame[frame]
        ]
        inverses = inverse_affines(
            [face_affines[k] for k in batch_faces],
            upscale_factor=final_upscale_factor)
        # The pass-through gates (round 4, D-R2.3.2) - the multi-face split
        # of the sibling's: the faces pass through when the batch covers
        # every restored face in order (a frame's faces are contiguous, so
        # the eligible frames' faces are ``range(len(restored_faces))``
        # exactly then); the background only when every frame is eligible -
        # the core enforces ``background.shape[0] == num_rows``
        # (``num_rows = len(eligible)`` here), so a clip with a face-less or
        # gray frame keeps the gather.
        if batch_faces == list(range(len(restored_faces))):
            faces_stacked = restored_faces
        else:
            # Round 19 (FIX 1): the multi-face split of the sibling's - the
            # lazy selected-rows source over the face-indexed stack, no
            # gather copy (see the sibling's comment).
            faces_stacked = paste_mod.LazyFaceSource(restored_faces).select(batch_faces)
        # The background of the batch (round 17) - the multi-face split of
        # the sibling's: the **provider** of the rows - the lazy source (the
        # paste stage's path) or the numpy stack (the direct-call path of
        # the tests and the tools).  The eligible **frames** are a view of
        # the source: the gather copy is gone, and the source converts the
        # rows per group inside the core.
        background_provider = paste_mod.frame_provider(input_frames_bgr)
        # Round 18: the clip's row count - the row count of the final
        # buffer.  It is read **before** the select (the select narrows
        # the provider to the eligible frames), and it is the frame count
        # the eligible frames are indexed by.
        num_input_frames = int(background_provider.shape[0])
        if eligible != list(range(num_input_frames)):
            background_provider = background_provider.select(eligible)

        canvas_h, canvas_w = paste_mod.paste_canvas_hw(
            background_provider.shape[1:3], final_upscale_factor, canvas_hw)
        # the canvas budget counts canvas rows - the eligible **frame**
        # count, not the face count (design section 3.4)
        mode = paste_mod.device_output_mode(self.device, len(eligible), canvas_h, canvas_w)
        # keep-fix5 (2026-11-03): a background model never takes the full
        # mode - the whole-video device canvas would block before the
        # first model frame; the chunked groups keep the progress alive.
        if self.bg_upscale_model is not None and mode == "full":
            mode = "chunked"

        if mode == "full":
            background = batched_bg_frames(
                background_provider.materialize(), final_upscale_factor,
                self.bg_upscale_model,
                self.device, canvas_hw=canvas_hw, return_device=True)
            canvas = paste_faces_batch(
                faces_bgr=faces_stacked,
                inverse_affines=inverses,
                background_bgr=background,
                device=self.device,
                parse_net=parse_net,
                face_upscale_model=self.face_upscale_model,
                device_output=True,
                face_frame=face_frame,
                num_frames=len(eligible),
                stage_bar=stage_bar)
            if draw_box:
                self._draw_boxes_on_canvas(canvas, inverses, (canvas_h, canvas_w), face_rows=face_frame)
            return {**dict(zip(eligible, list(canvas))), paste_mod.DEVICE_CANVAS_KEY: canvas}

        if mode == "chunked":
            # Round 18: the final whole-clip float32 buffer is allocated
            # **before** the paste and handed to the core - the eligible
            # frames land straight in it (the core writes them by index).
            # The core's own buffer would be a second whole-clip float32
            # buffer alive at the same time (the defect this round
            # removes).
            final_batch = torch.empty(
                (num_input_frames, canvas_h, canvas_w, 3), dtype=torch.float32)
            chunks = paste_mod.paste_faces_batch_chunked(
                faces_bgr=faces_stacked,
                inverse_affines=inverses,
                background_bgr=_core_background(background_provider),
                upscale_factor=final_upscale_factor,
                bg_upscale_model=self.bg_upscale_model,
                device=self.device,
                canvas_hw=canvas_hw,
                parse_net=parse_net,
                face_frame=face_frame,
                num_frames=len(eligible),
                stage_bar=stage_bar,
                face_upscale_model=self.face_upscale_model,
                out=final_batch,
                # the core's canvas row r is the r-th eligible frame, so
                # the eligible frames land in their final rows.
                out_rows=eligible)
            if draw_box:
                # the boxes land on the **final** rows (the float32 IMAGE
                # rows) - the eligible-space row of a face is
                # ``face_frame[face]``, so its final row is
                # ``eligible[face_frame[face]]``.
                self._draw_boxes_on_canvas(
                    chunks, inverses, (canvas_h, canvas_w),
                    face_rows=[eligible[row] for row in face_frame])
            # the mapped values are **views** into the final buffer (basic
            # integer indexing) - the tail drops them into a set, so no
            # second copy of the rows is ever alive.
            return {**{frame: chunks[frame] for frame in eligible},
                    paste_mod.DEVICE_OUTPUT_CHUNKS_KEY: chunks}

        background = batched_bg_frames(
            background_provider.materialize(), final_upscale_factor,
            self.bg_upscale_model,
            self.device, canvas_hw=canvas_hw)
        pasted = paste_faces_batch(
            faces_bgr=faces_stacked,
            inverse_affines=inverses,
            background_bgr=background,
            device=self.device,
            parse_net=parse_net,
            face_upscale_model=self.face_upscale_model,
            face_frame=face_frame,
            num_frames=len(eligible),
            stage_bar=stage_bar)
        return dict(zip(eligible, pasted))

    @torch.no_grad()
    def process_image(self, cv2_image_orig: np.ndarray, final_upscale_factor: float, has_aligned: bool,
                      only_center_face: bool, draw_box: bool):
        # The multi-face branch (C2, design section 3.5): ``only_center_face``
        # off on a non-aligned image restores **every** detected face.  The
        # single-face toggle and the aligned path keep the body below
        # byte-for-byte (the multi-face machinery lives entirely in
        # ``_process_image_multiface``).
        if not has_aligned and not only_center_face:
            return self._process_image_multiface(
                cv2_image_orig, final_upscale_factor, draw_box)
        # The aligned fast path (round 6, Opt 5, design section 5.2): the node
        # hands the processor an **already aligned** 512x512 face, so the
        # detection / crop / paste stages have nothing left to do and the
        # whole pipeline is one resize, one KEEP call and one output resize
        # (see :meth:`_restore_aligned_torch`).  The dispatch is in the
        # processor (not in the node): the node pins assert ``process_image``
        # is called, and the dispatch point is the only way to keep the
        # duplicate-to-2 contract, the black-face guard and the offload-always
        # ``finally`` in one place.  A configured face upsampler keeps the
        # standard pipeline (the precondition below) - the fast path never
        # applies one.
        if has_aligned and self.face_upscale_model is None:
            return self._restore_aligned_torch(cv2_image_orig, final_upscale_factor)

        try:
            self.face_helper.upscale_factor = final_upscale_factor
        
            input_img_list = [cv2_image_orig.copy()]
            avg_landmarks_list = None

            if not has_aligned:
                raw_landmarks_list_np = []
                img_for_detection = input_img_list[0]
                self.face_helper.clean_all()
                # The detection step calls into the vendored facelib: its console
                # writes are routed to the logger of the pack (D4).
                with capture_console_to_logger():
                    self.face_helper.read_image(img_for_detection)
                    num_det_faces = self.face_helper.get_face_landmarks_5(
                        only_center_face=only_center_face, resize=640, eye_dist_threshold=5,
                        only_keep_largest=True)

                keep_interrupt.check_interrupted()
                if num_det_faces >= 1:
                    raw_landmarks_list_np.append(self.face_helper.all_landmarks_5[0].reshape((10,)))
                else: return cv2_image_orig
                if not raw_landmarks_list_np: return cv2_image_orig
                raw_landmarks_np_array = np.array(raw_landmarks_list_np)
                for i_lm in range(raw_landmarks_np_array.shape[1]):
                    raw_landmarks_np_array[:, i_lm] = interpolate_sequence(raw_landmarks_np_array[:, i_lm])
                avg_landmarks_np_array = raw_landmarks_np_array.reshape(raw_landmarks_np_array.shape[0], 5, 2)
                avg_landmarks_list = [lm for lm in avg_landmarks_np_array]
        
            cropped_face_t_list = []
            img_to_crop = input_img_list[0]
            if not has_aligned:
                if avg_landmarks_list is None or not avg_landmarks_list: return img_to_crop
                self.face_helper.clean_all()
                # The crop step calls into the vendored facelib: its console
                # writes are routed to the logger of the pack (D4).
                with capture_console_to_logger():
                    self.face_helper.read_image(img_to_crop)
                    self.face_helper.all_landmarks_5 = [avg_landmarks_list[0]]
                    self.face_helper.align_warp_face()
            else:
                img_resized = resize_bgr_u8(img_to_crop, (512, 512))
                self.face_helper.is_gray = is_gray(img_resized, threshold=10)
                self.face_helper.cropped_faces = [img_resized]
            if not self.face_helper.cropped_faces: return cv2_image_orig
        
            cropped_face_cv2 = self.face_helper.cropped_faces[0]
            cropped_face_t = img2tensor(cropped_face_cv2 / 255., bgr2rgb=True, float32=True)
            # torchvision ``normalize`` became the equivalent in-place torch
            # arithmetic: (t - mean) / std with mean = std = 0.5, bit-exact.
            cropped_face_t.sub_(0.5).div_(0.5)
            cropped_face_t_list.append(cropped_face_t)
            cropped_faces_tensor = torch.stack(cropped_face_t_list, dim=0).unsqueeze(0).to(self.device)
            duplicated_input_frames = torch.cat([cropped_faces_tensor, cropped_faces_tensor], dim=1)
            net_output = self.keep_net(duplicated_input_frames, need_upscale=False)
            final_output_tensor = net_output[:, 0:1, ...].squeeze(0)
            restored_faces_cv2 = [tensor2img(x, rgb2bgr=True, min_max=(-1, 1)) for x in final_output_tensor]
            _soft_empty_cache()
            keep_interrupt.check_interrupted()

            img_for_pasting = input_img_list[0]
            self.face_helper.clean_all()
            if has_aligned:
                img_resized = resize_bgr_u8(img_for_pasting, (512, 512))
                self.face_helper.is_gray = is_gray(img_resized, threshold=10)
            else:
                if avg_landmarks_list is None or not avg_landmarks_list: return img_for_pasting
                with capture_console_to_logger():
                    self.face_helper.read_image(img_for_pasting)
                    self.face_helper.all_landmarks_5 = [avg_landmarks_list[0]]
                    self.face_helper.align_warp_face()

            if not restored_faces_cv2 or not restored_faces_cv2[0].any(): return cv2_image_orig
        
            self.face_helper.add_restored_face(restored_faces_cv2[0])

            if not has_aligned:
                bg_img_upscaled_by_model = self._run_upscaler(self.bg_upscale_model, img_for_pasting)

                h, w, _ = img_for_pasting.shape
                target_h, target_w = int(h * final_upscale_factor), int(w * final_upscale_factor)
                if bg_img_upscaled_by_model.shape[0] != target_h or bg_img_upscaled_by_model.shape[1] != target_w:
                    bg_img_final = cv2.resize(bg_img_upscaled_by_model, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    bg_img_final = bg_img_upscaled_by_model

                keep_interrupt.check_interrupted()
                # The paste step calls into the vendored facelib: its console
                # writes are routed to the logger of the pack (D4).
                with capture_console_to_logger():
                    self.face_helper.get_inverse_affine(None)
                    final_restored_image = self.face_helper.paste_faces_to_input_image(
                        upsample_img=bg_img_final,
                        draw_box=draw_box,
                        face_upsampler=self.face_upscale_model
                    )
            else:
                final_restored_image = self.face_helper.restored_faces[0]
                if self.face_upscale_model:
                     final_restored_image = self._run_upscaler(self.face_upscale_model, final_restored_image)
            
                h, w, _ = final_restored_image.shape
                target_h, target_w = int(512 * final_upscale_factor), int(512 * final_upscale_factor)
                if h != target_h or w != target_w:
                     final_restored_image = cv2.resize(final_restored_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            if final_restored_image is None:
                return cv2_image_orig

            return final_restored_image

        finally:
            # D-R2.4: the helper's stage arrays are the run's scratch -
            # one detected frame used to survive the run in ``input_img``.
            # The finally runs after the return value is evaluated; the
            # aligned path's returned ``restored_faces[0]`` array survives
            # the list clear by reference.
            self.face_helper.clean_all()
            self.face_helper.input_img = None

    @torch.no_grad()
    def _restore_aligned_torch(self, cv2_image_orig: np.ndarray,
                               final_upscale_factor: float) -> np.ndarray:
        """The aligned single-image fast path (round 6, Opt 5).

        ``has_aligned_face=True`` means the uploader already cropped and
        aligned a face: the detection, the crop, the vendored paste and the
        two ``512`` resizes of the standard leg all collapse into the one
        resize that feeds the net, so the aligned leg runs **on the device**
        without touching the face helper, the vendored ``img2tensor`` chain
        or the vendored paste.

        Deviations from the standard aligned leg (the owner-authorized pixel
        changes of the design):

        1. the input resize is the torch-native bilinear
           :func:`torch.nn.functional.interpolate` on the float tensor - the
           standard leg routes through :func:`resize_bgr_u8`
           (``keep_processor.py:509``), which rounds to uint8 first;
        2. the KEEP input tail keeps the spec's order: **``/255`` first, then**
           the normalize.  ``img2tensor(cropped / 255.)``
           (``keep_processor.py:515``) put the face in ``[0, 1]`` *before* the
           ``(t - 0.5) / 0.5``; reordering the two would double the range
           twice and feed the net ``~(-1, 509)`` instead of ``[-1, 1]``;
        3. the final upscale is float **bicubic** - the standard leg runs
           ``cv2.INTER_LANCZOS4`` on the uint8 face
           (``keep_processor.py:572``);
        4. the output quantizes at the very end, exactly like the vendored
           ``tensor2img`` it replaces, so the return is the cv2 BGR ``uint8``
           image the node expects - a float return would reach
           ``cv2_to_comfy_image`` (``keep_image_utils.py:38-42``, no ndim
           guard) and ``unsqueeze`` it into a 5-D "IMAGE" tensor.

        The channel flips (BGR -> RGB before the net, RGB -> BGR on the way
        out) exist because the node's ``comfy_image_to_cv2``
        (``keep_image_utils.py:34``) hands this method cv2 BGR.  Both polls
        survive - the io boundary one and the one after the KEEP call.
        """
        keep_interrupt.check_interrupted()
        img = torch.from_numpy(np.ascontiguousarray(cv2_image_orig)).to(self.device)
        # (H, W, 3) uint8 BGR -> (1, 3, 512, 512) float, bilinear.
        img512 = F.interpolate(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            size=(512, 512), mode="bilinear", align_corners=False)
        # /255 first (the replaced tail's ``cropped_face_cv2 / 255``), then
        # the BGR -> RGB flip and the normalize.
        face_norm = img512[:, [2, 1, 0]].div_(255.0).sub_(0.5).div_(0.5)
        input_seq = torch.cat([face_norm.unsqueeze(1), face_norm.unsqueeze(1)], dim=1)
        net_output = self.keep_net(input_seq, need_upscale=False)   # (1, 2, 3, 512, 512)
        keep_interrupt.check_interrupted()
        restored = net_output[:, 0:1, ...].squeeze(0)               # (1, 3, 512, 512) RGB
        restored = restored.clamp(-1, 1).add(1).div_(2)             # denormalize -> [0, 1]
        # The all-black guard (the analogue of ``keep_processor.py:540``): a
        # black restore returns the original, checked on the tensor before
        # the quantization.
        if not restored.any():
            return cv2_image_orig
        target = (int(512 * final_upscale_factor), int(512 * final_upscale_factor))
        # The size check mirrors the standard leg (``keep_processor.py:566``):
        # a factor below 1 still resizes.
        if restored.shape[-1] != target[0] or restored.shape[-2] != target[1]:
            restored = F.interpolate(
                restored, size=target, mode="bicubic", align_corners=False)
        # ``restored[0]`` drops the batch dim - ``process_image`` returns
        # (H, W, 3).
        return (restored[0].mul(255.0).round().clamp_(0, 255).to(torch.uint8)
                .permute(1, 2, 0)[..., [2, 1, 0]].cpu().numpy())  # (H, W, 3) BGR uint8

    def _process_image_multiface(self, cv2_image_orig: np.ndarray,
                                 final_upscale_factor: float, draw_box: bool) -> np.ndarray:
        """The multi-face single-image path (C2, design section 3.5).

        ``only_center_face=False`` restores **every** detected face of the
        image.  The single-face collapse was two sites deep: the detector
        call kept ``only_keep_largest=True`` (the vendored selector reduces
        to the largest face **first**, ``face_restoration_helper.py:247-255``
        - ``only_center_face=False`` never fired behind it), and the
        pipeline indexed ``all_landmarks_5[0]``.  Both are lifted here: the
        detector keeps all faces, the vendored ``align_warp_face`` crops
        each of them (it already loops), one KEEP batch holds every crop
        and **one** vendored ``paste_faces_to_input_image`` composites them
        all - the reference composite by construction (the vendored forward
        loop blends face k into the result of face k-1, ``:495``, and tails
        once per image, ``:497-507``).

        There is no landmark smoothing: the (F, 10) array of the single-face
        path fed a *temporal* interpolator along the face axis, which would
        blend coordinates **between** different faces - the landmarks are
        used as detected.

        The interrupt checks sit at the same stage boundaries as the
        single-face path (after detection, after the KEEP restore, before
        the paste - ``process_image`` / design section 2.2).
        """
        try:
            self.face_helper.upscale_factor = final_upscale_factor
            # One working copy: the three stages read the same image the
            # single-face path reads (detection, crop, paste), and the early
            # exits return the caller's original untouched.
            img = cv2_image_orig.copy()

            # Detection: all faces - the argument change **is** the fix (PM
            # risk #1 of the plan): keeping ``only_keep_largest=True`` would
            # collapse the detection before any downstream indexing could help.
            self.face_helper.clean_all()
            # The detection step calls into the vendored facelib: its console
            # writes are routed to the logger of the pack (D4).
            with capture_console_to_logger():
                self.face_helper.read_image(img)
                num_det_faces = self.face_helper.get_face_landmarks_5(
                    only_center_face=False, resize=640, eye_dist_threshold=5,
                    only_keep_largest=False)
            keep_interrupt.check_interrupted()
            if num_det_faces < 1:
                return cv2_image_orig
            # The (10,) round trip of the single-face path, without its
            # cross-axis interpolation (see the docstring): ``np.array``
            # materializes copies that survive the ``clean_all`` of the crop.
            raw_landmarks_list_np = [lm.reshape((10,)) for lm in self.face_helper.all_landmarks_5]
            raw_landmarks_np_array = np.array(raw_landmarks_list_np)
            avg_landmarks_np_array = raw_landmarks_np_array.reshape(raw_landmarks_np_array.shape[0], 5, 2)
            avg_landmarks_list = [lm for lm in avg_landmarks_np_array]

            # Crop: the vendored loop over every face (the single-face path
            # passed a one-element list until this round).  A ``None`` affine -
            # the LMEDS failure - leaves a ``None`` in **both** ``cropped_faces``
            # and ``affine_matrices`` (``face_restoration_helper.py:317-318``):
            # that face drops out of the batch, and the vendored paste skips a
            # ``None`` inverse gracefully (``:377-378``).
            self.face_helper.clean_all()
            # The crop step calls into the vendored facelib: its console
            # writes are routed to the logger of the pack (D4).
            with capture_console_to_logger():
                self.face_helper.read_image(img)
                self.face_helper.all_landmarks_5 = list(avg_landmarks_list)
                self.face_helper.align_warp_face()
            crop_indices = [
                i for i, affine in enumerate(self.face_helper.affine_matrices)
                if affine is not None]
            if not crop_indices:
                return cv2_image_orig

            # Restore: one KEEP batch over all the crops.  The image path's
            # duplicate-axis trick (the temporal KEEP wants a clip of >= 2)
            # doubles the face axis to ``2F`` and the first ``F`` outputs are
            # taken - the contracted faithful flatten (design section 3.5).
            cropped_face_t_list = []
            for i in crop_indices:
                cropped_face_cv2 = self.face_helper.cropped_faces[i]
                cropped_face_t = img2tensor(cropped_face_cv2 / 255., bgr2rgb=True, float32=True)
                # torchvision ``normalize`` became the equivalent in-place torch
                # arithmetic: (t - mean) / std with mean = std = 0.5, bit-exact.
                cropped_face_t.sub_(0.5).div_(0.5)
                cropped_face_t_list.append(cropped_face_t)
            num_faces = len(crop_indices)
            cropped_faces_tensor = torch.stack(cropped_face_t_list, dim=0).unsqueeze(0).to(self.device)
            duplicated_input_frames = torch.cat([cropped_faces_tensor, cropped_faces_tensor], dim=1)
            net_output = self.keep_net(duplicated_input_frames, need_upscale=False)
            final_output_tensor = net_output[:, 0:num_faces, ...].squeeze(0)
            restored_faces_cv2 = [tensor2img(x, rgb2bgr=True, min_max=(-1, 1)) for x in final_output_tensor]
            _soft_empty_cache()
            keep_interrupt.check_interrupted()

            # The per-face all-black guard (the analogue of the single-face
            # ``restored_faces_cv2[0].any()``): a fully black restore drops out
            # of the paste - together with its landmarks, so the vendored
            # face-to-inverse pairing stays positional - and if every face
            # drops, the original image comes back.
            survivors = [
                (avg_landmarks_list[i], face)
                for i, face in zip(crop_indices, restored_faces_cv2)
                if face.any()]
            if not survivors:
                return cv2_image_orig

            # Paste: one vendored composite of all the faces - the helper's own
            # multi-face flow (``add_restored_face`` per face, one
            # ``get_inverse_affine``, one ``paste_faces_to_input_image``).  The
            # paste-stage ``align_warp_face`` recomputes the affines of the
            # surviving faces (the same recompute the single-face path runs).
            self.face_helper.clean_all()
            # The paste step calls into the vendored facelib: its console
            # writes are routed to the logger of the pack (D4).
            with capture_console_to_logger():
                self.face_helper.read_image(img)
                self.face_helper.all_landmarks_5 = [lm for lm, _ in survivors]
                self.face_helper.align_warp_face()
            for _, face in survivors:
                self.face_helper.add_restored_face(face)

            bg_img_upscaled_by_model = self._run_upscaler(self.bg_upscale_model, img)

            h, w, _ = img.shape
            target_h, target_w = int(h * final_upscale_factor), int(w * final_upscale_factor)
            if bg_img_upscaled_by_model.shape[0] != target_h or bg_img_upscaled_by_model.shape[1] != target_w:
                bg_img_final = cv2.resize(bg_img_upscaled_by_model, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                bg_img_final = bg_img_upscaled_by_model

            keep_interrupt.check_interrupted()
            with capture_console_to_logger():
                self.face_helper.get_inverse_affine(None)
                final_restored_image = self.face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img_final,
                    draw_box=draw_box,
                    face_upsampler=self.face_upscale_model
                )

            if final_restored_image is None:
                return cv2_image_orig

            return final_restored_image

        finally:
            # D-R2.4: the helper's stage arrays are the run's scratch -
            # one detected frame used to survive the run in ``input_img``.
            # The finally runs after the return value is evaluated; the
            # aligned path's returned ``restored_faces[0]`` array survives
            # the list clear by reference.
            self.face_helper.clean_all()
            self.face_helper.input_img = None

    def _detection_frames(self,
                          image_sequence_tensor: torch.Tensor,
                          indices,
                          needs_upscale: list[bool]) -> dict[int, np.ndarray]:
        """The images the detector saw for ``indices``, in one group pass.

        The round-4 re-derivation (D-R2.2.1) of the released
        ``detection_frames`` list, in the **grouped** form: one batched
        conversion of the gathered rows instead of one single-row
        conversion per frame - the converter is elementwise, so every row
        is bitwise the row the per-row form produced (and the round-3 crop
        consumed), while the chunk pays one host transfer instead of
        ``len(indices)``.  For the sub-512 frames the vendored
        ``read_image`` re-runs to reproduce its INTER_LINEAR upscale - the
        deterministic image the crop warps consume and the paste sizes its
        canvas from (``face_restoration_helper.py:182-184``).  The caller
        keeps the helper's console writes captured; the rows are bounded by
        the caller's chunk (at most ``max_clip_length`` of them).

        Returns:
            ``{index: the detection image}`` - the raw converted row, or
            the vendored upscale of it.
        """
        unique = sorted({int(index) for index in indices})
        rows = comfy_batch_to_bgr_u8(image_sequence_tensor[unique])
        frames: dict[int, np.ndarray] = {}
        for position, index in enumerate(unique):
            row = rows[position]
            if needs_upscale[index]:
                # the same vendored re-derivation the per-row form runs -
                # the one ``read_image`` call site of the sequence path.
                # keep-fix5-implement1 (2026-11-03): ``check_gray=False``
                # skips the per-frame PIL ``is_gray`` - Stage 4 computes the
                # gray flags on the device (``is_gray_batch``) and the paste
                # fallback re-sets ``is_gray`` from them.
                self.face_helper.read_image(row, check_gray=False)
                frames[index] = self.face_helper.input_img
            else:
                frames[index] = row
        return frames

    def _detection_frame(self,
                         image_sequence_tensor: torch.Tensor,
                         index: int,
                         needs_upscale: list[bool]) -> np.ndarray:
        """The image the detector saw for ``index`` - the group form, one row.

        The paste fallback's per-frame derivation (D-R2.3.3): the same
        values :meth:`_detection_frames` returns, for a single index.
        """
        return self._detection_frames(
            image_sequence_tensor, [index], needs_upscale)[index]

    @torch.no_grad()
    def process_image_sequence(self,
                               image_sequence_tensor: torch.Tensor,
                               final_upscale_factor: float,
                               has_aligned_frames: bool,
                               only_center_face: bool,
                               draw_box: bool,
                               max_clip_length: int = 20):
        num_input_frames = image_sequence_tensor.shape[0]
        if num_input_frames == 0:
            return image_sequence_tensor

        # The multi-face branch (C2, design section 3.1): ``only_center_face``
        # off on a non-aligned sequence restores **every** detected face of
        # every frame.  The single-face toggle and the aligned path keep the
        # body below byte-for-byte (the multi-face machinery lives entirely
        # in ``_process_sequence_multiface``).
        try:
            if not has_aligned_frames and not only_center_face:
                return self._process_sequence_multiface(
                    image_sequence_tensor, final_upscale_factor, draw_box,
                    max_clip_length)

            # P4.7: the V3 progress API replaces the V1 ``comfy.utils.ProgressBar``
            # (``src/keep_progress.py`` reports through the execution API).
            # Round 4 (D-R2.3.1) added the second convert-in of the paste
            # stage: the total is five families per frame.
            progress = ProgressReporter(num_input_frames * 5)

            avg_landmarks_sequence = None
            # The affine matrices of the frames (design section 7.3): the crop
            # computes them once, the paste loop reuses them.
            affine_matrices: list = [None] * num_input_frames
            # The detection images are no longer retained (D-R2.1): the flag
            # records the frames whose detection image is the vendored
            # upscale, and the crop / paste re-derive it on demand.
            needs_upscale = [False] * num_input_frames
            # The canvas of the paste (F1), recorded at the detection pass:
            # the image the vendored ``read_image`` produced for frame 0 -
            # the raw frame, or the frame upscaled for min(h, w) < 512 (the
            # value ``detection_frames[0].shape[:2]`` used to provide).
            canvas_hw = None
            # The detection group size (D-R2.1): the staged detection
            # converts ``group_size`` frames, detects, deletes - no
            # whole-clip uint8 stack.  The 2-arg range keeps the G4 chunk
            # tripwire (``tests/test_keep_regressions.py``) pointing at the
            # one 3-arg ``range(0, num_input_frames, max_clip_length)`` - the
            # KEEP chunks of the fused loop below.
            group_size = max(1, max_clip_length)
            if not has_aligned_frames:
                logger.info("Detecting face landmarks")
                # The stage bar of the detection (the stage-bar design, D3
                # #1): one tick per processed frame, face-less frames too.
                detection_bar = StageBar("Detecting face landmarks", num_input_frames)
                # The conversion spend is counted upfront (the group slices
                # sum to the clip); the detection update lands after the
                # pass, where the whole-clip loop used to report it.
                progress.update(num_input_frames)
                raw_landmarks_list_video = []
                # The detection step calls into the vendored facelib: its
                # console writes are routed to the logger of the pack (D4).
                with capture_console_to_logger():
                    for group_index in range(
                            (num_input_frames + group_size - 1) // group_size):
                        group_start = group_index * group_size
                        group_end = min(group_start + group_size, num_input_frames)
                        # A slice converts to bitwise-identical rows (the
                        # batched converter is elementwise) - the group is
                        # deleted before the next one is allocated.
                        bgr_group = comfy_batch_to_bgr_u8(
                            image_sequence_tensor[group_start:group_end])
                        for i in range(group_start, group_end):
                            # The cancel poll (C1): ComfyUI's Cancel button
                            # raises here - a no-op outside ComfyUI
                            # (``src/keep_interrupt.py``).
                            keep_interrupt.check_interrupted()
                            img = bgr_group[i - group_start]
                            self.face_helper.clean_all()
                            # keep-fix5-implement1 (2026-11-03):
                            # ``check_gray=False`` skips the per-frame PIL
                            # ``is_gray`` - Stage 4 computes the gray flags
                            # on the device (``is_gray_batch``) and the paste
                            # fallback re-sets ``is_gray`` from them.
                            self.face_helper.read_image(img, check_gray=False)
                            # ``read_image`` replaces the input with an
                            # INTER_LINEAR upscale iff min(h, w) < 512
                            # (``face_restoration_helper.py:182-184``); the
                            # other branches cannot fire on uint8 3-channel
                            # rows.  The identity flags the frames the crop
                            # has to re-derive.
                            needs_upscale[i] = self.face_helper.input_img is not img
                            if canvas_hw is None:
                                canvas_hw = tuple(self.face_helper.input_img.shape[:2])
                            num_det_faces = self.face_helper.get_face_landmarks_5(
                                only_center_face=only_center_face, resize=640, eye_dist_threshold=5,
                                only_keep_largest=True)
                            if num_det_faces >= 1:
                                raw_landmarks_list_video.append(
                                    self.face_helper.all_landmarks_5[0].reshape((10,)))
                            else:
                                raw_landmarks_list_video.append(np.array([np.nan] * 10))
                            detection_bar.tick()
                        del bgr_group
                # The final line lands before the all-NaN early return below:
                # the early return carries the stage's bar line (D3 #1).
                detection_bar.finish()
                raw_landmarks_np_video = np.array(raw_landmarks_list_video)
                if np.all(np.isnan(raw_landmarks_np_video)):
                    return image_sequence_tensor
                for i_lm_coord in range(raw_landmarks_np_video.shape[1]):
                    raw_landmarks_np_video[:, i_lm_coord] = interpolate_sequence(
                        raw_landmarks_np_video[:, i_lm_coord])
                # The whole-clip smoothing stays after the whole detection
                # pass - it is the reason detection completes before crop.
                avg_landmarks_sequence = _gaussian_smooth(
                    raw_landmarks_np_video, sigma=5.0).reshape(num_input_frames, 5, 2)
                progress.update(num_input_frames)
            else:
                progress.update(num_input_frames)

            logger.info("Cropping and aligning faces")
            if not has_aligned_frames:
                # The affine estimation is the crop's vendored LMEDS call
                # (design section 7.3, tolerance decision D2): the matrices
                # are computed once, the fused loop warps them and the paste
                # loop reuses them.  A ``None`` matrix (the LMEDS failure)
                # keeps the zero-face and bg-only handling of the
                # no-landmark frames.
                # The crop step calls into the vendored facelib: its console
                # writes are routed to the logger of the pack (D4).
                with capture_console_to_logger():
                    valid_indices = [i for i in range(num_input_frames)
                                     if not np.isnan(avg_landmarks_sequence[i]).any()]
                    # The stage bar of the crop (the stage-bar design, D3
                    # #2): one tick per affine assignment; the aligned path
                    # constructs no bar (its stage line stays alone).
                    crop_bar = StageBar(
                        "Cropping and aligning faces", len(valid_indices))
                    estimated_affines = estimate_affines(
                        [avg_landmarks_sequence[i] for i in valid_indices],
                        self.face_helper.face_template)
                    for i, affine_matrix in zip(valid_indices, estimated_affines):
                        affine_matrices[i] = affine_matrix
                        crop_bar.tick()
                        if affine_matrix is None:
                            logger.warning(
                                f"Warning: Could not estimate affine matrix for landmark set 0 "
                                f"(frame {i}). Skipping this face."
                            )
                    crop_bar.finish()

            # The fused, chunk-bounded crop + restore (D-R2.2): the same
            # global chunks the restore always ran (``max_clip_length``
            # frames - the quality contract, G4).  Per chunk the crops are
            # warped in one batched device op per source-shape group from a
            # preallocated ``max_clip_length``-bounded staging array (no
            # list, no ``np.stack`` of the clip) and the restored rows land
            # in one preallocated whole-clip stack through the
            # row-independent per-chunk ``tensor2img_batch`` - no whole-clip
            # device batch, no whole-clip tensor2img transient.
            all_restored_faces_cv2 = np.empty(
                (num_input_frames, 512, 512, 3), np.uint8)
            logger.info("Restoring faces with KEEP")
            # The stage bar of the KEEP restore (the stage-bar design, D3
            # #3): one tick of the chunk's frame count per chunk; an OOM
            # leaves without the final line (the exception is the report).
            restore_bar = StageBar("Restoring faces with KEEP", num_input_frames)
            try:
                for start_idx in range(0, num_input_frames, max_clip_length):
                    keep_interrupt.check_interrupted()
                    end_idx = min(start_idx + max_clip_length, num_input_frames)
                    if has_aligned_frames:
                        # The aligned frames need no landmark/upscale
                        # derivation: the chunk's frames convert, resize and
                        # stack chunk-locally (the values of the whole-clip
                        # stack, one chunk at a time).
                        aligned_rows = comfy_batch_to_bgr_u8(
                            image_sequence_tensor[start_idx:end_idx])
                        resized_faces = [resize_bgr_u8(frame, (512, 512))
                                         for frame in aligned_rows]
                        del aligned_rows
                        aligned_u8 = torch.from_numpy(
                            np.stack(resized_faces)).permute(0, 3, 1, 2).to(self.device)
                        del resized_faces
                        current_clip_faces = bgr_u8_batch_to_normalized(aligned_u8)
                        del aligned_u8
                    else:
                        croppable_rows = [i for i in range(start_idx, end_idx)
                                          if affine_matrices[i] is not None]
                        # The frames without a (croppable) face carry the
                        # base's neutral crop (``torch.zeros(3, 512, 512) *
                        # 0.5`` of the base - 0.5 in the normalized scale,
                        # not 0): the chunk-local fill of the whole-clip
                        # tensor's values.
                        current_clip_faces = torch.full(
                            (end_idx - start_idx, 3, 512, 512), 0.5,
                            dtype=torch.float32, device=self.device)
                        if croppable_rows:
                            # The crop step calls into the vendored facelib:
                            # its console writes are routed to the logger of
                            # the pack (D4) - the derivation's ``read_image``
                            # included.
                            with capture_console_to_logger():
                                # The chunk's detection images in ONE group
                                # pass (D-R2.2.1, grouped): the conversion is
                                # elementwise, so the rows are bitwise the
                                # rows the per-row form produced - one host
                                # transfer per chunk instead of one per row.
                                chunk_detection_frames = self._detection_frames(
                                    image_sequence_tensor, croppable_rows,
                                    needs_upscale)
                                # One batched device warp per source-shape
                                # group replaces the per-row
                                # ``cv2.warpAffine``: the staging holds the
                                # detection frames (pre-warp) and stays
                                # bounded by the chunk (at most
                                # ``max_clip_length`` rows - in practice
                                # exactly one group).
                                for (h, w), members in _group_by_shape(
                                        chunk_detection_frames,
                                        croppable_rows).items():
                                    staged_u8 = np.empty(
                                        (len(members), h, w, 3), np.uint8)
                                    for row, i in enumerate(members):
                                        staged_u8[row] = chunk_detection_frames[i]
                                    batch = torch.from_numpy(staged_u8).permute(
                                        0, 3, 1, 2).to(self.device)
                                    del staged_u8
                                    warped_u8 = crop_faces_batch(
                                        batch,
                                        [affine_matrices[i] for i in members],
                                        (512, 512))
                                    del batch
                                    current_clip_faces[torch.as_tensor(
                                        [i - start_idx for i in members],
                                        dtype=torch.long, device=self.device)] = (
                                        bgr_u8_batch_to_normalized(warped_u8))
                                    del warped_u8
                            del chunk_detection_frames
                    batched_clip_faces = current_clip_faces.unsqueeze(0)
                    # keep-fix5-implement1 (2026-11-03) wrapped this forward
                    # in FP16 autocast (plan Step 3.4); REGRESSION FIX
                    # (2026-11-03, owner-measured): on the owner's GPU the
                    # FP16 forward returned all-NaN - the restored faces came
                    # out black (KEEP-diag zero_frac=1.0), the parse masks on
                    # them collapsed to 0, and the video output became the
                    # raw input.  The FP16-fragile op is not identified, so
                    # the restore runs FP32 (the pre-round-13 behavior)
                    # until it is.
                    if batched_clip_faces.shape[1] == 1:
                        current_clip_faces_for_net = torch.cat(
                            [batched_clip_faces, batched_clip_faces], dim=1)
                        clip_output_tensor = self.keep_net(
                            current_clip_faces_for_net, need_upscale=False)
                        kept_clip_tensor = clip_output_tensor[:, 0:1, ...]
                    else:
                        clip_output_tensor = self.keep_net(
                            batched_clip_faces, need_upscale=False)
                        kept_clip_tensor = clip_output_tensor
                    restored_u8 = tensor2img_batch(
                        kept_clip_tensor.squeeze(0), min_max=(-1, 1))
                    all_restored_faces_cv2[start_idx:end_idx] = restored_u8
                    del (current_clip_faces, batched_clip_faces,
                         clip_output_tensor, kept_clip_tensor, restored_u8)
                    progress.update(end_idx - start_idx)
                    restore_bar.tick(end_idx - start_idx)
            except torch.cuda.OutOfMemoryError:
                # The finished chunk's device transients die with the
                # exception; the release hands the cached blocks back before
                # ComfyUI sees the failure, and the exception propagates
                # unchanged (fail loudly).  On CPU the clause never matches.
                _soft_empty_cache()
                raise

            restore_bar.finish()

            # D-R2.3.1: convert-in #2 - the detection / crop / restore window
            # ran with no whole-clip uint8 BGR alive; the paste stage needs
            # the frames again.  Round 17 (the final stage's RAM): the frames
            # come from the **lazy source** - the rows are converted on
            # demand (the paste core per group, the fallback rows per
            # group), so the whole-clip uint8 stack is never built.
            # The converter is the module global, resolved here - after any
            # monkeypatch - so a spy sees every conversion of the run.
            frame_source = BgrFrameSource(
                image_sequence_tensor, convert=comfy_batch_to_bgr_u8)
            progress.update(num_input_frames)

            output_frames_cv2: list[np.ndarray] = []
            fallback_indices: list[int] = []
            device_canvas: torch.Tensor | None = None
            final_batch: torch.Tensor | None = None
            # Round 7 (the paste-budget repair): the paste's mode verdict reads
            # the **free** VRAM - ``device_canvas_budget`` (min of the 1 GiB cap
            # and the free share, ``keep_gpu_paste.py:233-251``) through
            # ComfyUI's ``get_free_memory(..., torch_free_too=True)``, which
            # counts what the torch cache can give back
            # (``keep_gpu_paste.py:178-192``).  Round 6 (Opt 6) deleted the
            # per-chunk ``_soft_empty_cache()`` of the restore loop above, so
            # the cached restore blocks hoard the memory the verdict reads: the
            # budget shrinks, the mode flips from ``full`` / ``chunked`` to
            # ``host``, and a clip that used to paste on the device takes the
            # host CPU assembly (the owner's paste-stage stall).  One release
            # here - after the last ``del`` of the restore loop and before the
            # verdict - restores the pre-Opt-6 budget without the per-chunk
            # sync cost Opt 6 removed (a single call per run, not one per
            # chunk; the ``finally`` / OOM releases stay).
            _soft_empty_cache()
            logger.info("Pasting faces and finalizing frames")
            # The stage bar of the paste (the stage-bar design, D3 #4): the
            # fallback rows tick here in the loop; the batched rows tick
            # inside the paste core per completed chunk/group - nobody ticks
            # twice, so the bar totals ``num_input_frames`` exactly (the
            # owner's fix 1.1, 2026-10-30).
            paste_bar = StageBar("Pasting faces and finalizing frames", num_input_frames)
            # The paste step calls into the vendored facelib: its console
            # writes are routed to the logger of the pack (D4).
            with capture_console_to_logger():
                # The gray flags of the frames (the vendored ``is_gray`` of
                # ``facelib/utils/misc.py:146``) - the same variances, without
                # the PIL round trip the per-frame paste made.  On CUDA they run
                # on the device (the paste-stage CPU optimization round, section
                # 4.2 - the host pass was the largest host-side item of the
                # stage); the CPU device keeps the numpy computation.
                gray_device = self.device if torch.device(self.device).type == "cuda" else None
                # Round 17: the flags are computed **per fallback group**,
                # with the group's frames (the whole-clip gray pass is gone)
                # - the eligible rows' flags are never read, so they stay
                # False.
                gray_flags = np.zeros(num_input_frames, dtype=bool)
                # The batched paste (design section 7.7): the default branch of
                # the paste stage runs on the device - the chunked ParseNet
                # forward, the batched parse mask, the batched inverse warp and
                # the batched blend.  The vendored paste stays reachable for
                # the non-default branches (the face upsampler and a helper
                # without a parse net); since the owner's fix 1.2
                # (2026-10-30) the grayscale frames ride the batched core and
                # since the owner's fix 4 (2026-10-31) ``draw_box`` rides it
                # too - the boxes land on the device canvases.
                # The paste bar is threaded in (fix 1.1): the core ticks the
                # eligible rows per completed chunk/group.
                batched_outputs = {}
                if not has_aligned_frames:
                    batched_outputs = self._batched_paste(
                        frame_source, all_restored_faces_cv2, affine_matrices,
                        avg_landmarks_sequence, final_upscale_factor, draw_box, gray_flags,
                        canvas_hw=canvas_hw, stage_bar=paste_bar)

                # The single tail buffer (review F4): the mode booleans are
                # set at allocation - and in the device modes the mapping
                # becomes a plain index set, because its dict values are row
                # views that would keep the canvas / chunk buffer alive
                # until the return.
                canvas = batched_outputs.pop(paste_mod.DEVICE_CANVAS_KEY, None)
                chunks_out = batched_outputs.pop(paste_mod.DEVICE_OUTPUT_CHUNKS_KEY, None)
                # The fallback rows (round 17): the rows the batched paste
                # did not take - the mapping's keys are the eligible rows.
                # Their frames and their gray flags come from the lazy
                # source, **per group**: the whole-clip uint8 stack and the
                # whole-clip gray pass are gone.
                fallback_rows = [i for i in range(num_input_frames)
                                 if i not in batched_outputs]
                gray_groups = self._fallback_gray_groups(
                    frame_source, fallback_rows, gray_flags, gray_device,
                    max_clip_length)
                # The first group is pulled here: the gray pass belongs to
                # the processor's contract even when no row falls back (the
                # empty group).
                group_rows, group_frames = next(gray_groups)
                device_canvas_mode = False
                chunks_mode = False
                if canvas is not None:
                    if len(batched_outputs) == num_input_frames:
                        # every frame is batched: the canvas already holds the
                        # rows in order - alias it (no fresh buffer, no copy)
                        device_canvas = canvas
                    else:
                        # one device canvas of the whole batch: the batched
                        # indices are already in it (it is the group canvas,
                        # indexed copy), the vendored-fallback frames are
                        # uploaded as one stack below, and the single
                        # convert-out runs on the device
                        device_canvas = torch.empty(
                            (num_input_frames, *canvas.shape[1:]), dtype=torch.uint8,
                            device=self.device)
                        device_canvas[list(batched_outputs)] = canvas
                    device_canvas_mode = True
                    batched_outputs = set(batched_outputs)
                    canvas = None
                if chunks_out is not None:
                    # Round 18: the twin's buffer **is** the final buffer -
                    # the eligible rows landed in it (the core wrote them
                    # by index) and the vendored-fallback frames are
                    # written host-side into the same object below.  One
                    # whole-clip float32 buffer in the mixed case; the
                    # all-eligible case is the same object (the alias - no
                    # copy).  The old eligible-rows scatter is gone.
                    final_batch = chunks_out
                    chunks_mode = True
                    batched_outputs = set(batched_outputs)
                    chunks_out = None
                    if final_batch.shape[0] != num_input_frames:
                        raise RuntimeError(
                            f"the paste buffer has {final_batch.shape[0]} rows "
                            f"for {num_input_frames} input frames")
                for i in range(num_input_frames):
                    keep_interrupt.check_interrupted()
                    if i in batched_outputs:
                        # the device modes already hold the frame (the canvas
                        # or the chunk buffer); the host mode keeps the mapping
                        # value.  No bar tick here: the core ticked the row
                        # when its chunk/group completed (fix 1.1).
                        if not device_canvas_mode and not chunks_mode:
                            output_frames_cv2.append(batched_outputs[i])
                        progress.update(1)
                        continue
                    # Fallback path for frames skipped by GPU batching: the
                    # loop owns this row's paste, so it ticks the bar.
                    paste_bar.tick(1)
                    # The row's frame (round 17): the lazy source's group
                    # that holds the row - the group is pulled when the row
                    # needs it, so only one group of uint8 frames is alive
                    # at a time.
                    while i not in group_rows:
                        if chunks_mode:
                            # Round 19 (FIX 2): the loop just advanced past a
                            # gray group - the group is complete, so its
                            # collected fallback frames convert **now** (per
                            # group, never the whole fallback) and are freed.
                            _flush_fallback_group(
                                final_batch, group_rows, output_frames_cv2,
                                fallback_indices)
                        group_rows, group_frames = next(gray_groups)
                    original_frame_bgr = group_frames[group_rows.index(i)]
                    # If the affine matrix is None, NO FACE was detected for
                    # the frame: do not run the vendored paste on it - return
                    # the background (the owner's fix 1.1, the critical fix).
                    if not has_aligned_frames:
                        if (
                            affine_matrices[i] is None
                            or avg_landmarks_sequence is None
                            or i >= len(avg_landmarks_sequence)
                            or np.isnan(avg_landmarks_sequence[i]).any()
                        ):
                            final_pasted_frame = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                            # Round 11 (2026-10-31): the fix-1.1 snippet's raw
                            # upscaler output cannot join the IMAGE batch when
                            # frame x model.scale != canvas x factor (the same
                            # disclosed deviation as the multiface
                            # short-circuit's resize): the output size is the
                            # paste canvas, reached with the same LANCZOS4
                            # resize the vendored paste applied to its no-face
                            # frames.
                            canvas_h, canvas_w = paste_mod.paste_canvas_hw(
                                original_frame_bgr.shape[:2], final_upscale_factor, canvas_hw)
                            if final_pasted_frame.shape[0] != canvas_h or final_pasted_frame.shape[1] != canvas_w:
                                final_pasted_frame = cv2.resize(
                                    final_pasted_frame, (canvas_w, canvas_h),
                                    interpolation=cv2.INTER_LANCZOS4)
                            output_frames_cv2.append(final_pasted_frame)
                            # the same bookkeeping as the sibling fallback rows:
                            # the device-output tails index the fallback rows by
                            # this list, so an unrecorded frame length-mismatches
                            fallback_indices.append(i)
                            progress.update(1)
                            continue
                    if i >= len(all_restored_faces_cv2):
                        final_pasted_frame = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                        output_frames_cv2.append(final_pasted_frame)
                        fallback_indices.append(i)
                        progress.update(1)
                        continue
                    restored_face_patch_cv2 = all_restored_faces_cv2[i]
                    self.face_helper.clean_all()
                    # Read once (design section 7.2): ``input_img`` is the
                    # image the vendored ``read_image`` produced - the frame
                    # upscaled for min(h, w) < 512, re-derived
                    # deterministically for the frames whose detection image
                    # was the upscale (D-R2.3.3), the converted row
                    # otherwise.  The vendored paste sizes its canvas from
                    # it (``face_restoration_helper.py:362-370``).
                    if needs_upscale[i]:
                        self.face_helper.input_img = self._detection_frame(
                            image_sequence_tensor, i, needs_upscale)
                    else:
                        self.face_helper.input_img = original_frame_bgr
                    self.face_helper.is_gray = bool(gray_flags[i])
                    # ``tensor2img_batch`` rows are uint8 already - no copy
                    # (D-R2.3.4; ``add_restored_face`` does not mutate).
                    self.face_helper.add_restored_face(restored_face_patch_cv2)

                    if not has_aligned_frames:
                        # Affine once (design section 7.3): the crop step
                        # computed the matrix - the paste loop of today ran
                        # a second ``align_warp_face`` to compute it again.
                        self.face_helper.affine_matrices = [affine_matrices[i]]

                        bg_img_upscaled_by_model = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                        h, w, _ = original_frame_bgr.shape
                        target_h, target_w = int(h * final_upscale_factor), int(w * final_upscale_factor)
                        if bg_img_upscaled_by_model.shape[0] != target_h or bg_img_upscaled_by_model.shape[1] != target_w:
                            bg_img_final = cv2.resize(bg_img_upscaled_by_model, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                        else:
                            bg_img_final = bg_img_upscaled_by_model

                        self.face_helper.upscale_factor = final_upscale_factor
                        self.face_helper.get_inverse_affine(None)
                        final_pasted_frame = self.face_helper.paste_faces_to_input_image(
                            upsample_img=bg_img_final,
                            draw_box=draw_box,
                            face_upsampler=self.face_upscale_model
                        )
                    else:
                        restored_face_with_model = self.face_helper.restored_faces[0]
                        if self.face_upscale_model:
                            restored_face_with_model = self._run_upscaler(self.face_upscale_model, restored_face_with_model)

                        h_restored, w_restored, _ = restored_face_with_model.shape
                        target_h, target_w = int(512 * final_upscale_factor), int(512 * final_upscale_factor)
                        if h_restored != target_h or w_restored != target_w:
                            final_pasted_frame = cv2.resize(restored_face_with_model, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                        else:
                            final_pasted_frame = restored_face_with_model

                        if len(final_pasted_frame.shape) == 2:
                            final_pasted_frame = _gray_to_bgr(final_pasted_frame)

                    output_frames_cv2.append(final_pasted_frame)
                    fallback_indices.append(i)
                    progress.update(1)

            paste_bar.finish()

            # The lazy source and its last group are the paste stage's
            # scratch (round 17) - release them before the convert-out
            # tails (D-R2.3.6).
            # Round 19 (FIX 2): in the chunked device mode the last gray
            # group's collected fallback frames convert **here** - before
            # the group bookkeeping is released (the flush reads the last
            # group's rows).  The loop already flushed every group it
            # advanced past; the last group is the only one still pending.
            if chunks_mode:
                _flush_fallback_group(
                    final_batch, group_rows, output_frames_cv2,
                    fallback_indices)
            del gray_groups, group_rows, group_frames
            del frame_source
            # The batched convert-out (design section 7.8): one stack and one
            # transfer, where the per-frame loop made a tensor per frame.
            if chunks_mode:
                # Round 19 (FIX 2): the chunked device mode converts the
                # vendored-fallback frames **per gray group** - the pull
                # site flushed every group the loop advanced past and the
                # flush above wrote the last group's rows.  The
                # whole-fallback stack and the whole-stack float32
                # conversion are gone: the peak is one group's stack plus
                # its conversion transient (<= max_clip_length rows),
                # independent of the fallback count.  The other output modes
                # keep the whole-stack form below (they are not the
                # production chunked path).
                return final_batch
            if device_canvas_mode:
                # F1: the fallback frames are still host arrays - one stacked
                # upload, never one transfer per frame; the convert-out runs on
                # the device in budget-sized sub-chunks (the out-of-place chain
                # peaks at about 9x the canvas bytes of device transient)
                if output_frames_cv2:
                    device_canvas[fallback_indices] = torch.from_numpy(
                        np.stack(output_frames_cv2)).to(self.device)
                return paste_mod.comfy_convert_out(
                    device_canvas, budget=paste_mod.device_canvas_budget(self.device))
            if not output_frames_cv2:
                return image_sequence_tensor
            return bgr_u8_batch_to_comfy(np.stack(output_frames_cv2))
        finally:
            # D-R2.4: the helper's stage arrays are the run's scratch - one
            # detected frame used to survive the run in ``input_img``.  The
            # finally runs after the return value is evaluated; the arrays
            # the return carries survive the list clears by reference.
            self.face_helper.clean_all()
            self.face_helper.input_img = None
            # P4.6 (Opt 6): the release is per run, not per chunk - the
            # chunk shapes repeat, so the caching allocator reuses its
            # blocks instead of accumulating, and the cached ones are
            # handed back once, when the run is done.
            _soft_empty_cache()

    def _process_sequence_multiface(self,
                                    image_sequence_tensor: torch.Tensor,
                                    final_upscale_factor: float,
                                    draw_box: bool,
                                    max_clip_length: int) -> torch.Tensor:
        """The multi-face sequence path (C2, design sections 3.2 and 3.3).

        ``only_center_face=False`` restores **every** detected face of every
        frame.  The single-face collapse was two sites deep: the detector
        call kept ``only_keep_largest=True`` (the vendored selector reduces
        to the largest face **first**,
        ``face_restoration_helper.py:247-255`` - ``only_center_face=False``
        never fired behind it), and the pipeline indexed
        ``all_landmarks_5[0]``.  Both are lifted here: the detector keeps all
        faces, every face of every frame becomes an entry of a **flattened**
        face batch, and the pastes are wired through the face-to-frame map of
        ``src/keep_gpu_paste.py`` (the t3 paste core).

        The face entries are ordered by ``(frame index, detection index)``:
        a frame's faces are contiguous, so the restore chunks, the paste map
        and the fallback all address "a frame's faces" as a contiguous slice.

        There is **no temporal landmark smoothing**: the single-track
        ``(frames, 10)`` model interpolates between *frames* - with a ragged
        face set that would blend the coordinates of different faces into
        each other.  Per-frame independent detection is the contracted
        behavior (per-face tracking is a future round).

        A ``None`` affine (the LMEDS failure of the vendored crop) drops its
        face from the batch before the crop - the vendored fallback would
        skip a ``None`` inverse gracefully (``:377-378``), and here there is
        no crop and no restored face for it to paste.

        Honest caveat (design section 3.2): the KEEP batch is flattened to
        ``(1, total_faces, 3, 512, 512)`` - the video path's optical-flow
        module sees the faces as frames, so a multi-face clip is outside the
        temporal prior's training condition (one face per frame); the
        quality on the owner's clips is their manual accept, per-frame
        sub-batching is the recorded fallback.

        The interrupt checks sit in every long loop: the detection per
        frame, the fused crop per face and per chunk, and the paste per
        frame (design section 2.2).
        """
        try:
            num_input_frames = image_sequence_tensor.shape[0]
            # Round 4 (D-R2.3.1) added the second convert-in of the paste
            # stage to the spend.
            progress = ProgressReporter(num_input_frames * 4)

            # The detection images are no longer retained (D-R2.1) - the
            # flag and the canvas stand in for the ``detection_frames`` list.
            needs_upscale = [False] * num_input_frames
            canvas_hw = None
            # The detection group size (D-R2.1): the staged detection
            # converts the group's frames, detects, deletes.
            group_size = max(1, max_clip_length)

            # Detection (design section 3.3): every face of every frame
            # becomes an entry.  ``read_image`` is called once per frame and
            # the conversion spend is counted upfront (the group slices sum
            # to the clip); the detection update lands after the pass.
            progress.update(num_input_frames)
            face_frames: list[int] = []
            face_landmarks: list[np.ndarray] = []
            logger.info("Detecting face landmarks")
            # The stage bar of the detection (the stage-bar design, D3 #5):
            # one tick per processed frame; a face-less frame ticks too.
            detection_bar = StageBar("Detecting face landmarks", num_input_frames)
            # The detection step calls into the vendored facelib: its console
            # writes are routed to the logger of the pack (D4).
            with capture_console_to_logger():
                for group_index in range(
                        (num_input_frames + group_size - 1) // group_size):
                    group_start = group_index * group_size
                    group_end = min(group_start + group_size, num_input_frames)
                    # A slice converts to bitwise-identical rows - the group
                    # is deleted before the next one is allocated (D-R2.1).
                    bgr_group = comfy_batch_to_bgr_u8(
                        image_sequence_tensor[group_start:group_end])
                    for i in range(group_start, group_end):
                        # The cancel poll (C1): ComfyUI's Cancel button raises
                        # here - a no-op outside ComfyUI (``src/keep_interrupt.py``).
                        keep_interrupt.check_interrupted()
                        img = bgr_group[i - group_start]
                        self.face_helper.clean_all()
                        # keep-fix5-implement1 (2026-11-03):
                        # ``check_gray=False`` skips the per-frame PIL
                        # ``is_gray`` - Stage 4 computes the gray flags on
                        # the device (``is_gray_batch``) and the paste
                        # fallback re-sets ``is_gray`` from them.
                        self.face_helper.read_image(img, check_gray=False)
                        # ``read_image`` replaces the input with an
                        # INTER_LINEAR upscale iff min(h, w) < 512 - the
                        # identity flags the frames the crop re-derives.
                        needs_upscale[i] = self.face_helper.input_img is not img
                        if canvas_hw is None:
                            canvas_hw = tuple(self.face_helper.input_img.shape[:2])
                        # All faces: the argument change **is** the fix (PM
                        # risk #1 of the plan) - ``only_keep_largest=True``
                        # would collapse the detection to the largest face
                        # before any downstream indexing could help.
                        self.face_helper.get_face_landmarks_5(
                            only_center_face=False, resize=640, eye_dist_threshold=5,
                            only_keep_largest=False)
                        for landmark in self.face_helper.all_landmarks_5:
                            face_frames.append(i)
                            # ``np.array`` materializes copies that survive the
                            # ``clean_all`` calls of the stages below.
                            face_landmarks.append(np.array(landmark, dtype=np.float64))
                        detection_bar.tick()
                    del bgr_group

            # The final line lands before the zero-faces early return below:
            # the early return carries the stage's bar line (D3 #5).
            detection_bar.finish()
            total_faces = len(face_landmarks)
            # Zero faces in the whole clip - the mirror of the all-NaN guard of
            # the single-face path (design section 3.2): the input comes back.
            if total_faces == 0:
                return image_sequence_tensor
            progress.update(num_input_frames)
            # The restore spend is counted in **faces** and the paste in frames
            # (design section 3.6): convert-in + detection + convert-in #2
            # spend ``3 * num_input_frames``, the restore ``total_faces`` and
            # the paste ``num_input_frames``.
            progress.total = num_input_frames * 4 + total_faces

            logger.info("Cropping and aligning faces")
            # The stage bar of the crop (the stage-bar design, D3 #6): ONE
            # tick after the estimate_affines call - the per-face LMEDS loop
            # lives in keep_gpu_ops.py and is opaque to the bar.  The unit
            # is **faces** (the pre-filter face count).
            crop_bar = StageBar(
                "Cropping and aligning faces", total_faces, unit="faces")
            # Crop (design section 3.3): the same vendored LMEDS call over all
            # the face landmarks (``keep_gpu_ops.estimate_affines``) and the
            # same cv2 warp per face - the affine once, reused by every paste.
            face_affines = list(estimate_affines(
                face_landmarks, self.face_helper.face_template))
            crop_bar.tick(len(face_landmarks))
            crop_bar.finish()
            # A ``None`` affine drops its face from the batch; with every affine
            # failed there is nothing to restore or paste.
            croppable = [k for k in range(total_faces) if face_affines[k] is not None]
            if not croppable:
                return image_sequence_tensor
            face_frames = [face_frames[k] for k in croppable]
            face_landmarks = [face_landmarks[k] for k in croppable]
            face_affines = [face_affines[k] for k in croppable]
            total_faces = len(croppable)

            # The fused, chunk-bounded crop + restore (D-R2.2): the same
            # global face chunks the restore always ran (``max_clip_length``
            # faces).  Per chunk the crops are warped in one batched device
            # op per source-shape group from a preallocated
            # ``max_clip_length``-bounded staging array - the involved
            # frames are derived once (at most ``max_clip_length + 1`` of
            # them, a frame's faces are contiguous) - and the restored rows
            # land in one preallocated whole-clip stack through the
            # row-independent per-chunk ``tensor2img_batch``.  The poll
            # cadence is today's, interleaved: the chunk top (the restore
            # loop's) and the per face (the staging row copy's).
            all_restored_faces_cv2 = np.empty((total_faces, 512, 512, 3), np.uint8)
            logger.info("Restoring faces with KEEP")
            # The stage bar of the KEEP restore (the stage-bar design, D3
            # #7): one tick of the chunk's face count per chunk; an OOM
            # leaves without the final line (the exception is the report).
            restore_bar = StageBar("Restoring faces with KEEP", total_faces, unit="faces")
            try:
                for start_idx in range(0, total_faces, max_clip_length):
                    keep_interrupt.check_interrupted()
                    end_idx = min(start_idx + max_clip_length, total_faces)
                    # The flattened faces-only batch: the time axis is the
                    # faces, not the frames (design section 3.2) - the
                    # frames without a face contribute no entry.  Every face
                    # of the chunk carries a valid affine (the
                    # ``croppable`` filter above), so the preallocated chunk
                    # tensor is fully written by the shape groups below.
                    current_clip_faces = torch.empty(
                        (end_idx - start_idx, 3, 512, 512),
                        dtype=torch.float32, device=self.device)
                    # The crop step calls into the vendored facelib: its
                    # console writes are routed to the logger of the pack
                    # (D4) - the derivation's ``read_image`` included.
                    with capture_console_to_logger():
                        # The chunk's involved frames in ONE group pass
                        # (D-R2.2.1, grouped): the conversion is elementwise,
                        # so the rows are bitwise the rows the per-row form
                        # produced - one host transfer per chunk instead of
                        # one per involved frame (a frame's faces are
                        # contiguous, so the involved frames stay
                        # ``max_clip_length``-bounded).
                        chunk_frames = self._detection_frames(
                            image_sequence_tensor,
                            face_frames[start_idx:end_idx], needs_upscale)
                        # One batched device warp per source-shape group
                        # replaces the per-face ``cv2.warpAffine``: the
                        # staging holds the detection frames (pre-warp) and
                        # stays bounded by the chunk (at most
                        # ``max_clip_length`` rows - in practice one group).
                        chunk_faces = {
                            k: chunk_frames[face_frames[k]]
                            for k in range(start_idx, end_idx)}
                        for (h, w), members in _group_by_shape(
                                chunk_faces, range(start_idx, end_idx)).items():
                            staged_u8 = np.empty((len(members), h, w, 3), np.uint8)
                            for row, k in enumerate(members):
                                # The per-face poll of today's crop loop (the
                                # cadence pin counts it; the chunk top
                                # polled above).
                                keep_interrupt.check_interrupted()
                                staged_u8[row] = chunk_faces[k]
                            batch = torch.from_numpy(staged_u8).permute(
                                0, 3, 1, 2).to(self.device)
                            del staged_u8
                            warped_u8 = crop_faces_batch(
                                batch, [face_affines[k] for k in members],
                                (512, 512))
                            del batch
                            current_clip_faces[torch.as_tensor(
                                [k - start_idx for k in members],
                                dtype=torch.long, device=self.device)] = (
                                bgr_u8_batch_to_normalized(warped_u8))
                            del warped_u8
                    del chunk_faces, chunk_frames
                    current_clip_faces = current_clip_faces.unsqueeze(0)
                    # The 1-entry chunk keeps the duplicate-to-2 trick, the
                    # progress counts faces.
                    # keep-fix5-implement1 (2026-11-03) wrapped this forward
                    # in FP16 autocast (plan Step 3.4); REGRESSION FIX
                    # (2026-11-03, owner-measured): on the owner's GPU the
                    # FP16 forward returned all-NaN - the restored faces came
                    # out black (KEEP-diag zero_frac=1.0), the parse masks on
                    # them collapsed to 0, and the video output became the
                    # raw input.  The FP16-fragile op is not identified, so
                    # the restore runs FP32 (the pre-round-13 behavior)
                    # until it is.
                    if current_clip_faces.shape[1] == 1:
                        current_clip_faces_for_net = torch.cat(
                            [current_clip_faces, current_clip_faces], dim=1)
                        clip_output_tensor = self.keep_net(
                            current_clip_faces_for_net, need_upscale=False)
                        kept_clip_tensor = clip_output_tensor[:, 0:1, ...]
                    else:
                        clip_output_tensor = self.keep_net(
                            current_clip_faces, need_upscale=False)
                        kept_clip_tensor = clip_output_tensor
                    restored_u8 = tensor2img_batch(
                        kept_clip_tensor.squeeze(0), min_max=(-1, 1))
                    all_restored_faces_cv2[start_idx:end_idx] = restored_u8
                    del (current_clip_faces, clip_output_tensor, kept_clip_tensor,
                         restored_u8)
                    progress.update(end_idx - start_idx)
                    restore_bar.tick(end_idx - start_idx)
            except torch.cuda.OutOfMemoryError:
                # The finished chunk's device transients die with the
                # exception; the release hands the cached blocks back before
                # ComfyUI sees the failure, and the exception propagates
                # unchanged (fail loudly).  On CPU the clause never matches.
                _soft_empty_cache()
                raise

            restore_bar.finish()

            # The faces of every frame, in detection order - the fused loop
            # keeps the entries ordered by (frame index, detection index).
            faces_by_frame: dict[int, list[int]] = {}
            for face_index, frame in enumerate(face_frames):
                faces_by_frame.setdefault(frame, []).append(face_index)

            # D-R2.3.1: convert-in #2 - the detection / crop / restore window
            # ran with no whole-clip uint8 BGR alive; the paste stage needs
            # the frames again.  Round 17 (the final stage's RAM): the frames
            # come from the **lazy source** - the rows are converted on
            # demand (the paste core per group, the fallback rows per
            # group), so the whole-clip uint8 stack is never built.
            # The converter is the module global, resolved here - after any
            # monkeypatch - so a spy sees every conversion of the run.
            frame_source = BgrFrameSource(
                image_sequence_tensor, convert=comfy_batch_to_bgr_u8)
            progress.update(num_input_frames)

            output_frames_cv2: list[np.ndarray] = []
            fallback_indices: list[int] = []
            device_canvas: torch.Tensor | None = None
            final_batch: torch.Tensor | None = None
            # Round 7 (the paste-budget repair): the paste's mode verdict reads
            # the **free** VRAM - ``device_canvas_budget`` (min of the 1 GiB cap
            # and the free share, ``keep_gpu_paste.py:233-251``) through
            # ComfyUI's ``get_free_memory(..., torch_free_too=True)``, which
            # counts what the torch cache can give back
            # (``keep_gpu_paste.py:178-192``).  Round 6 (Opt 6) deleted the
            # per-chunk ``_soft_empty_cache()`` of the restore loop above, so
            # the cached restore blocks hoard the memory the verdict reads: the
            # budget shrinks, the mode flips from ``full`` / ``chunked`` to
            # ``host``, and a clip that used to paste on the device takes the
            # host CPU assembly (the owner's paste-stage stall).  One release
            # here - after the last ``del`` of the restore loop and before the
            # verdict - restores the pre-Opt-6 budget without the per-chunk
            # sync cost Opt 6 removed (a single call per run, not one per
            # chunk; the ``finally`` / OOM releases stay).
            _soft_empty_cache()
            logger.info("Pasting faces and finalizing frames")
            # The stage bar of the paste (the stage-bar design, D3 #8; the
            # owner's fix 4, 2026-10-31): the bar is threaded into
            # ``_batched_paste_multiface`` - the batched rows tick inside
            # the paste core per completed chunk/group, the fallback rows
            # tick here in the loop - nobody ticks twice, so the bar totals
            # ``num_input_frames`` exactly (the sibling stage's fix-1.1
            # cadence).
            paste_bar = StageBar("Pasting faces and finalizing frames", num_input_frames)
            # The paste step calls into the vendored facelib: its console
            # writes are routed to the logger of the pack (D4).
            with capture_console_to_logger():
                # The gray flags of the frames (the vendored ``is_gray`` of
                # ``facelib/utils/misc.py:146``) - the same variances, without
                # the PIL round trip the per-frame paste made.
                gray_device = self.device if torch.device(self.device).type == "cuda" else None
                # Round 17: the flags are computed **per fallback group**,
                # with the group's frames (the whole-clip gray pass is gone)
                # - the eligible rows' flags are never read, so they stay
                # False.
                gray_flags = np.zeros(num_input_frames, dtype=bool)
                # The batched paste (design section 3.4): the eligible frames
                # paste on the device through the face-to-frame map - the faces,
                # the affines and the ROIs are face-indexed, the canvas, the
                # chunk buffer and the mapping stay frame-indexed.  The vendored
                # paste stays reachable for the non-default branches (the face
                # upsampler, the frames without a parse net and the frames with
                # an ineligible face); since the owner's fix 4 (2026-10-31)
                # ``draw_box`` rides the batch (the boxes land on the device
                # canvases) and the grayscale frames do too.
                batched_outputs = self._batched_paste_multiface(
                    frame_source, all_restored_faces_cv2, face_affines,
                    face_frames, face_landmarks, final_upscale_factor, draw_box,
                    gray_flags, canvas_hw=canvas_hw, stage_bar=paste_bar)

                # The single tail buffer (review F4): the mode booleans are
                # set at allocation - and in the device modes the mapping
                # becomes a plain index set, because its dict values are row
                # views that would keep the canvas / chunk buffer alive
                # until the return.
                canvas = batched_outputs.pop(paste_mod.DEVICE_CANVAS_KEY, None)
                chunks_out = batched_outputs.pop(paste_mod.DEVICE_OUTPUT_CHUNKS_KEY, None)
                # The fallback rows (round 17): the rows the batched paste
                # did not take - the mapping's keys are the eligible frames.
                # Their frames and their gray flags come from the lazy
                # source, **per group**.
                fallback_rows = [i for i in range(num_input_frames)
                                 if i not in batched_outputs]
                gray_groups = self._fallback_gray_groups(
                    frame_source, fallback_rows, gray_flags, gray_device,
                    max_clip_length)
                # The first group is pulled here: the gray pass belongs to
                # the processor's contract even when no row falls back (the
                # empty group).
                group_rows, group_frames = next(gray_groups)
                device_canvas_mode = False
                chunks_mode = False
                if canvas is not None:
                    if len(batched_outputs) == num_input_frames:
                        # every frame is batched: the canvas already holds the
                        # rows in order - alias it (no fresh buffer, no copy)
                        device_canvas = canvas
                    else:
                        # one device canvas of the whole batch: the batched indices
                        # are already in it (it is the group canvas, indexed copy),
                        # the vendored-fallback frames are uploaded as one stack
                        # below, and the single convert-out runs on the device
                        device_canvas = torch.empty(
                            (num_input_frames, *canvas.shape[1:]), dtype=torch.uint8,
                            device=self.device)
                        device_canvas[list(batched_outputs)] = canvas
                    device_canvas_mode = True
                    batched_outputs = set(batched_outputs)
                    canvas = None

                if chunks_out is not None:
                    # Round 18: the twin's buffer **is** the final buffer -
                    # the eligible frames landed in it (the core wrote them
                    # by index) and the vendored-fallback frames are
                    # written host-side into the same object below.  One
                    # whole-clip float32 buffer in the mixed case; the
                    # all-eligible case is the same object (the alias - no
                    # copy).  The old eligible-rows scatter is gone.
                    final_batch = chunks_out
                    chunks_mode = True
                    batched_outputs = set(batched_outputs)
                    chunks_out = None
                    if final_batch.shape[0] != num_input_frames:
                        raise RuntimeError(
                            f"the paste buffer has {final_batch.shape[0]} rows "
                            f"for {num_input_frames} input frames")
                for i in range(num_input_frames):
                    keep_interrupt.check_interrupted()
                    if i in batched_outputs:
                        # the device modes already hold the frame (the canvas
                        # or the chunk buffer); the host mode keeps the mapping
                        # value.  No bar tick here: the core ticked the row
                        # when its chunk/group completed (the owner's fix 4).
                        if not device_canvas_mode and not chunks_mode:
                            output_frames_cv2.append(batched_outputs[i])
                        progress.update(1)
                        continue
                    # Fallback path for the frames skipped by the batch: the
                    # loop owns this row's paste, so it ticks the bar.
                    paste_bar.tick(1)
                    # The row's frame (round 17): the lazy source's group
                    # that holds the row - the group is pulled when the row
                    # needs it, so only one group of uint8 frames is alive
                    # at a time.
                    while i not in group_rows:
                        if chunks_mode:
                            # Round 19 (FIX 2): the loop just advanced past a
                            # gray group - the group is complete, so its
                            # collected fallback frames convert **now** (per
                            # group, never the whole fallback) and are freed.
                            _flush_fallback_group(
                                final_batch, group_rows, output_frames_cv2,
                                fallback_indices)
                        group_rows, group_frames = next(gray_groups)
                    original_frame_bgr = group_frames[group_rows.index(i)]
                    # The background short-circuit (the owner's fix 4,
                    # 2026-10-31): a frame without a face - or whose every
                    # face lost its affine - has nothing to paste: skip the
                    # vendored helper entirely and return the background.
                    # The snippet's raw upscaler output cannot join the
                    # IMAGE batch when the vendored ``read_image`` upscaled
                    # the clip (a 256 frame among 512 rows) - the output
                    # size is the paste canvas, reached with the same
                    # LANCZOS4 resize the vendored paste applied to its
                    # no-face frames (adapted 2026-10-31, disclosed
                    # deviation).
                    frame_face_indices = faces_by_frame.get(i, [])
                    if not frame_face_indices or any(face_affines[k] is None for k in frame_face_indices):
                        final_pasted_frame = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                        canvas_h, canvas_w = paste_mod.paste_canvas_hw(
                            original_frame_bgr.shape[:2], final_upscale_factor, canvas_hw)
                        if final_pasted_frame.shape[0] != canvas_h or final_pasted_frame.shape[1] != canvas_w:
                            final_pasted_frame = cv2.resize(
                                final_pasted_frame, (canvas_w, canvas_h),
                                interpolation=cv2.INTER_LANCZOS4)
                        output_frames_cv2.append(final_pasted_frame)
                        fallback_indices.append(i)
                        progress.update(1)
                        continue

                    # The vendored fallback (design section 3.5): the helper gets
                    # **all** of the frame's faces at once - the restored faces
                    # and the crop matrices of the frame, one
                    # ``get_inverse_affine``, one ``paste_faces_to_input_image``
                    # - the reference composite by construction.
                    # ``draw_box=True`` draws one box per pasted face for free
                    # (the vendored per-face border masks,
                    # ``face_restoration_helper.py:428-438``).
                    self.face_helper.clean_all()
                    # Read once (design section 7.2): ``input_img`` is the
                    # image the vendored ``read_image`` produced - re-derived
                    # deterministically for the frames whose detection image
                    # was the upscale (D-R2.3.3), the converted row otherwise.
                    # The vendored paste sizes its canvas from it
                    # (``face_restoration_helper.py:362-370``).
                    if needs_upscale[i]:
                        self.face_helper.input_img = self._detection_frame(
                            image_sequence_tensor, i, needs_upscale)
                    else:
                        self.face_helper.input_img = original_frame_bgr
                    self.face_helper.is_gray = bool(gray_flags[i])
                    for k in frame_face_indices:
                        # ``tensor2img_batch`` rows are uint8 already - no copy
                        # (D-R2.3.4; ``add_restored_face`` does not mutate).
                        self.face_helper.add_restored_face(all_restored_faces_cv2[k])
                    # Affine once (design section 7.3): the crop stage computed
                    # the matrices - the paste reuses them, in the face order
                    # the restored faces were appended (the vendored paste
                    # pairs the two lists positionally).
                    self.face_helper.affine_matrices = [face_affines[k] for k in frame_face_indices]

                    bg_img_upscaled_by_model = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                    h, w, _ = original_frame_bgr.shape
                    target_h, target_w = int(h * final_upscale_factor), int(w * final_upscale_factor)
                    if bg_img_upscaled_by_model.shape[0] != target_h or bg_img_upscaled_by_model.shape[1] != target_w:
                        bg_img_final = cv2.resize(bg_img_upscaled_by_model, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        bg_img_final = bg_img_upscaled_by_model

                    self.face_helper.upscale_factor = final_upscale_factor
                    self.face_helper.get_inverse_affine(None)
                    final_pasted_frame = self.face_helper.paste_faces_to_input_image(
                        upsample_img=bg_img_final,
                        draw_box=draw_box,
                        face_upsampler=self.face_upscale_model
                    )
                    output_frames_cv2.append(final_pasted_frame)
                    fallback_indices.append(i)
                    progress.update(1)

            paste_bar.finish()

            # The lazy source and its last group are the paste stage's
            # scratch (round 17) - release them before the convert-out
            # tails (D-R2.3.6).
            # Round 19 (FIX 2): in the chunked device mode the last gray
            # group's collected fallback frames convert **here** - before
            # the group bookkeeping is released (the flush reads the last
            # group's rows).  The loop already flushed every group it
            # advanced past; the last group is the only one still pending.
            if chunks_mode:
                _flush_fallback_group(
                    final_batch, group_rows, output_frames_cv2,
                    fallback_indices)
            del gray_groups, group_rows, group_frames
            del frame_source
            # The batched convert-out (design section 7.8): one stack and one
            # transfer, where the per-frame loop made a tensor per frame.
            if chunks_mode:
                # Round 19 (FIX 2): the chunked device mode converts the
                # vendored-fallback frames **per gray group** - the pull
                # site flushed every group the loop advanced past and the
                # flush above wrote the last group's rows.  The
                # whole-fallback stack and the whole-stack float32
                # conversion are gone: the peak is one group's stack plus
                # its conversion transient (<= max_clip_length rows),
                # independent of the fallback count.  The other output modes
                # keep the whole-stack form below (they are not the
                # production chunked path).
                return final_batch
            if device_canvas_mode:
                # F1: the fallback frames are still host arrays - one stacked
                # upload, never one transfer per frame; the convert-out runs on
                # the device in budget-sized sub-chunks
                if output_frames_cv2:
                    device_canvas[fallback_indices] = torch.from_numpy(
                        np.stack(output_frames_cv2)).to(self.device)
                return paste_mod.comfy_convert_out(
                    device_canvas, budget=paste_mod.device_canvas_budget(self.device))
            if not output_frames_cv2:
                return image_sequence_tensor
            return bgr_u8_batch_to_comfy(np.stack(output_frames_cv2))
        finally:
            # D-R2.4: the helper's stage arrays are the run's scratch - one
            # detected frame used to survive the run in ``input_img``.
            self.face_helper.clean_all()
            self.face_helper.input_img = None
            # P4.6 (Opt 6): the per-run release - the chunk-boundary clears
            # are gone (see ``process_image_sequence``).  On the multi-face
            # dispatch path this finally runs in addition to the outer one:
            # two idempotent releases per run, cheap when there is nothing
            # to release.
            _soft_empty_cache()
