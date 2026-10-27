"""The face processor of the KEEP pack (P3.2).

The engine that detects, aligns, restores and pastes the faces of an image
or an image sequence.  It moved here from ``modules/keep_processor.py`` in
P3.2.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from .keep_console import capture_console_to_logger
from .keep_image_utils import (
    bgr_u8_batch_to_comfy,
    bgr_u8_batch_to_normalized,
    comfy_batch_to_bgr_u8,
    comfy_image_to_cv2,
    cv2_to_comfy_image,
    resize_bgr_u8,
)
from .keep_gpu_ops import (
    estimate_affines,
    inverse_affines,
    tensor2img_batch,
)
from . import keep_gpu_paste as paste_mod
from .keep_gpu_paste import (
    PARSE_FACE_SIZE,
    batched_bg_frames,
    is_gray_batch,
    paste_faces_batch,
)
from .keep_logger import logger
from .keep_progress import ProgressReporter

# Imports from vendored basicsr and facelib
#
# P2.1: the vendored ``basicsr.utils.video_util`` module is deleted (it ran
# a pip install of ``ffmpeg-python`` at import time).  Its video helpers are
# not importable any more and the dead video path of the processor that used
# them was removed with it (P4.7).  The image / image-sequence paths are
# unaffected.
#
# P2.7: the dummy image helpers are gone - a broken vendored dependency must
# fail loudly instead of returning ``None`` images.
try:  # the package import (the node package is the parent package)
    from ..modules.deps.basicsr.utils.img_util import img2tensor, tensor2img
    from ..modules.deps.facelib.utils.misc import is_gray
except ImportError:  # the flat import (the repository root on sys.path)
    from modules.deps.basicsr.utils.img_util import img2tensor, tensor2img
    from modules.deps.facelib.utils.misc import is_gray


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


def _gaussian_smooth(values: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Smooth a ``(frames, values)`` array along the frame axis.

    The torch-native replacement of the pack's former ``gaussian_filter1d``
    smoothing (the temporal landmark filter): the kernel is the reference
    implementation's own formula (``exp(-x**2 / (2*sigma**2))`` normalized
    by its sum, with the radius ``int(truncate * sigma + 0.5)`` of the
    ``truncate=4.0`` default), the boundary extension is symmetric (the
    reference ``mode="reflect"``) and the correlation runs as a float64
    depthwise ``conv1d`` - one channel per value column.  The output keeps
    the dtype of the input; the measured difference against the reference is
    ~4e-14 in float64 (``tests/test_image_op_equivalence.py``).
    """
    source = np.asarray(values)
    radius = int(4.0 * sigma + 0.5)
    positions = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-(positions**2) / (2.0 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    # ``mode="reflect"`` of the reference == numpy ``mode="symmetric"``: the
    # edges are mirrored with the edge value repeated.
    padded = np.pad(source, ((radius, radius), (0, 0)), mode="symmetric")
    signal = torch.from_numpy(padded).double().T.unsqueeze(0)
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


class KEEPFaceProcessor:
    def __init__(self, model_pack):
        self.keep_net = model_pack.keep_net
        self.face_helper = model_pack.face_helper
        self.bg_upscale_model = model_pack.bg_upscale_model
        self.face_upscale_model = model_pack.face_upscale_model
        self.device = model_pack.device
        self.model_type_str = model_pack.model_type_str

    def _run_upscaler(self, model, cv2_image):
        """Helper to run a comfy-native upscaler on a cv2 image."""
        from comfy.utils import tiled_scale  # lazy: the pack imports without ComfyUI

        if model is None:
            return cv2_image

        img_tensor = cv2_to_comfy_image(cv2_image).to(self.device)
        img_tensor_bchw = img_tensor.movedim(-1, -3)
        s = tiled_scale(img_tensor_bchw, lambda a: model.model(a), tile_x=512, tile_y=512, overlap=64, upscale_amount=model.scale)
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return comfy_image_to_cv2(s)

    def _batched_paste(self,
                       input_frames_bgr: np.ndarray,
                       restored_faces: np.ndarray,
                       affine_matrices: list,
                       avg_landmarks_sequence,
                       final_upscale_factor: float,
                       draw_box: bool,
                       gray_flags: np.ndarray,
                       canvas_hw: tuple[int, int] | None = None) -> dict:
        """The batched paste of the default branch (design section 7.7).

        The frames the batched paste can handle - a valid affine, a
        restored face of the ParseNet size, no box, no face upsampler, a
        color frame and an available parse net - are pasted on the device:
        the chunked ParseNet forward, the batched parse mask, the batched
        inverse warp and the batched blend.

        Every other frame keeps the vendored ``paste_faces_to_input_image``
        (design section 7.7, step 6): the box, the face upsampler, the
        grayscale frames, the frames without a face and a face helper
        without a parse net.

        Returns:
            ``{frame index: the pasted frame}`` - the frames that are not
            in the mapping take the vendored path.
        """
        parse_net = getattr(self.face_helper, "face_parse", None)
        if parse_net is None or not getattr(self.face_helper, "use_parse", False):
            return {}
        if draw_box or self.face_upscale_model is not None:
            return {}
        if avg_landmarks_sequence is None:
            return {}

        eligible = []
        for i in range(len(restored_faces)):
            if i >= len(affine_matrices) or affine_matrices[i] is None:
                continue
            if i >= len(avg_landmarks_sequence) or np.isnan(avg_landmarks_sequence[i]).any():
                continue
            if gray_flags[i]:
                continue
            if restored_faces[i].shape[:2] != (PARSE_FACE_SIZE, PARSE_FACE_SIZE):
                continue
            eligible.append(i)
        if not eligible:
            return {}

        # The inverse warps of the paste (the vendored ``get_inverse_affine``:
        # ``cv2.invertAffineTransform(affine) * upscale_factor``) and the
        # background of the batch - one ``tiled_scale`` call for the whole
        # group instead of one per frame.
        inverses = inverse_affines(
            [affine_matrices[i] for i in eligible], upscale_factor=final_upscale_factor)
        # the device-output mode (F1 of the paste-step round): the background
        # stays on the device and the pasted stack is returned as one uint8
        # tensor instead of N host arrays - the assembly of the output stack
        # (the per-frame download, the canvas copies, the ``np.stack`` and
        # the convert-out's host math) moves to one indexed copy plus one
        # convert-out.  The CPU device and an over-budget canvas keep the
        # host output byte-for-byte.
        canvas_shape = None
        on_device = self.device.type == "cuda"
        background_stacked = np.stack([input_frames_bgr[i] for i in eligible])
        height, width = background_stacked.shape[1:3]
        target_h, target_w = int(height * final_upscale_factor), int(width * final_upscale_factor)
        canvas_h, canvas_w = target_h, target_w
        if canvas_hw is not None:
            canvas_h = max(target_h, int(canvas_hw[0] * final_upscale_factor))
            canvas_w = max(target_w, int(canvas_hw[1] * final_upscale_factor))
        canvas_bytes = len(eligible) * canvas_h * canvas_w * 3
        if on_device and canvas_bytes <= paste_mod.PASTE_DEVICE_STACK_MAX_BYTES:
            canvas_shape = (canvas_h, canvas_w)
        if canvas_shape is not None:
            background = batched_bg_frames(
                background_stacked, final_upscale_factor, self.bg_upscale_model,
                self.device, canvas_hw=canvas_hw, return_device=True)
            canvas = paste_faces_batch(
                faces_bgr=np.stack([restored_faces[i] for i in eligible]),
                inverse_affines=inverses,
                background_bgr=background,
                device=self.device,
                parse_net=parse_net,
                device_output=True)
            return {**dict(zip(eligible, list(canvas))), paste_mod.DEVICE_CANVAS_KEY: canvas}
        background = batched_bg_frames(
            background_stacked, final_upscale_factor, self.bg_upscale_model,
            self.device, canvas_hw=canvas_hw)
        pasted = paste_faces_batch(
            faces_bgr=np.stack([restored_faces[i] for i in eligible]),
            inverse_affines=inverses,
            background_bgr=background,
            device=self.device,
            parse_net=parse_net)
        return dict(zip(eligible, pasted))

    @torch.no_grad()
    def process_image(self, cv2_image_orig: np.ndarray, final_upscale_factor: float, has_aligned: bool,
                      only_center_face: bool, draw_box: bool):
        
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
        
        self.face_helper.add_restored_face(restored_faces_cv2[0].astype('uint8'))

        if not has_aligned:
            bg_img_upscaled_by_model = self._run_upscaler(self.bg_upscale_model, img_for_pasting)

            h, w, _ = img_for_pasting.shape
            target_h, target_w = int(h * final_upscale_factor), int(w * final_upscale_factor)
            if bg_img_upscaled_by_model.shape[0] != target_h or bg_img_upscaled_by_model.shape[1] != target_w:
                bg_img_final = cv2.resize(bg_img_upscaled_by_model, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                bg_img_final = bg_img_upscaled_by_model

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

        # P4.7: the V3 progress API replaces the V1 ``comfy.utils.ProgressBar``
        # (``src/keep_progress.py`` reports through the execution API).
        progress = ProgressReporter(num_input_frames * 4)

        # The batched convert-in (design section 7.1): the elementwise math
        # of the per-frame conversion on the whole stack - one host transfer
        # instead of one per frame.
        input_frames_bgr = comfy_batch_to_bgr_u8(image_sequence_tensor)
        progress.update(num_input_frames)
        
        avg_landmarks_sequence = None
        # The affine matrices of the frames (design section 7.3): the crop
        # computes them once, the paste loop reuses them.
        affine_matrices: list = [None] * num_input_frames
        # The images the detector saw (design section 7.2): the crop warps
        # them - no frame is read twice.
        detection_frames: list = []
        if not has_aligned_frames:
            logger.info("Detecting face landmarks")
            raw_landmarks_list_video = []
            # The detection step calls into the vendored facelib: its console
            # writes are routed to the logger of the pack (D4).
            with capture_console_to_logger():
                for i in range(num_input_frames):
                    img = input_frames_bgr[i]
                    self.face_helper.clean_all()
                    self.face_helper.read_image(img)
                    # Read once (design section 7.2): the detection read is
                    # the only ``read_image`` of the frame - the crop warps
                    # the image the detector saw (until now the crop and the
                    # paste loops re-read the very same frame).
                    detection_frames.append(self.face_helper.input_img)
                    num_det_faces = self.face_helper.get_face_landmarks_5(
                        only_center_face=only_center_face, resize=640, eye_dist_threshold=5,
                        only_keep_largest=True)
                    if num_det_faces >= 1:
                        raw_landmarks_list_video.append(self.face_helper.all_landmarks_5[0].reshape((10,)))
                    else:
                        raw_landmarks_list_video.append(np.array([np.nan] * 10))
            raw_landmarks_np_video = np.array(raw_landmarks_list_video)
            if np.all(np.isnan(raw_landmarks_np_video)):
                return image_sequence_tensor 
            for i_lm_coord in range(raw_landmarks_np_video.shape[1]):
                raw_landmarks_np_video[:, i_lm_coord] = interpolate_sequence(raw_landmarks_np_video[:, i_lm_coord])
            avg_landmarks_sequence = _gaussian_smooth(raw_landmarks_np_video, sigma=5.0).reshape(num_input_frames, 5, 2)
            progress.update(num_input_frames)
        else:
             progress.update(num_input_frames)

        logger.info("Cropping and aligning faces")
        # The crop step calls into the vendored facelib: its console writes
        # are routed to the logger of the pack (D4).
        with capture_console_to_logger():
            if not has_aligned_frames:
                # Affine once (design section 7.3): the matrices of the
                # vendored ``align_warp_face`` - the same
                # ``cv2.estimateAffinePartial2D(..., LMEDS)`` call - are
                # computed here for the crop and reused by the paste loop,
                # which recomputed them until now.  A ``None`` matrix (the
                # LMEDS failure of the vendored helper) keeps the zero-face
                # and bg-only handling of the no-landmark frames.
                valid_indices = [i for i in range(num_input_frames)
                                 if not np.isnan(avg_landmarks_sequence[i]).any()]
                estimated_affines = estimate_affines(
                    [avg_landmarks_sequence[i] for i in valid_indices],
                    self.face_helper.face_template)
                for i, affine_matrix in zip(valid_indices, estimated_affines):
                    affine_matrices[i] = affine_matrix
                    if affine_matrix is None:
                        logger.warning(
                            f"Warning: Could not estimate affine matrix for landmark set 0 "
                            f"(frame {i}). Skipping this face."
                        )
                # The crop (design section 7.4, tolerance decision D2): the
                # warp stays on cv2 - the batched ``grid_sample`` of the
                # device drifts up to 8/255 against cv2's fixed-point
                # sampling on high-frequency content and the crop is a
                # uint8-bound op (the KEEP input has to stay bit-exact).
                # The warp source is the image the detector saw - the raw
                # frame, or the frame ``read_image`` upscaled for
                # min(h, w) < 512.  The img2tensor + normalize tail stays
                # batched on the device.
                croppable = [i for i in valid_indices if affine_matrices[i] is not None]
                # The frames without a face carry the base's neutral crop
                # (``torch.zeros(3, 512, 512) * 0.5`` of the base - 0.5 in
                # the normalized scale, not 0).
                cropped_batch = torch.full(
                    (num_input_frames, 3, 512, 512), 0.5,
                    dtype=torch.float32, device=self.device)
                if croppable:
                    cropped_u8 = np.stack(
                        [
                            cv2.warpAffine(
                                detection_frames[i], affine_matrices[i], (512, 512),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(135, 133, 132),
                            )
                            for i in croppable
                        ]
                    )
                    croppable_index = torch.as_tensor(
                        croppable, dtype=torch.long, device=self.device)
                    warped = bgr_u8_batch_to_normalized(
                        torch.from_numpy(cropped_u8).permute(0, 3, 1, 2).to(self.device))
                    cropped_batch[croppable_index] = warped
                    del warped, cropped_u8
            else:
                # The frames are the faces: the resize of the aligned path
                # stays per frame (no batched resize exists - the design
                # section 7.7 keeps the per-frame resizes), the img2tensor +
                # normalize tail runs once for the batch.
                resized_faces = [resize_bgr_u8(frame, (512, 512)) for frame in input_frames_bgr]
                aligned_u8 = torch.from_numpy(np.stack(resized_faces)).permute(0, 3, 1, 2).to(self.device)
                cropped_batch = bgr_u8_batch_to_normalized(aligned_u8)
                del aligned_u8
            batched_cropped_faces = cropped_batch.unsqueeze(0)

        temp_restored_face_tensors = []
        logger.info("Restoring faces with KEEP")
        for start_idx in range(0, num_input_frames, max_clip_length):
            end_idx = min(start_idx + max_clip_length, num_input_frames)
            current_clip_faces = batched_cropped_faces[:, start_idx:end_idx, ...]
            if current_clip_faces.shape[1] == 0: continue
            if current_clip_faces.shape[1] == 1:
                 current_clip_faces_for_net = torch.cat([current_clip_faces, current_clip_faces], dim=1)
                 clip_output_tensor = self.keep_net(current_clip_faces_for_net, need_upscale=False)
                 temp_restored_face_tensors.append(clip_output_tensor[:, 0:1, ...])
            else:
                 clip_output_tensor = self.keep_net(current_clip_faces, need_upscale=False)
                 temp_restored_face_tensors.append(clip_output_tensor)
            progress.update(end_idx - start_idx)
            # P4.6: the chunk boundary - the cached blocks of the finished
            # chunk are released before the next chunk is allocated.
            _soft_empty_cache()

        if not temp_restored_face_tensors:
            return image_sequence_tensor
        all_restored_face_tensors = torch.cat(temp_restored_face_tensors, dim=1).squeeze(0)
        # The batched tensor2img (design section 7.6): the vendored
        # elementwise chain on the whole stack - one host transfer instead of
        # one per frame, the same pixels (the op order is unchanged).
        all_restored_faces_cv2 = tensor2img_batch(all_restored_face_tensors, min_max=(-1, 1))
        del batched_cropped_faces, all_restored_face_tensors, temp_restored_face_tensors
        _soft_empty_cache()

        output_frames_cv2: list[np.ndarray] = []
        fallback_indices: list[int] = []
        device_canvas: torch.Tensor | None = None
        logger.info("Pasting faces and finalizing frames")
        # The paste step calls into the vendored facelib: its console writes
        # are routed to the logger of the pack (D4).
        with capture_console_to_logger():
            # The gray flags of the frames (the vendored ``is_gray`` of
            # ``facelib/utils/misc.py:146``) - the same variances, without
            # the PIL round trip the per-frame paste made.  On CUDA they run
            # on the device (the paste-stage CPU optimization round, section
            # 4.2 - the host pass was the largest host-side item of the
            # stage); the CPU device keeps the numpy computation.
            gray_device = self.device if torch.device(self.device).type == "cuda" else None
            gray_flags = is_gray_batch(input_frames_bgr, device=gray_device)
            # The batched paste (design section 7.7): the default branch of
            # the paste stage runs on the device - the chunked ParseNet
            # forward, the batched parse mask, the batched inverse warp and
            # the batched blend.  The vendored paste stays reachable for
            # the non-default branches (``draw_box``, the face upsampler,
            # the grayscale frames and a helper without a parse net).
            batched_outputs = {}
            if not has_aligned_frames:
                # The canvas of the paste (F1): the image the vendored
                # ``read_image`` produced - the raw frame, or the frame
                # upscaled for min(h, w) < 512.  The vendored paste sizes
                # its canvas from ``self.input_img``
                # (``face_restoration_helper.py:362-370``).
                canvas_hw = tuple(detection_frames[0].shape[:2])
                batched_outputs = self._batched_paste(
                    input_frames_bgr, all_restored_faces_cv2, affine_matrices,
                    avg_landmarks_sequence, final_upscale_factor, draw_box, gray_flags,
                    canvas_hw=canvas_hw)

            canvas = batched_outputs.pop(paste_mod.DEVICE_CANVAS_KEY, None)
            if canvas is not None:
                # one device canvas of the whole batch: the batched indices
                # are already in it (it is the group canvas, indexed copy),
                # the vendored-fallback frames are uploaded as one stack
                # below, and the single convert-out runs on the device
                device_canvas = torch.empty(
                    (num_input_frames, *canvas.shape[1:]), dtype=torch.uint8,
                    device=self.device)
                device_canvas[list(batched_outputs)] = canvas
            for i in range(num_input_frames):
                original_frame_bgr = input_frames_bgr[i]
                if i in batched_outputs:
                    # the device mode already holds the frame in the canvas;
                    # the host mode keeps the mapping value
                    if device_canvas is None:
                        output_frames_cv2.append(batched_outputs[i])
                    progress.update(1)
                    continue
                if i >= len(all_restored_faces_cv2):
                    final_pasted_frame = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                    output_frames_cv2.append(final_pasted_frame)
                    fallback_indices.append(i)
                    continue
                restored_face_patch_cv2 = all_restored_faces_cv2[i]
                self.face_helper.clean_all()
                # Read once (design section 7.2): the paste loop of today ran
                # a third ``read_image`` per frame.  On the uint8 BGR frames
                # of the batch conversion it only sets ``input_img`` (the
                # guard of the vendored paste) and ``is_gray`` (the gray
                # conversion of ``add_restored_face``) - both are set here
                # from the very same frame.  ``input_img`` is the image the
                # vendored ``read_image`` produced (the frame upscaled for
                # min(h, w) < 512): the vendored paste sizes its canvas
                # from it (``face_restoration_helper.py:362-370``) and the
                # base called ``read_image`` in the paste loop.
                self.face_helper.input_img = (
                    detection_frames[i] if detection_frames else original_frame_bgr)
                self.face_helper.is_gray = bool(gray_flags[i])
                self.face_helper.add_restored_face(restored_face_patch_cv2.astype('uint8'))

                if not has_aligned_frames:
                    if avg_landmarks_sequence is None or i >= len(avg_landmarks_sequence) or np.isnan(avg_landmarks_sequence[i]).any():
                        final_pasted_frame = self._run_upscaler(self.bg_upscale_model, original_frame_bgr)
                        output_frames_cv2.append(final_pasted_frame)
                        continue
                    # Affine once (design section 7.3): the crop step
                    # computed the matrix - the paste loop of today ran a
                    # second ``align_warp_face`` to compute it again.
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
            
        # The batched convert-out (design section 7.8): one stack and one
        # transfer, where the per-frame loop made a tensor per frame.
        if device_canvas is not None:
            # F1: the fallback frames are still host arrays - one stacked
            # upload, never one transfer per frame
            if output_frames_cv2:
                device_canvas[fallback_indices] = torch.from_numpy(
                    np.stack(output_frames_cv2)).to(self.device)
            return bgr_u8_batch_to_comfy(device_canvas)
        if not output_frames_cv2:
             return image_sequence_tensor
        final_batch_tensor = bgr_u8_batch_to_comfy(np.stack(output_frames_cv2))
        return final_batch_tensor