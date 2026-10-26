"""Profiling probe for the GPU video pipeline work (stage timings + micro benchs).

This tool changes no production code.  It times the stages of
``src/keep_processor.py::process_image_sequence`` on synthetic frames, in the
embedded interpreter, so the implementation order of the GPU batching follows
measured numbers instead of guesses.

The report holds **two** measurements:

* ``baseline_clone`` - the **base implementation** (``2e8d9b9``) as a clone
  inside this file (the ``stage_clone`` function).  The base code is gone
  from ``src/``, but the clone still runs against the untouched vendored
  helpers: it is the **control** of the session.  The same-session control
  proves that the machine did not drift between the before run and the after
  run - the before/after deltas are read against it, not against a number
  measured days earlier.
* ``pipeline`` - the **shipped implementation** (the batched path of
  ``src/keep_processor.py``).  Its stages are timed by wrapping its ops
  (the ``_StageProbe`` class); every wrapper syncs the device exactly like
  the stage timer of the clone, so the two stage tables are comparable.

The real model weights (KEEP, RetinaFace, ParseNet) are NOT required: the
probe builds the **real vendored architectures with random weights**.  Random
weights change the output pixels but not the FLOPs, so every stage timing is
representative; the pixels are garbage by construction and never compared.

Stages (mirroring ``process_image_sequence`` of base ``2e8d9b9``):

  s1 convert_in   per-frame ``comfy_image_to_cv2`` loop          (:254-258)
  s2 detect       per-frame read_image + get_face_landmarks_5    (:266-277)
                  (+ landmark interpolation + gaussian smoothing (:278-283))
  s3 crop_align   per-frame read_image + align_warp_face +
                  img2tensor + normalize + stack + to(device)    (:288-319)
  s4 keep_infer   the KEEP chunk loop (chunks of max_clip_length) (:321-336)
  s5 tensor2img   per-frame ``tensor2img`` loop                  (:341)
  s6 paste        per-frame paste loop: read_image +
                  align_warp_face (again) + get_inverse_affine +
                  paste_faces_to_input_image (ParseNet + warps)  (:345-400)
  s7 convert_out  per-frame ``cv2_to_comfy_image`` + cat         (:402-405)
  e2e             one full ``KEEPFaceProcessor.process_image_sequence`` call

The crop of the **shipped** pipeline is timed through the op that actually
runs it: the per-frame ``cv2.warpAffine`` of decision D2 (the uint8-bound
crop stayed on cv2 - the gray ``borderValue=(135, 133, 132)`` discriminates
it from the warps of the vendored paste) or, on a tree that uses the batched
crop, ``crop_faces_batch``.  The probe adapts to both shapes; an op that is
not in the namespace of its owner is reported in ``unwrapped_ops`` and never
silently ignored.

Micro benchmarks (old op loop vs the batched GPU candidate):

  m1 crop warp      cv2.warpAffine loop vs one batched F.grid_sample
  m2 parse          per-frame ParseNet forward vs one batched forward
  m3 parse blur     2x cv2.GaussianBlur(101,101,s=11) loop vs batched torch conv
  m4 paste warp     per-frame inverse warpAffine vs batched F.grid_sample
  m5 conversions    per-frame tensor<->numpy loops vs one batched transfer

Run (embedded interpreter):

    # the before run (the base implementation was in src/ then)
    COMFYUI_ROOT=<ComfyUI tree> python tools/prof_gpu_video_pipeline.py \
        --device cuda --frames 48 --json .dev/prof-before-cuda.json

    # the after run (the shipped batched pipeline)
    COMFYUI_ROOT=<ComfyUI tree> python tools/prof_gpu_video_pipeline.py \
        --device cuda --frames 48 --json .dev/prof-after-cuda.json
    COMFYUI_ROOT=<ComfyUI tree> python tools/prof_gpu_video_pipeline.py \
        --device cpu --frames 48 --skip-micro --json .dev/prof-after-cpu.json

Exit code 0 when the probe ran; the report content is the evidence.  The
after run measures the **control** (the base clone) and the **shipped
pipeline** in the same session, so the before/after delta is read against a
number of the same day.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

REPO_ROOT = Path(__file__).resolve().parent.parent

FRAME_H, FRAME_W = 720, 1280
KEEP_CHUNK = 20  # the max_clip_length default (the quality contract)

#: The gray ``BORDER_CONSTANT`` value of the vendored face crop
#: (``face_restoration_helper.py:326-328``), in BGR order.  It is the
#: **discriminator** of the crop warp: the warps of the vendored paste use
#: the default black border.
CROP_BORDER_BGR = (135, 133, 132)


def _is_crop_border(value) -> bool:
    """Is this ``borderValue`` the gray border of the vendored crop?"""
    if value is None:
        return False
    try:
        flat = np.asarray(value).reshape(-1)
        return tuple(int(v) for v in flat) == CROP_BORDER_BGR
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def resolve_comfyui_root(explicit: str | None) -> str | None:
    """The ComfyUI source tree to import comfy/ from (arg > env; no fallback)."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    env_root = os.environ.get("COMFYUI_ROOT")
    if env_root:
        candidates.append(env_root)
    for candidate in candidates:
        if candidate and (Path(candidate) / "comfy" / "model_management.py").is_file():
            return str(Path(candidate))
    return None


class Timer:
    """Cumulative stage timer with torch.cuda synchronization.

    Two clocks per stage: the **wall** time (``time.perf_counter``) and the
    **process CPU** time (``time.process_time()`` - the CPU time of all the
    threads of the process).  The owner's complaint is about the CPU *load*,
    and the two numbers are very different here: an OpenCV ``parallel_for``
    warp burns every core for the whole of its wall time, so its process CPU
    time is a multiple of its wall time.
    """

    def __init__(self, device: str):
        self.device = device
        self.totals: dict[str, float] = {}
        #: the process CPU time per stage (the sum over the threads)
        self.cpu_totals: dict[str, float] = {}

    def sync(self):
        if self.device == "cuda":
            import torch

            torch.cuda.synchronize()

    def measure(self, name: str, fn):
        self.sync()
        start = time.perf_counter()
        cpu_start = time.process_time()
        result = fn()
        # the process CPU time of the **call** only: the device
        # synchronisation waits with the driver's spin threads running, and
        # that spin would otherwise be charged to the stage
        cpu_elapsed = time.process_time() - cpu_start
        self.sync()
        self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - start
        self.cpu_totals[name] = self.cpu_totals.get(name, 0.0) + cpu_elapsed
        return result

    def table(self, num_frames: int) -> list[dict]:
        rows = []
        for name, seconds in self.totals.items():
            cpu_seconds = self.cpu_totals.get(name, 0.0)
            rows.append(
                {
                    "stage": name,
                    "total_s": round(seconds, 3),
                    "per_frame_ms": round(seconds * 1000.0 / num_frames, 2),
                    "cpu_total_s": round(cpu_seconds, 3),
                    "cpu_per_frame_ms": round(cpu_seconds * 1000.0 / num_frames, 2),
                    # the CPU load of the stage: the process CPU time per
                    # second of wall time (1.0 = one core, 8.0 = every core)
                    "cpu_cores": round(cpu_seconds / seconds, 2) if seconds > 0 else None,
                    "share_pct": None,  # filled by the caller over the known stages
                }
            )
        return rows


def build_synthetic_landmarks(face_template: np.ndarray, num_frames: int, rng):
    """One face per frame: the 512-template landmarks at a jittered position."""
    landmarks = np.zeros((num_frames, 5, 2), dtype=np.float64)
    for i in range(num_frames):
        scale = 0.72 + rng.uniform(-0.01, 0.01)
        center = np.array([FRAME_W * 0.5 + rng.uniform(-8, 8), FRAME_H * 0.46 + rng.uniform(-6, 6)])
        landmarks[i] = (face_template - 256.0) * scale + center
    return landmarks


def bbox_from_landmarks(landmarks_5x2: np.ndarray) -> np.ndarray:
    """The (1, 15) RetinaFace decode row: box + score + 5 (x, y) landmarks."""
    x_min, y_min = landmarks_5x2.min(axis=0) - 40.0
    x_max, y_max = landmarks_5x2.max(axis=0) + 60.0
    row = [x_min, y_min, x_max, y_max, 0.99]
    for point in landmarks_5x2:
        row.extend([point[0], point[1]])
    return np.array([row], dtype=np.float32)


def make_synthetic_detector(device: str, landmarks: np.ndarray):
    """A detector whose forward is the *real* RetinaFace (random weights).

    ``get_face_landmarks_5`` calls ``detect_faces(input_img, 0.97)``; the
    synthetic wrapper runs the real forward (the timed cost) and then returns
    one synthetic detection row so the pipeline continues with real landmarks.
    """
    import torch

    from modules.deps.facelib.detection.retinaface.retinaface import RetinaFace

    retina = RetinaFace(network_name="resnet50", half=False).eval().to(device)

    class SyntheticDetector:
        def __init__(self):
            self.frame_idx = 0

        def detect_faces(self, image, conf_threshold=0.8, **_kwargs):
            try:
                retina.detect_faces(image, conf_threshold=conf_threshold)
            except Exception:  # random weights may decode nothing - cost already paid
                pass
            row = bbox_from_landmarks(landmarks[self.frame_idx % len(landmarks)])
            self.frame_idx += 1
            return row

    return SyntheticDetector()


def make_synthetic_parser(device: str):
    """The real ParseNet architecture, random weights."""
    from modules.deps.facelib.parsing.parsenet import ParseNet

    return ParseNet(in_size=512, out_size=512).eval().to(device)


def make_synthetic_keep(device: str):
    """The real KEEP architecture, random weights (flownet owned by the arch)."""
    from src import keep_model_configs
    import modules.deps.basicsr.archs.keep_arch  # noqa: F401 - fills the registry

    arch_class = keep_model_configs.ARCH_REGISTRY.get("KEEP")
    if arch_class is None:
        raise RuntimeError("the KEEP arch did not register")
    architecture = dict(keep_model_configs.KEEP_MODEL_CONFIGS["KEEP"]["architecture"])
    architecture["flownet_path"] = None
    return arch_class(**architecture).eval().to(device)


def build_pack(device: str, landmarks: np.ndarray):
    """A stand-in KEEPModelPack: real vendored helpers + random-weight archs."""
    import torch

    import src.keep_processor as processor_mod
    from src.keep_processor import KEEPFaceProcessor

    helper_module_name = "modules.deps.facelib.utils.face_restoration_helper"
    import importlib

    helper_module = importlib.import_module(helper_module_name)

    detector = make_synthetic_detector(device, landmarks)
    parser = make_synthetic_parser(device)

    original_init_det = helper_module.init_detection_model
    original_init_parse = helper_module.init_parsing_model
    helper_module.init_detection_model = lambda *a, **k: detector
    helper_module.init_parsing_model = lambda *a, **k: parser
    try:
        helper_class = helper_module.FaceRestoreHelper
        face_helper = helper_class(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model="retinaface_resnet50", save_ext="png", use_parse=True,
            device=device, model_rootpath=REPO_ROOT / "modules" / "deps" / "facelib",
        )
    finally:
        helper_module.init_detection_model = original_init_det
        helper_module.init_parsing_model = original_init_parse

    keep_net = make_synthetic_keep(device)

    class Pack:
        pass

    pack = Pack()
    pack.keep_net = keep_net
    pack.face_helper = face_helper
    pack.bg_upscale_model = None  # the Load node default (nodes/keep_model_loader.py:92)
    pack.face_upscale_model = None
    pack.device = torch.device(device)
    pack.model_type_str = "KEEP-synthetic"
    processor = KEEPFaceProcessor(pack)
    return pack, processor


def make_input_frames(num_frames: int, device: str, landmarks, seed: int = 17):
    """Synthetic IMAGE batch (N, H, W, 3) float32 [0, 1] with a face-ish blob."""
    import torch

    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 0.95, size=(FRAME_H, FRAME_W, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    face_mask = ((xx - FRAME_W * 0.5) ** 2 + ((yy - FRAME_H * 0.46) * 1.25) ** 2) < (150.0**2)
    base[face_mask] *= 0.7

    frames = np.repeat(base[None, ...], num_frames, axis=0)
    frames += rng.normal(0, 0.01, size=frames.shape).astype(np.float32)
    frames = np.clip(frames, 0.0, 1.0)
    tensor = torch.from_numpy(frames).to(device)
    return tensor, rng


# ---------------------------------------------------------------------------
# The stage clone of process_image_sequence (base 2e8d9b9) - timing only
# ---------------------------------------------------------------------------


@torch.no_grad()
def stage_clone(processor, image_sequence, num_frames, landmarks, timer, max_clip_length=KEEP_CHUNK):
    import torch

    import src.keep_processor as processor_mod

    face_helper = processor.face_helper
    device = processor.device

    timer.measure("s1_convert_in", lambda: [
        processor_mod.comfy_image_to_cv2(image_sequence[i].unsqueeze(0))
        for i in range(num_frames)
    ])
    input_frames_bgr = [
        processor_mod.comfy_image_to_cv2(image_sequence[i].unsqueeze(0))
        for i in range(num_frames)
    ]

    def detect():
        raw = []
        for i in range(num_frames):
            face_helper.clean_all()
            face_helper.read_image(input_frames_bgr[i])
            face_helper.get_face_landmarks_5(
                only_center_face=True, resize=640, eye_dist_threshold=5, only_keep_largest=True)
            raw.append(face_helper.all_landmarks_5[0].reshape((10,)))
        return np.array(raw)

    raw_landmarks = timer.measure("s2_detect", detect)
    landmarks_seq = timer.measure(
        "s2b_smooth",
        lambda: processor_mod._gaussian_smooth(raw_landmarks, sigma=5.0).reshape(num_frames, 5, 2),
    )

    def crop():
        tensors = []
        for i in range(num_frames):
            face_helper.clean_all()
            face_helper.read_image(input_frames_bgr[i])
            face_helper.all_landmarks_5 = [landmarks_seq[i]]
            face_helper.align_warp_face()
            face_cv2 = face_helper.cropped_faces[0]
            face_t = processor_mod.img2tensor(face_cv2 / 255.0, bgr2rgb=True, float32=True)
            face_t.sub_(0.5).div_(0.5)
            tensors.append(face_t)
        return torch.stack(tensors, dim=0).unsqueeze(0).to(device)

    batched_cropped = timer.measure("s3_crop_align", crop)

    def keep():
        chunks = []
        for start_idx in range(0, num_frames, max_clip_length):
            end_idx = min(start_idx + max_clip_length, num_frames)
            clip = batched_cropped[:, start_idx:end_idx, ...]
            if clip.shape[1] == 1:
                clip = torch.cat([clip, clip], dim=1)
                out = processor.keep_net(clip, need_upscale=False)
                chunks.append(out[:, 0:1, ...])
            else:
                chunks.append(processor.keep_net(clip, need_upscale=False))
            # P4.6 - the base releases the cached blocks at the chunk boundary
            # (``keep_processor.py:336`` of the base).  The clone must keep
            # it: without the release the stage would be ~1 s faster than
            # the real base pipeline and the control would not be a control.
            processor_mod._soft_empty_cache()
        return torch.cat(chunks, dim=1).squeeze(0)

    restored = timer.measure("s4_keep_infer", keep)

    def tensor2img_loop():
        faces = [
            processor_mod.tensor2img(t, rgb2bgr=True, min_max=(-1, 1)) for t in restored
        ]
        # the base releases the cache after the conversion too
        processor_mod._soft_empty_cache()
        return faces

    timer.measure("s5_tensor2img", tensor2img_loop)
    restored_faces = [
        processor_mod.tensor2img(t, rgb2bgr=True, min_max=(-1, 1)) for t in restored
    ]

    def paste():
        out = []
        face_helper.upscale_factor = 1.0
        for i in range(num_frames):
            frame = input_frames_bgr[i]
            face_helper.clean_all()
            face_helper.read_image(frame)
            face_helper.add_restored_face(restored_faces[i].astype("uint8"))
            face_helper.all_landmarks_5 = [landmarks_seq[i]]
            face_helper.align_warp_face()
            face_helper.get_inverse_affine(None)
            out.append(face_helper.paste_faces_to_input_image(
                upsample_img=frame, draw_box=False, face_upsampler=None))
        return out

    output_frames = timer.measure("s6_paste", paste)

    timer.measure("s7_convert_out", lambda: torch.cat(
        [processor_mod.cv2_to_comfy_image(frame) for frame in output_frames], dim=0))
    return input_frames_bgr, landmarks_seq, restored_faces


def keep_stage_bench(processor, input_frames_bgr, affines, stack, num_frames, device, timer):
    """Why does the KEEP stage cost more in the shipped pipeline?

    The KEEP inference is **not redesignable**: the chunk loop, the net and
    the cache release are the same in the base and in the shipped code (the
    chunk contract is a quality contract).  Two things differ between the
    two measurements of the stage:

    * the **input** - the base clone warps the frames with the vendored
      ``cv2.warpAffine``, the shipped pipeline warps them with the batched
      ``grid_sample``; the two crops differ by at most 1/255,
    * the **granularity of the timer** - the clone measures the whole
      stage with one synchronization, the shipped pipeline wraps the net
      call itself (one synchronization per chunk).

    The four numbers below separate the two effects: the crop (the input)
    and the granularity (the measurement) are varied against each other.
    """
    import cv2
    import torch

    import src.keep_processor as processor_mod
    from src.keep_gpu_ops import crop_faces_batch
    from src.keep_image_utils import bgr_u8_batch_to_normalized

    # the crop of the base clone: the vendored cv2 warp + the img2tensor tail
    cv2_faces = []
    for i, affine in enumerate(affines):
        warped = cv2.warpAffine(
            input_frames_bgr[i], affine, (512, 512),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
        tensor = processor_mod.img2tensor(warped / 255.0, bgr2rgb=True, float32=True)
        cv2_faces.append(tensor.sub_(0.5).div_(0.5))
    cv2_crop = torch.stack(cv2_faces, dim=0).unsqueeze(0).to(device)

    # the crop of the shipped pipeline: the batched grid_sample warp
    batched_crop = bgr_u8_batch_to_normalized(
        crop_faces_batch(stack, list(affines))).unsqueeze(0)

    def time_chunks(crop, sync_per_chunk: bool) -> float:
        timer.sync()
        started = time.perf_counter()
        for start_idx in range(0, num_frames, KEEP_CHUNK):
            end_idx = min(start_idx + KEEP_CHUNK, num_frames)
            processor.keep_net(crop[:, start_idx:end_idx, ...], need_upscale=False)
            if sync_per_chunk:
                timer.sync()
        timer.sync()
        return round(time.perf_counter() - started, 3)

    # the warmup of both crops (the kernels are warm for every measurement).
    # The clip needs **two** frames: the temporal attention of the arch (the
    # gmflow transformer) cannot run on a single frame - the same reason the
    # processor duplicates a one-frame chunk.
    for crop in (cv2_crop, batched_crop):
        processor.keep_net(crop[:, :2, ...], need_upscale=False)
    timer.sync()

    measurements = {
        "cv2_crop_stage_sync_s": time_chunks(cv2_crop, False),
        "cv2_crop_chunk_sync_s": time_chunks(cv2_crop, True),
        "batched_crop_stage_sync_s": time_chunks(batched_crop, False),
        "batched_crop_chunk_sync_s": time_chunks(batched_crop, True),
    }
    difference = (cv2_crop - batched_crop).abs().max().item()
    measurements["crop_max_difference"] = round(difference, 6)
    measurements["crop_difference_in_255"] = round(difference * 127.5, 3)
    return measurements


# ---------------------------------------------------------------------------
# The shipped pipeline: the same stages, timed by wrapping its ops
# ---------------------------------------------------------------------------


class _StageProbe:
    """Times the stages of the **shipped** pipeline by wrapping its ops.

    The wrappers are the same kind of measurement as the stage timer of the
    base clone: the device is synchronized before and after every wrapped
    call, so the stage boundaries are the same in both tables.  The cost of
    the extra synchronizations is negligible (a few hundred calls against a
    run of tens of seconds) and the residual of the run (the parts that are
    not a wrapped op - the logging, the progress reports, the bookkeeping)
    is reported as ``unattributed_s``.

    Nothing in ``src/`` is touched: the probe only rebinds the attributes of
    the imported modules and of the stand-in pack for the duration of the
    measurement and restores them afterwards.
    """

    def __init__(self, timer: Timer, processor, processor_mod):
        self.timer = timer
        self.processor = processor
        self.processor_mod = processor_mod
        self._saved: list[tuple[object, str, object, bool]] = []
        #: The ops of the shipped tree that are **not** in the namespace of
        #: their owner - reported in the JSON, never silently ignored.
        self.missing: list[str] = []
        #: How many cv2 crop warps the discriminator caught - the proof that
        #: the crop stage is really timed (one call per croppable frame).
        self.cv2_crop_calls = 0
        #: Is the crop timed by the batched crop helper (the pre-D2 shape)?
        self._batched_crop_wrapped = False

    def _wrap(self, owner, attribute: str, stage: str) -> bool:
        """Wrap one op of the shipped pipeline with the stage timer.

        A **missing** attribute is not an error: the shipped tree may have
        replaced the op (the D2 decision of the design moved the crop back
        to a per-frame ``cv2.warpAffine``, so ``crop_faces_batch`` is not in
        the namespace of the processor any more).  The op is then timed by
        another wrapper (the cv2 crop warp, see :meth:`_wrap_cv2_crop`) and
        the missing attribute is reported in the JSON.
        """
        original = getattr(owner, attribute, None)
        if original is None:
            owner_name = getattr(owner, "__name__", type(owner).__name__)
            self.missing.append(f"{owner_name}.{attribute}")
            return False
        timer = self.timer

        def wrapper(*args, **kwargs):
            return timer.measure(stage, lambda: original(*args, **kwargs))

        self._saved.append(
            (owner, attribute, original, attribute in getattr(owner, "__dict__", {}))
        )
        setattr(owner, attribute, wrapper)
        return True

    def _wrap_cv2_crop(self, stage: str) -> None:
        """Times the **cv2 crop warp** of the shipped pipeline.

        The D2 decision (design section 14.3) moved the crop back to a
        per-frame ``cv2.warpAffine`` (``src/keep_processor.py``) - bit-exact
        against the base, at the cost of the batched ``grid_sample``.  The
        **discriminator** is the gray ``BORDER_CONSTANT`` value of the
        vendored crop (``(135, 133, 132)``,
        ``face_restoration_helper.py:326-328``): the warps of the vendored
        paste use the default black border and stay out of the crop stage.

        The wrapper is a no-op when the batched crop helper is in the
        namespace (the D2 decision reverted): the crop is then timed by the
        helper's own wrapper and nothing is counted twice.
        """
        if self._batched_crop_wrapped:
            return
        import cv2

        original = cv2.warpAffine
        timer = self.timer
        probe = self

        def wrapper(*args, **kwargs):
            if _is_crop_border(kwargs.get("borderValue")):
                probe.cv2_crop_calls += 1
                return timer.measure(stage, lambda: original(*args, **kwargs))
            return original(*args, **kwargs)

        self._saved.append(
            (cv2, "warpAffine", original, "warpAffine" in getattr(cv2, "__dict__", {}))
        )
        cv2.warpAffine = wrapper

    def __enter__(self) -> "_StageProbe":
        mod = self.processor_mod
        helper = self.processor.face_helper
        # s1 - the batched convert-in (one transfer for the whole batch)
        self._wrap(mod, "comfy_batch_to_bgr_u8", "s1_convert_in")
        # s2 - the vendored detection (the read and the detector call; the
        # ``clean_all`` of the loop is not wrapped - it only clears lists)
        self._wrap(helper, "read_image", "s2_detect")
        self._wrap(helper, "get_face_landmarks_5", "s2_detect")
        # s2b - the landmark smoothing (torch, already batched)
        self._wrap(mod, "_gaussian_smooth", "s2b_smooth")
        # s3 - the affine capture and the batched crop
        self._wrap(mod, "estimate_affines", "s3_crop_align")
        self._batched_crop_wrapped = self._wrap(mod, "crop_faces_batch", "s3_crop_align")
        # The crop itself: the per-frame ``cv2.warpAffine`` of the D2
        # decision (the gray border tells it apart from the warps of the
        # vendored paste).  A no-op when the batched crop is in use.
        self._wrap_cv2_crop("s3_crop_align")
        self._wrap(mod, "bgr_u8_batch_to_normalized", "s3_crop_align")
        # s4 - the KEEP inference (the chunk loop of the quality contract)
        self._wrap(self.processor, "keep_net", "s4_keep_infer")
        # s5 - the batched tensor2img
        self._wrap(mod, "tensor2img_batch", "s5_tensor2img")
        # s6 - the batched paste (and the vendored fallback of the frames
        # the batched paste does not take)
        self._wrap(mod, "is_gray_batch", "s6_paste")
        self._wrap(self.processor, "_batched_paste", "s6_paste")
        self._wrap(helper, "paste_faces_to_input_image", "s6_paste")
        # s7 - the batched convert-out
        self._wrap(mod, "bgr_u8_batch_to_comfy", "s7_convert_out")
        return self

    def __exit__(self, *exc_info) -> bool:
        for owner, attribute, original, was_instance_attribute in reversed(self._saved):
            if was_instance_attribute:
                setattr(owner, attribute, original)
            else:
                try:
                    delattr(owner, attribute)
                except AttributeError:  # pragma: no cover - defensive
                    setattr(owner, attribute, original)
        return False


class _PasteProbe:
    """The paste sub-stages of the shipped pipeline (wall + process CPU).

    The probe rebinds the attributes of the modules that **call** the
    paste ops (``src.keep_processor``, ``src.keep_gpu_paste``) and of
    ``cv2`` for the duration of the measurement, and restores them
    afterwards - the same pattern as :class:`_StageProbe`.  No production
    code is edited for the instrumentation.

    The sub-stage names are the ones of the architect's plan (doc section
    2): ``p1_face_warp``, ``p2_mask_warp``, ``p3_blend``, ``p6_bg_resize``,
    ``p7_is_gray``, ``p8_parse_forward`` and ``p8b_parse_post``.
    ``p4_bg_upload`` and ``p5_result_download`` are the **copies** - they
    are measured by the transfer accounting pass (:class:`_TransferCounter`),
    not by a function call.

    The wrap follows the *caller's* namespace: ``src.keep_processor``
    imported ``is_gray_batch`` and ``paste_faces_batch`` by name, so
    rebinding them in ``src.keep_gpu_paste`` would not change what the
    processor calls.
    """

    def __init__(self, timer: Timer, processor, processor_mod):
        self.timer = timer
        self.processor = processor
        self.processor_mod = processor_mod
        self._saved: list[tuple[object, str, object, bool]] = []
        #: the ops that are not in the namespace of their owner
        self.missing: list[str] = []
        #: the call counter per sub-stage
        self.counts: dict[str, int] = {}
        #: the arguments of the last batched paste (the accounting pass)
        self.captured: dict = {}

    def _wrap(self, owner, attribute: str, stage: str, capture: dict | None = None) -> bool:
        """Wrap one paste op with the sub-stage timer (and count the calls)."""
        original = getattr(owner, attribute, None)
        if original is None:
            owner_name = getattr(owner, "__name__", type(owner).__name__)
            self.missing.append(f"{owner_name}.{attribute}")
            return False
        timer = self.timer
        probe = self

        def wrapper(*args, **kwargs):
            probe.counts[stage] = probe.counts.get(stage, 0) + 1
            if capture is not None:
                capture.update(kwargs)
            return timer.measure(stage, lambda: original(*args, **kwargs))

        self._saved.append(
            (owner, attribute, original, attribute in getattr(owner, "__dict__", {}))
        )
        setattr(owner, attribute, wrapper)
        return True

    def _wrap_cv2(self) -> None:
        """Wrap the two cv2 ops of the paste stage.

        * ``cv2.warpAffine`` - the **face warp** of the paste.  The crop
          warp of the shipped pipeline is told apart by the gray border of
          the vendored crop (``(135, 133, 132)``), exactly like
          :meth:`_StageProbe._wrap_cv2_crop`; the paste warps use the
          default black border.
        * ``cv2.resize`` with ``INTER_LANCZOS4`` - the conditional
          background resize of the paste (the only LANCZOS4 resize of the
          sequence path).
        """
        import cv2

        original_warp = cv2.warpAffine
        original_resize = cv2.resize
        timer = self.timer
        probe = self

        def warp_wrapper(*args, **kwargs):
            if _is_crop_border(kwargs.get("borderValue")):
                return original_warp(*args, **kwargs)
            probe.counts["p1_face_warp"] = probe.counts.get("p1_face_warp", 0) + 1
            return timer.measure("p1_face_warp", lambda: original_warp(*args, **kwargs))

        def resize_wrapper(*args, **kwargs):
            interpolation = kwargs.get("interpolation")
            if interpolation is None and len(args) > 2:
                interpolation = args[2]
            if interpolation != cv2.INTER_LANCZOS4:
                return original_resize(*args, **kwargs)
            probe.counts["p6_bg_resize"] = probe.counts.get("p6_bg_resize", 0) + 1
            return timer.measure("p6_bg_resize", lambda: original_resize(*args, **kwargs))

        self._saved.append(
            (cv2, "warpAffine", original_warp, "warpAffine" in getattr(cv2, "__dict__", {}))
        )
        self._saved.append(
            (cv2, "resize", original_resize, "resize" in getattr(cv2, "__dict__", {}))
        )
        cv2.warpAffine = warp_wrapper
        cv2.resize = resize_wrapper

    def __enter__(self) -> "_PasteProbe":
        import src.keep_gpu_paste as paste_mod

        mod = self.processor_mod
        # the gray flags (the processor calls its own imported name)
        self._wrap(mod, "is_gray_batch", "p7_is_gray")
        # the batched paste (the whole path) - its arguments feed the
        # transfer accounting pass
        self._wrap(mod, "paste_faces_batch", "p0_paste_total", capture=self.captured)
        # the ops the paste module calls by name (its own namespace)
        self._wrap(paste_mod, "warp_affine_batch", "p2_mask_warp")
        self._wrap(paste_mod, "parse_soft_masks", "p8b_parse_post")
        # the blend: the blend arithmetic is queued before this call, so the
        # leading device sync of the timer flushes it - the measured wall
        # time is the blend's device work + the quantising tail
        self._wrap(paste_mod, "_quantize_and_clip", "p3_blend")
        self._wrap_cv2()
        # the ParseNet forward (the floor of the paste stage)
        parse_net = getattr(self.processor.face_helper, "face_parse", None)
        if parse_net is not None:
            self._wrap(parse_net, "forward", "p8_parse_forward")
        return self

    def __exit__(self, *exc_info) -> bool:
        for owner, attribute, original, was_instance_attribute in reversed(self._saved):
            if was_instance_attribute:
                setattr(owner, attribute, original)
            else:
                try:
                    delattr(owner, attribute)
                except AttributeError:  # pragma: no cover - defensive
                    setattr(owner, attribute, original)
        return False

    def table(self, num_frames: int) -> list[dict]:
        """The paste sub-stage table (wall + process CPU + the call counts)."""
        rows = []
        for name, seconds in self.timer.totals.items():
            cpu_seconds = self.timer.cpu_totals.get(name, 0.0)
            rows.append(
                {
                    "substage": name,
                    "wall_s": round(seconds, 3),
                    "wall_per_frame_ms": round(seconds * 1000.0 / num_frames, 2),
                    "cpu_s": round(cpu_seconds, 3),
                    "cpu_per_frame_ms": round(cpu_seconds * 1000.0 / num_frames, 2),
                    "cpu_cores": round(cpu_seconds / seconds, 2) if seconds > 0 else None,
                    "calls": self.counts.get(name, 0),
                }
            )
        return rows


class _TransferCounter(TorchFunctionMode):
    """The host<->device byte counters of the paste path.

    ``TorchFunctionMode`` intercepts the torch API calls, so the copies
    that **change the device** (``Tensor.to(device)``, ``Tensor.cpu()``)
    are seen with their tensor: their bytes are the tensor's size.  The
    pageable copies are blocking, so their wall time is the real transfer
    time and their process CPU time is the memcpy's CPU time.

    The counters are the evidence for "the transfers are canvas-sized"
    (the before-profile of the ROI round); after the ROI change the
    production counters (``PASTE_TRANSFER_STATS``) are the authority and
    the probe reads them as well.
    """

    def __init__(self):
        super().__init__()
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.h2d_calls = 0
        self.d2h_calls = 0
        self.h2d_wall_s = 0.0
        self.d2h_wall_s = 0.0
        self.h2d_cpu_s = 0.0
        self.d2h_cpu_s = 0.0
        self.h2d_sizes: dict[str, int] = {}
        self.d2h_sizes: dict[str, int] = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        started = time.perf_counter()
        cpu_started = time.process_time()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - started
        cpu_elapsed = time.process_time() - cpu_started
        self._account(func, args, kwargs, elapsed, cpu_elapsed)
        return result

    def _account(self, func, args, kwargs, elapsed: float, cpu_elapsed: float) -> None:
        if func not in (torch.Tensor.to, torch.Tensor.cpu, torch.Tensor.cuda):
            return
        tensor = args[0] if args else None
        if not isinstance(tensor, torch.Tensor):
            return
        source = tensor.device
        if func is torch.Tensor.cpu:
            target = torch.device("cpu")
        elif func is torch.Tensor.cuda:
            target = torch.device("cuda")
        else:
            device = kwargs.get("device", args[1] if len(args) > 1 else None)
            if not isinstance(device, (str, torch.device)):
                return  # a dtype conversion: no transfer
            target = torch.device(device)
        if target.type == source.type:
            return  # no device change
        nbytes = tensor.numel() * tensor.element_size()
        key = f"{tuple(tensor.shape)}:{tensor.dtype}".replace(" ", "")
        if target.type == "cuda":
            self.h2d_bytes += nbytes
            self.h2d_calls += 1
            self.h2d_wall_s += elapsed
            self.h2d_cpu_s += cpu_elapsed
            self.h2d_sizes[key] = self.h2d_sizes.get(key, 0) + nbytes
        else:
            self.d2h_bytes += nbytes
            self.d2h_calls += 1
            self.d2h_wall_s += elapsed
            self.d2h_cpu_s += cpu_elapsed
            self.d2h_sizes[key] = self.d2h_sizes.get(key, 0) + nbytes


def stage_rows(timer: Timer, num_frames: int) -> tuple[list[dict], float]:
    """The stage table of a timer (with the share of the known stages)."""
    rows = timer.table(num_frames)
    total = sum(
        row["total_s"]
        for row in rows
        if row["stage"].startswith(("s1", "s2", "s3", "s4", "s5", "s6", "s7"))
    )
    for row in rows:
        if row["stage"].startswith(("s1", "s2", "s3", "s4", "s5", "s6", "s7")):
            row["share_pct"] = round(100.0 * row["total_s"] / max(total, 1e-9), 1)
    return rows, round(total, 3)


# ---------------------------------------------------------------------------
# Micro benchmarks: the current per-frame op loop vs the batched candidate
# ---------------------------------------------------------------------------


@torch.no_grad()
def micro_bench(processor, input_frames_bgr, landmarks_seq, restored_faces, device, num_frames, timer):
    import torch

    import torch.nn.functional as F

    import cv2

    face_helper = processor.face_helper
    results: dict[str, dict] = {}

    # --- affine matrices once per frame (numpy, the SAME vendored call) ----
    face_template = face_helper.face_template
    timer.measure("m0_affine_numpy", lambda: [
        cv2.estimateAffinePartial2D(
            landmarks_seq[i], face_template, method=cv2.LMEDS)[0]
        for i in range(num_frames)
    ])
    affines = [
        cv2.estimateAffinePartial2D(
            landmarks_seq[i], face_template, method=cv2.LMEDS)[0]
        for i in range(num_frames)
    ]
    inverse_affines = [cv2.invertAffineTransform(a) * 1.0 for a in affines]

    stack = torch.stack([torch.from_numpy(np.ascontiguousarray(f)) for f in input_frames_bgr])
    stack = stack.permute(0, 3, 1, 2).float().to(device)  # (N, 3, H, W)

    # the theta tensors are precomputed OUTSIDE the timers on purpose: the
    # design computes them once in numpy (tiny) - only the GPU op is timed
    def thetas_from(mats):
        thetas = np.zeros((len(mats), 2, 3), dtype=np.float32)
        for i, a in enumerate(mats):
            # cv2.warpAffine inverts M internally: the sampling map is inv(M)
            thetas[i] = np.linalg.inv(np.vstack([a, [0.0, 0.0, 1.0]]))[:2]
        return torch.from_numpy(thetas).to(device)

    theta_crop = thetas_from(affines)
    theta_paste = thetas_from(inverse_affines)

    # --- m1 crop warp: cv2 loop vs batched affine_grid + grid_sample ------
    def cv2_crop():
        out = []
        for a in affines:
            out.append(cv2.warpAffine(
                input_frames_bgr[0], a, (512, 512),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132)))
        return out

    timer.measure("m1a_crop_cv2_loop", cv2_crop)

    def batched_crop():
        grid = F.affine_grid(theta_crop, (num_frames, 3, 512, 512), align_corners=False)
        return F.grid_sample(stack, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    timer.measure("m1b_crop_grid_sample", batched_crop)

    # --- m2 parse forward: per frame vs batch sizes (the VRAM sweep) ------
    parse = processor.face_helper.face_parse
    face_tensors = torch.randn(num_frames, 3, 512, 512, device=device)
    sweep: dict[str, float] = {}

    def parse_loop():
        for i in range(num_frames):
            parse(face_tensors[i : i + 1])[0]

    timer.measure("m2a_parse_per_frame", parse_loop)
    sweep["1"] = round(timer.totals["m2a_parse_per_frame"], 3)

    for batch in (4, 8, 16, num_frames):
        key = f"m2_bs{batch}"
        if batch > num_frames:
            continue
        try:
            timer.measure(key, lambda b=batch: [
                parse(face_tensors[j : j + b])[0] for j in range(0, num_frames, b)
            ])
            sweep[str(batch)] = round(timer.totals[key], 3)
        except torch.OutOfMemoryError:
            sweep[str(batch)] = "OOM"
            if device == "cuda":
                torch.cuda.empty_cache()

    # --- m3 parse post blur: cv2 loop vs batched torch conv ---------------
    rng = np.random.default_rng(3)
    raw_masks = [rng.uniform(0, 255, size=(512, 512)).astype(np.float32) for _ in range(num_frames)]

    def blur_cv2():
        for m in raw_masks:
            b = cv2.GaussianBlur(m, (101, 101), 11)
            cv2.GaussianBlur(b, (101, 101), 11)

    timer.measure("m3a_blur_cv2_loop", blur_cv2)

    def blur_batched():
        kernel = cv2.getGaussianKernel(101, 11).astype(np.float32)
        weight = torch.from_numpy(np.outer(kernel, kernel)).unsqueeze(0).unsqueeze(0).to(device)
        batch = torch.from_numpy(np.stack(raw_masks)).unsqueeze(1).to(device)
        for _ in range(2):
            batch = F.conv2d(batch, weight, padding=(50, 50))
        return batch

    timer.measure("m3b_blur_torch_batch", blur_batched)

    # --- m4 paste warp: per-frame inverse warpAffine vs batched grid_sample
    faces_u8 = [np.ascontiguousarray(f) for f in restored_faces]

    def paste_cv2():
        out = []
        for inv in inverse_affines:
            out.append(cv2.warpAffine(faces_u8[0], inv, (FRAME_W, FRAME_H)))
        return out

    timer.measure("m4a_paste_cv2_loop", paste_cv2)

    def paste_batched():
        src = torch.stack([torch.from_numpy(f) for f in faces_u8]).permute(0, 3, 1, 2).float().to(device)
        grid = F.affine_grid(theta_paste, (num_frames, 3, FRAME_H, FRAME_W), align_corners=False)
        return F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    timer.measure("m4b_paste_grid_sample", paste_batched)

    # --- m5 conversions: faithful per-frame (transfer included) vs batched -
    comfy_batch = torch.rand(num_frames, FRAME_H, FRAME_W, 3, device=device)

    def convert_in_loop():
        out = []
        for i in range(num_frames):
            frame = comfy_batch[i].unsqueeze(0)
            out.append((frame * 255).to(torch.uint8)[..., [2, 1, 0]].cpu().numpy())
        return out

    timer.measure("m5a_convert_in_per_frame", convert_in_loop)

    def convert_in_batched():
        return (comfy_batch * 255).to(torch.uint8)[..., [2, 1, 0]].cpu().numpy()

    timer.measure("m5b_convert_in_batched", convert_in_batched)

    out_u8 = torch.randint(0, 256, (num_frames, FRAME_H, FRAME_W, 3), dtype=torch.uint8, device=device)

    def convert_out_loop():
        out = []
        for i in range(num_frames):
            frame = out_u8[i].cpu().numpy()
            out.append(torch.from_numpy(np.ascontiguousarray(frame[..., [2, 1, 0]])).to(torch.float32) / 255.0)
        return out

    timer.measure("m5c_convert_out_per_frame", convert_out_loop)

    def convert_out_batched():
        return out_u8[..., [2, 1, 0]].to(torch.float32).div(255.0).cpu()

    timer.measure("m5d_convert_out_batched", convert_out_batched)

    # --- m6 detection split: read_image vs the detector call --------------
    def read_image_loop():
        for frame in input_frames_bgr:
            face_helper.clean_all()
            face_helper.read_image(frame)

    timer.measure("m6a_read_image_loop", read_image_loop)

    detector = face_helper.face_detector

    def detect_faces_loop():
        for frame in input_frames_bgr:
            try:
                detector.detect_faces(frame, conf_threshold=0.97)
            except Exception:
                pass

    timer.measure("m6b_detect_faces_loop", detect_faces_loop)

    # --- m7 the KEEP stage: the input and the granularity of the timer ---
    results["keep_stage"] = keep_stage_bench(
        processor, input_frames_bgr, affines, stack, num_frames, device, timer)

    results["notes"] = (
        "theta/affine tensors are precomputed outside the timers (the design "
        "computes them once, tiny); m2 sweeps the ParseNet batch size for the "
        "VRAM cliff; m6 splits the detection stage into read_image vs the "
        "detector call (random-weight NMS caveat: the postprocess cost is not "
        "representative of trained weights); m7 separates the two effects "
        "that make the KEEP stage of the shipped pipeline look different "
        "from the clone (the input crop and the granularity of the timer)."
    )
    results["parse_batch_sweep_s"] = sweep
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", dest="comfyui_root", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--skip-e2e", action="store_true",
                        help="skip the runs of the shipped pipeline (the control "
                             "of the base clone and the micro benchmarks stay)")
    parser.add_argument("--skip-micro", action="store_true",
                        help="skip the micro benchmarks of the batched candidates")
    args = parser.parse_args(argv)

    if args.comfyui_root:
        os.environ["COMFYUI_ROOT"] = args.comfyui_root
    comfyui_root = resolve_comfyui_root(args.comfyui_root)
    if comfyui_root is None:
        print("No ComfyUI tree found: set COMFYUI_ROOT or pass --comfyui-root")
        return 2
    sys.path.insert(0, comfyui_root)
    sys.path.insert(0, str(REPO_ROOT))

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("cuda requested but not available")
            return 2

    report = {
        "device": device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "frames": args.frames,
        "frame_size": [FRAME_H, FRAME_W],
        "keep_chunk": KEEP_CHUNK,
        "probe_revision": "t5-after-repair2",
    }

    rng = np.random.default_rng(args.seed)
    face_template = np.array(
        [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
         [201.26117, 371.41043], [313.08905, 371.15118]])
    landmarks = build_synthetic_landmarks(face_template, args.frames, rng)

    pack, processor = build_pack(device, landmarks)
    image_sequence, _ = make_input_frames(args.frames, "cpu", landmarks, seed=args.seed)
    image_sequence = image_sequence.to(device)

    # warmup (CUDA kernels, lazy inits) - the reported numbers are warm
    warm = image_sequence[: min(3, args.frames)].clone()
    processor.process_image_sequence(warm, 1.0, False, True, False, max_clip_length=KEEP_CHUNK)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- the control: the base implementation (the clone of 2e8d9b9) -----
    timer = Timer(device)
    stage_clone(processor, image_sequence, args.frames, landmarks, timer)
    rows, total_stage_s = stage_rows(timer, args.frames)
    report["baseline_clone"] = {
        "note": (
            "the base implementation (2e8d9b9) as a clone inside this probe - "
            "the control of this session"
        ),
        "stages": rows,
        "stages_total_s": total_stage_s,
    }
    if device == "cuda":
        report["baseline_clone"]["peak_vram_mb"] = round(
            torch.cuda.max_memory_allocated() / (2**20), 1)

    # --- the shipped pipeline --------------------------------------------
    if not args.skip_e2e:
        import src.keep_processor as processor_mod

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # the clean end to end run (no wrappers): the number that is
        # comparable with the e2e of the before run
        timer_e2e = Timer(device)
        timer_e2e.sync()
        start = time.perf_counter()
        processor.process_image_sequence(
            image_sequence, 1.0, False, True, False, max_clip_length=KEEP_CHUNK)
        timer_e2e.sync()
        e2e_s = round(time.perf_counter() - start, 3)

        # the instrumented run: the same call, timed stage by stage
        timer_new = Timer(device)
        paste_timer = Timer(device)
        with _StageProbe(timer_new, processor, processor_mod) as probe:
            with _PasteProbe(paste_timer, processor, processor_mod) as paste_probe:
                processor.process_image_sequence(
                    image_sequence, 1.0, False, True, False, max_clip_length=KEEP_CHUNK)
        rows_new, total_new = stage_rows(timer_new, args.frames)
        measured_new = sum(timer_new.totals.values())
        report["pipeline"] = {
            "note": (
                "the shipped pipeline (the batched path of src/keep_processor.py) - "
                "its stages are timed by wrapping its ops"
            ),
            "stages": rows_new,
            "stages_total_s": total_new,
            "e2e_s": e2e_s,
            "instrumented_total_s": round(measured_new, 3),
            "unattributed_s": round(e2e_s - measured_new, 3),
            # the ops of the shipped tree that are not in the namespace of
            # their owner (the D2 decision moved the crop back to cv2) - the
            # crop stage is then timed by the cv2 crop warp instead
            "unwrapped_ops": probe.missing,
            # the cv2 crop warps the discriminator caught (one per croppable
            # frame) - the proof that the crop stage is really timed
            "cv2_crop_calls": probe.cv2_crop_calls,
        }
        report["paste_substages"] = {
            "note": (
                "the paste stage sub-stages of the shipped pipeline - the wall "
                "time (device-synced) and the process CPU time (the CPU load) "
                "of every sub-stage; p4_bg_upload and p5_result_download are "
                "the copies and are measured by the accounting pass below"
            ),
            "substages": paste_probe.table(args.frames),
            "unwrapped_ops": paste_probe.missing,
            "wall_total_s": round(sum(paste_timer.totals.values()), 3),
            "cpu_total_s": round(sum(paste_timer.cpu_totals.values()), 3),
        }

        # --- the transfer accounting pass (the byte counters) ---------------
        # The batched paste is called again with the arguments the pipeline
        # used, under a mode that counts the host<->device copies.  The
        # pageable copies are blocking, so their wall time is the real
        # transfer time.
        captured = paste_probe.captured
        transfer: dict = {
            "note": (
                "the host<->device copies of the batched paste, counted by a "
                "TorchFunctionMode over a direct call of paste_faces_batch "
                "with the arguments of the instrumented run"
            ),
            "the_batched_paste_ran": bool(captured.get("faces_bgr") is not None),
        }
        if captured.get("faces_bgr") is not None:
            counter = _TransferCounter()
            with counter:
                processor_mod.paste_faces_batch(
                    faces_bgr=captured["faces_bgr"],
                    inverse_affines=captured["inverse_affines"],
                    background_bgr=captured["background_bgr"],
                    device=captured.get("device", processor.device),
                    parse_net=captured.get("parse_net"),
                )
            frames = int(np.asarray(captured["faces_bgr"]).shape[0])
            background = np.asarray(captured["background_bgr"])
            transfer.update(
                {
                    "frames": frames,
                    "h2d_bytes": counter.h2d_bytes,
                    "d2h_bytes": counter.d2h_bytes,
                    "h2d_calls": counter.h2d_calls,
                    "d2h_calls": counter.d2h_calls,
                    "h2d_wall_s": round(counter.h2d_wall_s, 3),
                    "d2h_wall_s": round(counter.d2h_wall_s, 3),
                    "h2d_cpu_s": round(counter.h2d_cpu_s, 3),
                    "d2h_cpu_s": round(counter.d2h_cpu_s, 3),
                    "h2d_bytes_per_frame": counter.h2d_bytes // max(frames, 1),
                    "d2h_bytes_per_frame": counter.d2h_bytes // max(frames, 1),
                    "canvas_bytes_per_frame": int(background[0].nbytes),
                    "h2d_sizes": counter.h2d_sizes,
                    "d2h_sizes": counter.d2h_sizes,
                }
            )
        import src.keep_gpu_paste as paste_mod

        production = getattr(paste_mod, "PASTE_TRANSFER_STATS", None)
        if production is not None:
            transfer["production_counters"] = (
                dict(production) if isinstance(production, dict)
                else {k: v for k, v in vars(production).items() if not k.startswith("_")}
            )
        report["paste_transfer_bytes"] = transfer
        if device == "cuda":
            report["pipeline"]["peak_vram_mb"] = round(
                torch.cuda.max_memory_allocated() / (2**20), 1)

    if not args.skip_micro and device == "cuda":
        import src.keep_processor as processor_mod

        input_frames_bgr = [
            processor_mod.comfy_image_to_cv2(image_sequence[i].unsqueeze(0))
            for i in range(args.frames)
        ]
        face_rng = np.random.default_rng(11)
        restored_faces = [
            face_rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
            for _ in range(args.frames)
        ]
        micro_timer = Timer(device)
        micro = micro_bench(
            processor, input_frames_bgr, landmarks, restored_faces, device, args.frames, micro_timer)
        micro["timings_s"] = {k: round(v, 3) for k, v in micro_timer.totals.items()}
        report["micro"] = micro

    text = json.dumps(report, indent=2)
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
