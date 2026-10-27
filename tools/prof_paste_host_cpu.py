"""The host-CPU attribution of the "Pasting faces and finalizing frames" step.

The instrument of task t1 of ``.dev/docs/2026-10-27-keep-paste-step-host-cpu-design.md``
(section 2.1).  It measures the **host** cost of the paste step of
``src/keep_processor.py::process_image_sequence`` - the step the owner reported
as the remaining CPU load after the model finished.

What it reuses: the ``Timer``, ``_StageProbe``, ``_PasteProbe`` and
``_TransferCounter`` of ``tools/prof_gpu_video_pipeline.py`` (imported by path,
never copied) so the numbers of the two tools are comparable.

What it adds:

1. **The stub-net pack** - a vendored ``FaceRestoreHelper`` whose detector is a
   synthetic detection row and whose parse net is a 1x1 convolution (the fixture
   of ``tests/test_gpu_paste_equivalence.py:93-107``), plus a pass-through KEEP
   net.  No weights are read and no architecture is constructed, on any device:
   the ``--device cuda`` path of the probe (the real RetinaFace / ParseNet /
   KEEP architectures) is **not** used here and stays gated on the owner's
   written go-ahead.
2. **The routing recorder** - the per-frame answer to the plan's first
   question: which frames take the batched device paste, which take the
   vendored per-frame fallback, and for which reason.  It captures the
   arguments of ``_batched_paste``, counts the ``add_restored_face`` calls of
   the paste loop and recomputes every predicate of
   ``src/keep_processor.py:171-189`` frame by frame.
3. **The extended sub-stage set** - the probe's paste sub-stages plus the ones
   the CPU attribution needs: the ROI face warp, the ROI box, the fixed-point
   terms, the background maximum, the full-canvas fallback, the ParseNet
   forward itself (wrapped through ``__call__`` - the stub has no ``forward``)
   and the progress reports.
4. **The repeat-to-budget micro benches** for the sub-tick items
   (``time.process_time`` has one system tick of granularity on Windows - a
   sub-2-tick op reports 0 CPU).  The host copies of the ROI patch, the
   download and the final ``np.stack`` are repeated until each accrues
   >= 200 ms of wall time and reported with their budget.

Run (embedded interpreter):

    COMFYUI_ROOT=<ComfyUI source tree> python tools/prof_paste_host_cpu.py \\
        --device cpu --frames 8 --frame-hw 360 640 \\
        --json .dev/prof-paste-host-cpu.json
    COMFYUI_ROOT=<ComfyUI source tree> python tools/prof_paste_host_cpu.py \\
        --device cpu --frames 4 --frame-hw 720 1280 \\
        --json .dev/prof-paste-host-cpu-720p.json

Exit code 0 when the attribution ran; the report content is the evidence.
This tool changes no production code.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = REPO_ROOT / "tools" / "prof_gpu_video_pipeline.py"

FACE_TEMPLATE = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]])

#: the repeat budget of the micro benches: an op is repeated until it accrues
#: this much wall time, so its process-CPU number covers several system ticks
#: (``time.process_time`` has ~15.6 ms of granularity on Windows)
MICRO_BUDGET_S = 0.2


def load_probe():
    """The profiling probe as a module - its measurement classes are reused."""
    spec = importlib.util.spec_from_file_location("keep_prof_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The stub pack (no weights, no architectures - t1 fixture of the design)
# ---------------------------------------------------------------------------


class _StubParseNet:
    """The fixture of ``tests/test_gpu_paste_equivalence.py:93-107``.

    A 1x1 convolution over the color channels: the class map is a function of
    the pixels alone, so the forward is batch invariant by construction.
    """

    def __init__(self, seed: int = 5) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(19, 3, 1, 1, generator=generator) * 0.35
        self.calls = 0

    def __call__(self, samples):
        self.calls += 1
        return (F.conv2d(samples, self.weight), None)


class _PassThroughKeep:
    """A KEEP net that returns its input: the paste stage never sees pixels
    of the keep net, only the copy traffic around them."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, frames, need_upscale=False):
        self.calls.append((tuple(frames.shape), need_upscale))
        return frames


class _SyntheticDetector:
    """A detector that returns the saved landmark row of the current frame.

    The vendored ``get_face_landmarks_5`` call path is the real one; only the
    detection forward is replaced (the probe does the same, with the real
    RetinaFace forward in front - this instrument does not need it).
    """

    def __init__(self, landmarks: np.ndarray) -> None:
        self.frame_idx = 0
        self.landmarks = landmarks

    def detect_faces(self, image, conf_threshold=0.8, **_kwargs):
        row = _SyntheticDetector._row(self.landmarks[self.frame_idx % len(self.landmarks)])
        self.frame_idx += 1
        return row

    @staticmethod
    def _row(landmarks_5x2: np.ndarray) -> np.ndarray:
        """The (1, 15) RetinaFace decode row: box + score + 5 landmarks."""
        x_min, y_min = landmarks_5x2.min(axis=0) - 40.0
        x_max, y_max = landmarks_5x2.max(axis=0) + 60.0
        row = [x_min, y_min, x_max, y_max, 0.99]
        for point in landmarks_5x2:
            row.extend([point[0], point[1]])
        return np.array([row], dtype=np.float32)


def build_stub_pack(device: str, landmarks: np.ndarray):
    """The owner's default configuration: parse net on, no upscale models."""
    import importlib

    import src.keep_processor as processor_mod  # noqa: F401 - the import path
    from src.keep_processor import KEEPFaceProcessor

    helper_module = importlib.import_module(
        "modules.deps.facelib.utils.face_restoration_helper")

    detector = _SyntheticDetector(landmarks)
    parser = _StubParseNet()
    keep_net = _PassThroughKeep()

    original_init_det = helper_module.init_detection_model
    original_init_parse = helper_module.init_parsing_model
    helper_module.init_detection_model = lambda *a, **k: detector
    helper_module.init_parsing_model = lambda *a, **k: parser
    try:
        face_helper = helper_module.FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model="retinaface_resnet50", save_ext="png", use_parse=True,
            device=device, model_rootpath=REPO_ROOT / "modules" / "deps" / "facelib")
    finally:
        helper_module.init_detection_model = original_init_det
        helper_module.init_parsing_model = original_init_parse

    pack = type(
        "Pack", (),
        {
            "keep_net": keep_net,
            "face_helper": face_helper,
            "bg_upscale_model": None,   # the Load node default
            "face_upscale_model": None,
            "device": torch.device(device),
            "model_type_str": "KEEP-stub",
        },
    )()
    processor = KEEPFaceProcessor(pack)
    return pack, processor, parser, keep_net


def synthetic_landmarks(num_frames: int, height: int, width: int, seed: int) -> np.ndarray:
    """One face per frame: the 512 template at a jittered in-frame position."""
    rng = np.random.default_rng(seed)
    base_scale = min(height, width) / 512.0 * 0.72
    landmarks = np.zeros((num_frames, 5, 2), dtype=np.float64)
    for i in range(num_frames):
        scale = base_scale + rng.uniform(-0.005, 0.005)
        center = np.array([width * 0.5 + rng.uniform(-8, 8), height * 0.46 + rng.uniform(-6, 6)])
        landmarks[i] = (FACE_TEMPLATE - 256.0) * scale + center
    return landmarks


def smooth_frames(num_frames: int, height: int, width: int, seed: int) -> np.ndarray:
    """The comfy IMAGE batch (N, H, W, 3) float32 [0, 1], smooth content.

    Low-frequency content is the realistic case for the paste; the frame size
    is the caller's, so the host-only items are priced at production size.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 0.95, size=(height, width, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    face_mask = ((xx - width * 0.5) ** 2 + ((yy - height * 0.46) * 1.25 ** 2) ** 2) < (150.0**2)
    base[face_mask] *= 0.7
    frames = np.repeat(base[None, ...], num_frames, axis=0)
    frames += rng.normal(0, 0.01, size=frames.shape).astype(np.float32)
    return np.clip(frames, 0.0, 1.0)


# ---------------------------------------------------------------------------
# The routing recorder (the plan's first question, answered per frame)
# ---------------------------------------------------------------------------


def frame_routes(frames_bgr, restored_faces, affine_matrices, landmarks_sequence,
                 gray_flags, draw_box, face_upscaler, parse_net, use_parse,
                 has_aligned, eligible) -> list[dict]:
    """The route and the reason of every frame - the predicates of
    ``src/keep_processor.py:171-189`` recomputed from the captured arguments."""
    routes = []
    for i in range(len(frames_bgr)):
        if parse_net is None or not use_parse:
            reason = "no parse net / use_parse off"
        elif draw_box or face_upscaler is not None:
            reason = "draw_box or face upsampler (whole batch)"
        elif landmarks_sequence is None:
            reason = "avg_landmarks is None (whole batch)"
        elif has_aligned:
            reason = "has_aligned_frames (the aligned branch)"
        elif i >= len(affine_matrices) or affine_matrices[i] is None:
            reason = "no affine matrix"
        elif i >= len(landmarks_sequence) or np.isnan(landmarks_sequence[i]).any():
            reason = "NaN landmark row"
        elif gray_flags[i]:
            reason = "gray frame"
        elif i >= len(restored_faces):
            reason = "no restored face (background only)"
        elif restored_faces[i].shape[:2] != (512, 512):
            reason = "restored face is not 512x512"
        else:
            reason = None
        routes.append({
            "frame": i,
            "route": "batched" if i in eligible else "vendored_fallback",
            "reason": reason,
        })
    return routes


class _RoutingRecorder:
    """Captures the eligible set, the ``_batched_paste`` arguments and the
    ``add_restored_face`` calls of the paste loop."""

    def __init__(self, processor, helper) -> None:
        self.processor = processor
        self.helper = helper
        self.eligible: list[int] = []
        self.args: dict = {}
        self.add_restored_calls = 0
        self._saved: list[tuple] = []

    def __enter__(self) -> "_RoutingRecorder":
        original = self.processor._batched_paste

        def wrapper(*args, **kwargs):
            bound = dict(zip(
                ("input_frames_bgr", "restored_faces", "affine_matrices",
                 "avg_landmarks_sequence", "final_upscale_factor", "draw_box",
                 "gray_flags", "canvas_hw"), args))
            bound.update(kwargs)
            self.args = bound
            result = original(*args, **kwargs)
            self.eligible = sorted(result)
            return result

        self._saved.append(("_batched_paste", original))
        self.processor._batched_paste = wrapper

        original_add = self.helper.add_restored_face

        def add_wrapper(face, **kwargs):
            self.add_restored_calls += 1
            return original_add(face, **kwargs)

        self._saved.append(("add_restored_face", original_add))
        self.helper.add_restored_face = add_wrapper
        return self

    def __exit__(self, *exc_info) -> bool:
        for name, original in self._saved:
            setattr(self.processor if name == "_batched_paste" else self.helper, name, original)
        return False

    def restore(self) -> None:
        self.__exit__()


# ---------------------------------------------------------------------------
# The repeat-to-budget micro benches (the sub-tick host items)
# ---------------------------------------------------------------------------


def _repeat(timer, name: str, fn, budget_s: float = MICRO_BUDGET_S) -> int:
    """Repeat ``fn`` until it accrues ``budget_s`` of wall time; the process
    CPU time of every repetition is accumulated under ``name``."""
    repetitions = 0
    timer.sync()
    cpu_start = time.process_time()
    start = time.perf_counter()
    while True:
        fn()
        repetitions += 1
        elapsed = time.perf_counter() - start
        if elapsed >= budget_s and repetitions >= 3:
            break
        if elapsed >= 20.0 * budget_s:  # a slow op: stop growing
            break
    cpu_elapsed = time.process_time() - cpu_start
    timer.totals[name] = timer.totals.get(name, 0.0) + elapsed
    timer.cpu_totals[name] = timer.cpu_totals.get(name, 0.0) + cpu_elapsed
    return repetitions


def host_copy_benches(timer, device: str, height: int, width: int) -> dict:
    """The host copies of the ROI paste, at the given frame size.

    The copies are inline in ``_paste_chunk_roi`` (``src/keep_gpu_paste.py``)
    - they cannot be wrapped by name, so they are priced here with the exact
    op sequence of that code path, repeated to the tick budget.  The ROI is
    the face footprint plus one pixel: a realistic patch of a 512 face at a
    0.72 scale of a 360x640 frame.
    """
    face_scale = min(height, width) / 512.0 * 0.72
    roi_h = roi_w = max(16, int(512 * face_scale) + 8)
    roi_h = min(roi_h, height)
    roi_w = min(roi_w, width)
    patch = np.random.default_rng(7).integers(
        0, 256, size=(roi_h, roi_w, 3), dtype=np.uint8)
    background = np.random.default_rng(8).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8)
    quantized = torch.randint(0, 256, (3, roi_h, roi_w), dtype=torch.uint8)

    def upload():
        raw = np.ascontiguousarray(patch)
        tensor = torch.from_numpy(raw).to(device)
        return tensor.permute(2, 0, 1).to(torch.float32)

    def download():
        rows = quantized.permute(1, 2, 0).contiguous().cpu().numpy()
        frame = background.copy()
        frame[8:8 + roi_h, 8:8 + roi_w] = rows
        return frame

    def whole_stack(num=8):
        frames = [np.random.default_rng(9 + i).integers(
            0, 256, size=(height, width, 3), dtype=np.uint8) for i in range(num)]
        return np.stack(frames)

    benches = {}
    benches["p4_bg_upload_reps"] = _repeat(timer, "p4_bg_upload", upload)
    benches["p5_result_download_reps"] = _repeat(timer, "p5_result_download", download)
    benches["p10_output_stack_reps"] = _repeat(
        timer, "p10_output_stack", lambda: whole_stack(8))
    benches["roi_patch_bytes"] = int(patch.nbytes)
    benches["canvas_bytes"] = int(background.nbytes)
    benches["frame_hw"] = [height, width]
    return benches


# ---------------------------------------------------------------------------
# The attribution run
# ---------------------------------------------------------------------------


def attribute(device: str, num_frames: int, height: int, width: int, warmup: int = 2) -> dict:
    probe = load_probe()
    landmarks = synthetic_landmarks(num_frames, height, width, seed=17)
    pack, processor, parser, keep_net = build_stub_pack(device, landmarks)
    frames = smooth_frames(num_frames, height, width, seed=17)
    image_sequence = torch.from_numpy(frames).to(device)

    if warmup:
        processor.process_image_sequence(
            image_sequence[:min(warmup, num_frames)].clone(), 1.0, False, True, False,
            max_clip_length=20)

    stage_timer = probe.Timer(device)
    paste_timer = probe.Timer(device)
    import src.keep_processor as processor_mod
    import src.keep_gpu_paste as paste_mod
    import src.keep_progress as progress_mod

    # the extended paste sub-stages: the per-frame ROI prep, the ParseNet
    # forward of the stub (through ``__call__``) and the progress reports
    extras: list[tuple[object, str, str]] = [
        (paste_mod, "warp_face_roi", "p1b_roi_face_warp"),
        (paste_mod, "face_roi_box", "p1c_roi_box"),
        (paste_mod, "warp_fixed_point_terms", "p1d_fixed_point_terms"),
        (paste_mod, "_background_max_outside", "p1e_bg_max_outside"),
        (paste_mod, "_paste_frame_full_canvas", "p1f_full_canvas"),
        (parser, "__call__", "p8a_parse_forward"),
        (progress_mod.ProgressReporter, "update", "p9_progress"),
    ]
    saved_extras: list[tuple[object, str, object]] = []
    for owner, name, stage in extras:
        original = getattr(owner, name, None)
        if original is None:
            continue

        def make_wrapper(original, stage):
            def wrapper(*args, **kwargs):
                return paste_timer.measure(stage, lambda: original(*args, **kwargs))
            return wrapper

        saved_extras.append((owner, name, original))
        setattr(owner, name, make_wrapper(original, stage))

    routing = _RoutingRecorder(processor, processor.face_helper)
    try:
        with probe._StageProbe(stage_timer, processor, processor_mod) as stage_probe:
            with probe._PasteProbe(paste_timer, processor, processor_mod) as paste_probe:
                with routing:
                    processor.process_image_sequence(
                        image_sequence, 1.0, False, True, False, max_clip_length=20)
    finally:
        for owner, name, original in saved_extras:
            setattr(owner, name, original)

    # the F1 before/after: the SAME paste call in the host-output and the
    # device-output mode, one session, one fixture.  On a CPU device the
    # arithmetic of the two modes is identical, so the delta is exactly the
    # removed host assembly (the upload, the download, the canvas copy and
    # the stack).  The CUDA run of the same comparison is the owner-gated
    # V2; this one is honest at the host-copy level on the CPU fixture.
    captured = getattr(paste_probe, "captured", {})
    comparison: dict = {
        "note": (
            "the same paste_faces_batch call in both output modes - the "
            "device-output delta is the removed host assembly (the CPU "
            "fixture prices it exactly; the CUDA run is the owner-gated V2)"
        ),
    }
    if captured.get("faces_bgr") is not None:
        call_kwargs = dict(
            faces_bgr=captured["faces_bgr"],
            inverse_affines=captured["inverse_affines"],
            background_bgr=captured["background_bgr"],
            device=captured.get("device", processor.device),
            parse_net=captured.get("parse_net"),
            frame_chunk=None,
        )
        mode_timer = probe.Timer(device)
        results = {}
        for label, extra in (("host_output", {}), ("device_output", {"device_output": True})):
            paste_mod.reset_paste_transfer_stats()
            results[label] = mode_timer.measure(
                label, lambda k=dict(call_kwargs, **extra): paste_mod.paste_faces_batch(**k))
            stats = dict(paste_mod.PASTE_TRANSFER_STATS)
            comparison[label] = {
                "per_frame_ms": round(mode_timer.totals[label] * 1000.0 / num_frames, 2),
                "cpu_per_frame_ms": round(mode_timer.cpu_totals[label] * 1000.0 / num_frames, 2),
                "counters": stats,
            }
        host_frames = results["host_output"]
        canvas = results["device_output"]
        if torch.is_tensor(canvas):
            comparison["bit_identical"] = bool(
                np.array_equal(np.stack(host_frames), canvas.cpu().numpy()))

    # the transfer accounting (the same pass the probe runs)
    transfer: dict = {"the_batched_paste_ran": bool(routing.args)}
    if routing.args:
        if captured.get("faces_bgr") is not None:
            counter = probe._TransferCounter()
            with counter:
                processor_mod.paste_faces_batch(
                    faces_bgr=captured["faces_bgr"],
                    inverse_affines=captured["inverse_affines"],
                    background_bgr=captured["background_bgr"],
                    device=captured.get("device", processor.device),
                    parse_net=captured.get("parse_net"))
            background = np.asarray(captured["background_bgr"])
            transfer.update({
                "frames": int(np.asarray(captured["faces_bgr"]).shape[0]),
                "h2d_bytes": counter.h2d_bytes,
                "d2h_bytes": counter.d2h_bytes,
                "h2d_bytes_per_frame": counter.h2d_bytes // max(int(np.asarray(captured["faces_bgr"]).shape[0]), 1),
                "d2h_bytes_per_frame": counter.d2h_bytes // max(int(np.asarray(captured["faces_bgr"]).shape[0]), 1),
                "canvas_bytes_per_frame": int(background[0].nbytes),
            })

    # the host-copy micro benches (the sub-tick items)
    copy_timer = probe.Timer(device)
    benches = host_copy_benches(copy_timer, device, height, width)

    stage_rows, stages_total_s = probe.stage_rows(stage_timer, num_frames)
    paste_rows = paste_timer.table(num_frames)
    copy_rows = copy_timer.table(num_frames)

    # the routes: the eligible set of the run, the reasons recomputed
    args = routing.args
    routes = []
    if args:
        routes = frame_routes(
            args["input_frames_bgr"], args["restored_faces"], args["affine_matrices"],
            args["avg_landmarks_sequence"], args["gray_flags"],
            args["draw_box"], None, getattr(processor.face_helper, "face_parse", None),
            bool(getattr(processor.face_helper, "use_parse", False)),
            False, set(routing.eligible))

    return {
        "device": device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "frames": num_frames,
        "frame_hw": [height, width],
        "fixture": "stub nets - no weights, no architectures",
        "stages": {"rows": stage_rows, "total_s": stages_total_s,
                   "unwrapped_ops": stage_probe.missing},
        "paste_substages": {
            "note": (
                "p1b-p1f and p8a are this instrument's extensions; p4/p5/p10 "
                "are the repeat-to-budget micro benches of the host copies "
                "(they cannot be wrapped by name) and live in copy_benches"
            ),
            "rows": paste_rows,
            "unwrapped_ops": paste_probe.missing,
            "wall_total_s": round(sum(paste_timer.totals.values()), 3),
            "cpu_total_s": round(sum(paste_timer.cpu_totals.values()), 3),
        },
        "routing": {
            "eligible_frames": routing.eligible,
            "add_restored_face_calls": routing.add_restored_calls,
            "parse_forward_calls": parser.calls,
            "rows": routes,
        },
        "copy_benches": {
            "note": (
                "the host copies of the ROI paste and the final np.stack, at "
                "this frame size - the process CPU granularity is one system "
                "tick (~15.6 ms), so every row is repeated to >= 200 ms of "
                "wall and the repetition count rides along"
            ),
            "rows": copy_rows,
            "bench": benches,
        },
        "output_mode_comparison": comparison,
        "paste_transfer_bytes": transfer,
        "production_counters": dict(paste_mod.PASTE_TRANSFER_STATS),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--frame-hw", type=int, nargs=2, default=[360, 640],
                        metavar=("H", "W"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--comfyui-root", default=None)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    if args.comfyui_root:
        os.environ["COMFYUI_ROOT"] = args.comfyui_root
    probe = load_probe()
    comfyui_root = probe.resolve_comfyui_root(args.comfyui_root)
    if comfyui_root is None:
        print("No ComfyUI tree found: set COMFYUI_ROOT or pass --comfyui-root")
        return 2
    sys.path.insert(0, comfyui_root)
    sys.path.insert(0, str(REPO_ROOT))

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but not available")
        return 2

    report = attribute(device, args.frames, args.frame_hw[0], args.frame_hw[1],
                       warmup=args.warmup)
    text = json.dumps(report, indent=2)
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
