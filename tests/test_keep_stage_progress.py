"""The stage progress bars of the sequence path (the stage-bar design).

The helper pins (P1-P11, P16) inject the clock and the logger double - no
wall-clock value appears in an assertion.  The processor pins (P12-P14)
monkeypatch ``keep_processor.StageBar`` with a recording double and drive
the two sequence paths with the CPU-only stub packs the neighboring files
use (the stub helper of ``tests/test_keep_processor.py``, the vendored
rig of ``tests/test_multiface_sequence.py``).  P15 is the coexistence
contract: the neighboring files run green **unchanged** after the
insertions.
"""

from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

#: The source tree (the static pins read the modules as text).
SRC_DIR = Path(__file__).resolve().parent.parent / "src"

#: The vendored 5-point 512 template (``face_restoration_helper.py:52-53``).
FACE_TEMPLATE = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]]
)

#: The frame size of the multi-face rig (the minimum side is 512, so the
#: vendored ``read_image`` does not upscale and the paste canvas is the
#: frame size).
FRAME_HW = (512, 640)

#: The faces of the recurring multi-face layout - an overlapping pair and a
#: single face (the rig of ``tests/test_multiface_sequence.py``).
FACE_A = (FACE_TEMPLATE - 256.0) * 0.3 + np.array([300.0, 240.0])
FACE_B = (FACE_TEMPLATE - 256.0) * 0.3 + np.array([308.0, 252.0])
FACE_C = (FACE_TEMPLATE - 256.0) * 0.3 + np.array([200.0, 380.0])


# ---------------------------------------------------------------------------
# The clock / stream doubles of the helper pins
# ---------------------------------------------------------------------------
class _Clock:
    """The injected clock: one value per read, the last one repeats."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)
        self._index = 0

    def __call__(self) -> float:
        value = self._values[self._index] if self._index < len(self._values) else self._values[-1]
        self._index += 1
        return value

    @property
    def reads(self) -> int:
        """How many times the bar read the clock (the O(1) contract)."""
        return self._index


class _LoggerRecorder:
    """The module-logger double: it records the ``info`` payloads."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, line):
        self.lines.append(line)


# ---------------------------------------------------------------------------
# The fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stage_progress_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_stage_progress")


@pytest.fixture(scope="module")
def processor_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_processor")


@pytest.fixture(scope="module")
def helper_module():
    return importlib.import_module(
        "ComfyUI_KEEP.modules.deps.facelib.utils.face_restoration_helper")


@pytest.fixture
def make_helper(helper_module, monkeypatch):
    """A factory of real ``FaceRestoreHelper`` instances without weights.

    The detection / parsing initializations are stubbed (the weights are
    not on this machine), so every helper runs the real vendored code
    (the rig of ``tests/test_multiface_sequence.py``).
    """
    monkeypatch.setattr(helper_module, "init_detection_model", lambda *a, **k: object())
    monkeypatch.setattr(helper_module, "init_parsing_model", lambda *a, **k: object())

    def _make(use_parse: bool = False):
        return helper_module.FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model="retinaface_resnet50", save_ext="png", use_parse=use_parse,
            device="cpu", model_rootpath="unused",
        )

    return _make


# ---------------------------------------------------------------------------
# P1 - the exact line format (the logger, the injected clock)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_bar_line_format_is_pinned(stage_progress_module, monkeypatch):
    """P1 - the logged line is exactly the design's example line.

    25 of 100 frames after 10.0 s: floor percent 25, rate ``25/10`` =
    2.5 fps, ETA ``75/2.5`` = 30 s, elapsed 10 s.  The render is **one
    log record** - the tag filter of the pack logger adds the ``[KEEP]``
    tag, the root logger of ComfyUI renders the line like its own.
    """
    StageBar = stage_progress_module.StageBar
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)

    bar = StageBar("Detecting face landmarks", 100, clock=_Clock(0.0, 10.0))
    bar.tick(25)

    assert recorder.lines == [
        "Detecting face landmarks 25/100 (25%) 2.5 fps ETA 00:30 elapsed 00:10",
    ], recorder.lines


# ---------------------------------------------------------------------------
# P2 - the percent floor
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_percent_floors_never_reach_100_early(stage_progress_module, monkeypatch):
    """P2 - ``1/3`` renders ``33%``; nothing before ``done == total`` shows 100%."""
    StageBar = stage_progress_module.StageBar
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)

    bar = StageBar("Cropping and aligning faces", 3, clock=_Clock(0.0, 3.0, 6.0))
    bar.tick(1)
    bar.tick(1)

    first, second = recorder.lines
    assert " 1/3 (33%) " in first, first
    assert " 2/3 (66%) " in second, second
    assert "(100%)" not in first and "(100%)" not in second, recorder.lines


# ---------------------------------------------------------------------------
# P3 - the rate and the ETA arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_rate_and_eta_arithmetic(stage_progress_module, monkeypatch):
    """P3 - 10 frames in 4.0 s: 2.5 fps, ETA 00:12 for the remaining 30.

    ``done == 0`` renders ``--:--`` with ``0.0 fps``; ``done == total``
    renders ``ETA 00:00``.
    """
    StageBar = stage_progress_module.StageBar
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)

    bar = StageBar("Restoring faces with KEEP", 40, clock=_Clock(0.0, 4.0))
    bar.tick(10)
    assert recorder.lines[-1] == (
        "Restoring faces with KEEP 10/40 (25%) 2.5 fps ETA 00:12 elapsed 00:04"
    ), recorder.lines

    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)
    bar = StageBar("Restoring faces with KEEP", 40, clock=_Clock(0.0, 4.0))
    bar.finish()
    assert recorder.lines[-1] == (
        "Restoring faces with KEEP 0/40 (0%) 0.0 fps ETA --:-- elapsed 00:04"
    ), recorder.lines

    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)
    bar = StageBar("Restoring faces with KEEP", 10, clock=_Clock(0.0, 4.0))
    bar.tick(10)
    assert "ETA 00:00" in recorder.lines[-1], recorder.lines


# ---------------------------------------------------------------------------
# P4 - the mm:ss rendering (floor, unbounded minutes)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_mmss_rendering_is_floor_with_unbounded_minutes(stage_progress_module, monkeypatch):
    """P4 - 75 s -> ``01:15``; 3661 s -> ``61:01`` (minutes unbounded)."""
    StageBar = stage_progress_module.StageBar
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)

    # the elapsed of 75 s: 1 of 10 frames (rate 1/75, ETA 675 s = 11:15)
    bar = StageBar("Restoring faces with KEEP", 10, clock=_Clock(0.0, 75.0))
    bar.tick(1)
    assert "elapsed 01:15" in recorder.lines[-1], recorder.lines
    assert "ETA 11:15" in recorder.lines[-1], recorder.lines

    # the ETA of 3661 s: 1 face of 3662 at rate 1.0 (the interval is
    # shortened so the tick at t = 1.0 renders)
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)
    bar = StageBar(
        "Restoring faces with KEEP", 3662, interval_s=0.5,
        clock=_Clock(0.0, 1.0))
    bar.tick(1)
    assert "ETA 61:01" in recorder.lines[-1], recorder.lines
    assert "elapsed 00:01" in recorder.lines[-1], recorder.lines


# ---------------------------------------------------------------------------
# P5 - the cadence gate
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_cadence_gate_bounds_the_renders(stage_progress_module, monkeypatch):
    """P5 - the time gate fires at ``interval_s`` and bounds the renders.

    Ticks at t = 0.1 .. 1.9 render nothing; the tick at t = 2.1 renders
    exactly once; ticks every 0.01 s over 10 s render at most six times
    (the bound holds regardless of the tick rate); ``finish()`` renders
    even when no cadence render fired.
    """
    StageBar = stage_progress_module.StageBar
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)

    # ticks before the interval render nothing; the first tick past the
    # interval renders exactly once
    times = [0.1 * k for k in range(1, 20)]  # 0.1 .. 1.9
    clock = _Clock(0.0, *times, 2.1)
    bar = StageBar("Detecting face landmarks", 100, clock=clock)
    for _ in range(19):
        bar.tick(1)
    assert recorder.lines == [], recorder.lines
    bar.tick(1)  # the tick at t = 2.1
    assert len(recorder.lines) == 1, recorder.lines

    # the bound holds regardless of the tick rate: 0.01 s steps over 10 s
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)
    clock = _Clock(0.0, *[0.01 * k for k in range(1, 1001)])  # 0.01 .. 10.00
    bar = StageBar("Detecting face landmarks", 100000, clock=clock)
    for _ in range(1000):
        bar.tick(1)
    assert len(recorder.lines) <= 6, len(recorder.lines)

    # finish renders even when no cadence render fired
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)
    bar = StageBar("Detecting face landmarks", 10, clock=_Clock(0.0, 1.0))
    bar.tick(1)  # at t = 1.0 - below the interval
    assert recorder.lines == [], recorder.lines
    bar.finish()  # at t = 1.0 (the clock repeats its last value)
    assert len(recorder.lines) == 1, recorder.lines


# ---------------------------------------------------------------------------
# P6 - the final line (N/N (100%), ETA 00:00, idempotent)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_final_line_is_exact_and_finish_is_idempotent(stage_progress_module, monkeypatch):
    """P6 - the final render shows ``N/N (100%)`` with ``ETA 00:00``.

    ``finish()`` renders even when no cadence render fired, is idempotent
    (the second call renders nothing) and a ``tick()`` after ``finish()``
    is a no-op.
    """
    StageBar = stage_progress_module.StageBar
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)
    clock = _Clock(0.0, 1.0, 2.0)

    bar = StageBar("Pasting faces and finalizing frames", 5, clock=clock)
    bar.tick(5)  # at t = 1.0 - below the interval, no cadence render
    assert recorder.lines == [], recorder.lines
    bar.finish()  # at t = 2.0 - the final render

    final_line = (
        "Pasting faces and finalizing frames 5/5 (100%) 2.5 fps ETA 00:00 elapsed 00:02"
    )
    assert recorder.lines == [final_line], recorder.lines

    bar.finish()  # idempotent
    bar.tick(1)  # a no-op after finish
    assert recorder.lines == [final_line], recorder.lines


# ---------------------------------------------------------------------------
# P7 - the console hygiene (the logger owns every byte of the bar)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_bar_writes_no_console_byte(stage_progress_module, monkeypatch):
    """P7a - the rendered line carries no carriage return and no newline.

    The console mode of the earlier design repainted the line in place
    (``"\\r" + line``) on ``sys.__stdout__`` - the bytes never reached the
    logger of the host.  The line is now a plain log record: the host
    terminates it, the pack does not.
    """
    StageBar = stage_progress_module.StageBar
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)

    bar = StageBar("Detecting face landmarks", 10, clock=_Clock(0.0, 2.5, 3.0))
    bar.tick(1)
    bar.finish()

    assert recorder.lines, "the bar rendered nothing"
    for line in recorder.lines:
        assert "\r" not in line, repr(line)
        assert "\n" not in line, repr(line)
        assert "[KEEP]" not in line, "the tag belongs to the logger's filter"


@pytest.mark.unit
def test_the_bar_has_no_stream_parameter(stage_progress_module):
    """P7b - the bar takes no stream: it has no console to write to."""
    StageBar = stage_progress_module.StageBar

    with pytest.raises(TypeError):
        StageBar("Detecting face landmarks", 10, stream=object())


# ---------------------------------------------------------------------------
# P8 - the inert bar (total == 0)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_zero_total_bar_is_inert(stage_progress_module, monkeypatch):
    """P8 - a bar over zero work never renders."""
    StageBar = stage_progress_module.StageBar
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)

    bar = StageBar("Cropping and aligning faces", 0, clock=_Clock(0.0, 1.0, 2.0))
    bar.tick(1)
    bar.tick(5)
    bar.finish()

    assert recorder.lines == [], recorder.lines


# ---------------------------------------------------------------------------
# P9 - the faces unit
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_faces_unit_renders_faces_per_second(stage_progress_module, monkeypatch):
    """P9 - ``unit="faces"`` renders ``N.N faces/s``; an unknown unit fails."""
    StageBar = stage_progress_module.StageBar
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)

    # 40 of 96 faces after 4.8 s: rate 8.3 faces/s, ETA 00:06 (the
    # design's faces example)
    bar = StageBar(
        "Restoring faces with KEEP", 96, unit="faces",
        clock=_Clock(0.0, 4.8))
    bar.tick(40)
    assert recorder.lines[-1] == (
        "Restoring faces with KEEP 40/96 (41%) 8.3 faces/s ETA 00:06 elapsed 00:04"
    ), recorder.lines

    with pytest.raises(ValueError):
        StageBar("Restoring faces with KEEP", 10, unit="bogus")


# ---------------------------------------------------------------------------
# P10 - the module never touches a console stream
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_module_never_references_a_console_stream():
    """P10 - the source of the bar mentions no console stream at all.

    The static pin: the module must not import ``sys`` and must not name
    ``stdout`` / ``stderr`` - every byte of the bar leaves through the
    logger.
    """
    source = (SRC_DIR / "keep_stage_progress.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            imported.add(node.module.split(".")[0] if node.module else "")

    assert "sys" not in imported, imported

    named = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"stdout", "stderr", "__stdout__", "__stderr__"}
    ]
    assert named == [], named


# ---------------------------------------------------------------------------
# P11 - the render carries no residue of the console mode
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_render_carries_no_console_residue(stage_progress_module, monkeypatch):
    """P11 - a fast early render and a slower final render stay clean lines.

    The console mode padded the shorter render to the length of the
    previous one so no residue byte survived on the terminal.  As log
    records the lines need no padding - they are separate lines, each one
    terminated by the host.
    """
    StageBar = stage_progress_module.StageBar
    recorder = _LoggerRecorder()
    monkeypatch.setattr(stage_progress_module, "logger", recorder)
    clock = _Clock(0.0, 2.1, 2.2, 100.0)

    bar = StageBar("Detecting face landmarks", 1000000, clock=clock)
    bar.tick(900000)  # the long cadence render at t = 2.1
    bar.tick(1)  # no render at t = 2.2 (below the interval)
    bar.finish()  # the shorter final render at t = 100.0

    line1 = "Detecting face landmarks 900000/1000000 (90%) 428571.4 fps ETA 00:00 elapsed 00:02"
    line2 = "Detecting face landmarks 900001/1000000 (90%) 9000.0 fps ETA 00:11 elapsed 01:40"
    assert recorder.lines[0] == line1, recorder.lines
    assert recorder.lines[-1] == line2, recorder.lines


# ---------------------------------------------------------------------------
# P16 - the stdlib-only import pin
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_module_imports_the_stdlib_and_the_pack_logger_only():
    """P16 - the module's imports are ``time`` + the pack logger."""
    tree = ast.parse(
        (SRC_DIR / "keep_stage_progress.py").read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":  # the annotations convention
                continue
            roots.add(node.module.split(".")[0] if node.module else "")
    assert roots == {"time", "keep_logger"}, roots


# ---------------------------------------------------------------------------
# The single-face processor rig (the stubs of ``tests/test_keep_processor.py``)
# ---------------------------------------------------------------------------
class _StubKeepNet:
    """A KEEP net that records the clips it is called with."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, frames, need_upscale=False):
        self.calls.append((tuple(frames.shape), need_upscale))
        return torch.zeros_like(frames)


class _StubFaceHelper:
    """The facelib helper, reduced to the calls the processor makes."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.upscale_factor = 1.0
        self.is_gray = False
        self.cropped_faces: list = []
        self.restored_faces: list = []
        self.all_landmarks_5: list = []
        self.affine_matrices: list = []
        self.input_img = None
        self.face_template = FACE_TEMPLATE
        self.calls: list[str] = []

    def clean_all(self):
        self.calls.append("clean_all")
        self.cropped_faces = []
        self.restored_faces = []

    # keep-fix5-implement1 (2026-11-03): the vendored ``read_image`` gained
    # the ``check_gray`` flag (plan 2.2) - the sequence sites pass it.
    def read_image(self, image, check_gray: bool = True):
        self.calls.append("read_image")
        # the vendored helper upscales when min(h, w) < 512
        # (``face_restoration_helper.py:179-184``) - the stub reproduces it
        self.input_img = image
        if min(image.shape[:2]) < 512:
            factor = 512.0 / min(image.shape[:2])
            self.input_img = cv2.resize(
                image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

    def add_restored_face(self, face, **kwargs):
        self.calls.append("add_restored_face")
        self.restored_faces = [face]

    def get_inverse_affine(self, *args, **kwargs):
        self.calls.append("get_inverse_affine")

    def paste_faces_to_input_image(self, *args, **kwargs):
        self.calls.append("paste_faces_to_input_image")
        return np.zeros((512, 512, 3), dtype=np.uint8)


class _DetectingFaceHelper(_StubFaceHelper):
    """A face helper that detects one face per frame (degenerate zeros)."""

    def __init__(self) -> None:
        super().__init__()
        self.landmarks = np.zeros(10, dtype=np.float32)

    def get_face_landmarks_5(self, **kwargs):
        self.calls.append("get_face_landmarks_5")
        self.all_landmarks_5 = [self.landmarks]
        return 1


class _BlankFaceHelper(_DetectingFaceHelper):
    """A face helper that detects nothing (the all-NaN clip of P14)."""

    def get_face_landmarks_5(self, **kwargs):
        self.calls.append("get_face_landmarks_5")
        self.all_landmarks_5 = []
        return 0


def _make_pack(keep_net, face_helper):
    """A model pack the processor can consume."""
    return types.SimpleNamespace(
        keep_net=keep_net,
        face_helper=face_helper,
        bg_upscale_model=None,
        face_upscale_model=None,
        device=torch.device("cpu"),
        model_type_str="KEEP",
    )


# ---------------------------------------------------------------------------
# The multi-face processor rig (the vendored rig of
# ``tests/test_multiface_sequence.py``)
# ---------------------------------------------------------------------------
class _StubParseNet:
    """A parse net with the vendored contract (``[0]`` are the logits)."""

    def __init__(self, seed: int = 5) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(19, 3, 1, 1, generator=generator) * 0.35

    def __call__(self, samples):
        return (F.conv2d(samples, self.weight), None)


class _KeepNet:
    """A KEEP net that returns its input (the restored face is the crop)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, frames, need_upscale=False):
        self.calls.append((tuple(frames.shape), need_upscale))
        return frames


class _SequenceDetector:
    """A face detector with per-frame face sets (the appearing clip)."""

    def __init__(self, per_frame_faces: list[list[np.ndarray]]) -> None:
        scale = max(1, 640 / min(FRAME_HW))
        self._per_frame: list[np.ndarray | None] = []
        for faces in per_frame_faces:
            rows = []
            for landmarks in faces:
                (x1, y1), (x2, y2) = landmarks.min(axis=0), landmarks.max(axis=0)
                rows.append(np.concatenate(([x1, y1, x2, y2, 0.99], landmarks.reshape(-1))))
            self._per_frame.append(np.stack(rows) * scale if rows else None)
        self._index = 0

    def detect_faces(self, image, confidence):
        rows = self._per_frame[self._index]
        self._index += 1
        return rows


def _smooth_frames(num_frames: int, seed: int = 41) -> np.ndarray:
    """A clip of smooth frames (the warp residual stays inside its bound)."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(num_frames):
        small = rng.uniform(0.0, 255.0, size=(32, 32, 3)).astype(np.float32)
        smooth = cv2.resize(small, (FRAME_HW[1], FRAME_HW[0]), interpolation=cv2.INTER_CUBIC)
        frames.append(np.clip(smooth, 0, 255).astype(np.uint8))
    return np.stack(frames)


def _sequence_tensor(frames_bgr: np.ndarray) -> torch.Tensor:
    """The BGR frames as the ComfyUI IMAGE tensor the node hands over."""
    image_utils = importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")
    return image_utils.bgr_u8_batch_to_comfy(frames_bgr)


def _attach_detector(face_helper, detector):
    face_helper.face_detector = detector


def _run_sequence(keep_net, face_helper, frames, **kwargs):
    """One ``process_image_sequence`` run through the multi-face branch."""
    processor_module = importlib.import_module("ComfyUI_KEEP.src.keep_processor")
    processor = processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))
    return processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=kwargs.pop("final_upscale_factor", 1.0),
        has_aligned_frames=kwargs.pop("has_aligned_frames", False),
        only_center_face=kwargs.pop("only_center_face", False),
        draw_box=kwargs.pop("draw_box", False),
        max_clip_length=kwargs.pop("max_clip_length", 20),
    )


# ---------------------------------------------------------------------------
# P12 - the single-face counts
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_single_face_stage_bars_count_the_stages(processor_module, monkeypatch):
    """P12 - one bar per stage, cumulative across the KEEP chunks.

    5 frames with ``max_clip_length=2``: the detection ticks are 5x1; the
    crop ticks sum to the frame count (all landmarks valid in the stub);
    the restore ticks are the chunk sizes ``[2, 2, 1]``; the paste ticks
    are 5x1.  Exactly one ``finish()`` per bar, ordered stage after
    stage, and the V3 reporter keeps its exact G6 sequence (the delta
    form - cumulative it is the ``[5, 10, 12, 14, 15, 20, 21, 22, 23, 24,
    25]`` of the G6 pin).
    """
    bars: list = []
    progress_events: list = []

    class _RecordingStageBar:
        """The StageBar double: it records every call the processor makes."""

        def __init__(self, label, total, *, unit="frames",
                     interval_s=None, clock=None):
            self.label = label
            self.total = total
            self.unit = unit
            self.ticks: list[int] = []
            self.finishes = 0
            self.ops: list[str] = []
            bars.append(self)

        def tick(self, n=1):
            self.ops.append(("tick", n))
            self.ticks.append(n)

        def finish(self):
            self.ops.append(("finish",))
            self.finishes += 1

    class _RecordingProgress:
        def __init__(self, total):
            progress_events.append(("total", total))
            self.total = total

        def update(self, value):
            progress_events.append(("update", value))

    monkeypatch.setattr(processor_module, "StageBar", _RecordingStageBar)
    monkeypatch.setattr(processor_module, "ProgressReporter", _RecordingProgress)

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _DetectingFaceHelper())
    )
    frames = torch.rand((5, 16, 16, 3), dtype=torch.float32)
    processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=2,
    )

    assert [bar.label for bar in bars] == [
        "Detecting face landmarks",
        "Cropping and aligning faces",
        "Restoring faces with KEEP",
        "Pasting faces and finalizing frames",
    ], [bar.label for bar in bars]
    assert [bar.unit for bar in bars] == ["frames", "frames", "frames", "frames"], bars
    detection, crop, restore, paste = bars
    assert detection.ticks == [1, 1, 1, 1, 1], detection.ticks
    assert crop.ticks == [1, 1, 1, 1, 1], crop.ticks
    assert restore.ticks == [2, 2, 1], restore.ticks
    assert paste.ticks == [1, 1, 1, 1, 1], paste.ticks
    assert [bar.total for bar in bars] == [5, 5, 5, 5], [bar.total for bar in bars]
    # one finish per bar, and every bar's last op is its finish
    assert [bar.finishes for bar in bars] == [1, 1, 1, 1], [bar.finishes for bar in bars]
    assert [bar.ops[-1] for bar in bars] == [("finish",)] * 4, [bar.ops[-1] for bar in bars]
    # the V3 reporter keeps its exact sequence (the G6 values, delta form)
    assert progress_events == [
        ("total", 25),
        ("update", 5),  # the batched convert-in
        ("update", 5),  # the detection
        ("update", 2), ("update", 2), ("update", 1),  # the KEEP chunks
        ("update", 5),  # the paste stage's second convert-in (D-R2.3.1)
        ("update", 1), ("update", 1), ("update", 1), ("update", 1), ("update", 1),  # the paste
    ], progress_events


# ---------------------------------------------------------------------------
# P13 - the multi-face counts
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_multiface_stage_bars_count_the_stages(
    processor_module, make_helper, monkeypatch
):
    """P13 - the same over ``_process_sequence_multiface``, in faces.

    5 frames with the layout ``[[A, B], [C], [], [A], [C]]``: the
    detection ticks are 5x1 (frames); the crop is ONE tick equal to the
    face count (``unit="faces"``); the restore per-chunk ticks sum to the
    face count (``[2, 2, 1]`` at ``max_clip_length=2``).  Round 10 (the
    owner's fix 4, 2026-10-31): the paste bar is threaded into the
    batched multiface paste - the four batched rows tick per completed
    group (``[4]``, one group at this canvas) and the face-less frame 2
    ticks in the loop (``[1]``), so the bar still totals the frame count
    exactly (previously the stage ticked every row in the loop,
    ``[1, 1, 1, 1, 1]``).  The V3 reporter keeps its exact sequence and
    its faces-aware total.
    """
    bars: list = []
    progress_events: list = []
    progress_instances: list = []

    class _RecordingStageBar:
        def __init__(self, label, total, *, unit="frames",
                     interval_s=None, clock=None):
            self.label = label
            self.total = total
            self.unit = unit
            self.ticks: list[int] = []
            self.finishes = 0
            self.ops: list[str] = []
            bars.append(self)

        def tick(self, n=1):
            self.ops.append(("tick", n))
            self.ticks.append(n)

        def finish(self):
            self.ops.append(("finish",))
            self.finishes += 1

    class _RecordingProgress:
        def __init__(self, total):
            progress_events.append(("total", total))
            progress_instances.append(self)
            self.total = total

        def update(self, value):
            progress_events.append(("update", value))

    monkeypatch.setattr(processor_module, "StageBar", _RecordingStageBar)
    monkeypatch.setattr(processor_module, "ProgressReporter", _RecordingProgress)

    frames = _sequence_tensor(_smooth_frames(5))
    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet(seed=137)
    _attach_detector(
        face_helper, _SequenceDetector([[FACE_A, FACE_B], [FACE_C], [], [FACE_A], [FACE_C]]))

    _run_sequence(_KeepNet(), face_helper, frames, max_clip_length=2)

    assert [bar.label for bar in bars] == [
        "Detecting face landmarks",
        "Cropping and aligning faces",
        "Restoring faces with KEEP",
        "Pasting faces and finalizing frames",
    ], [bar.label for bar in bars]
    assert [bar.unit for bar in bars] == ["frames", "faces", "faces", "frames"], bars
    detection, crop, restore, paste = bars
    assert detection.ticks == [1, 1, 1, 1, 1], detection.ticks
    assert crop.ticks == [5], crop.ticks  # ONE tick of the face count
    assert restore.ticks == [2, 2, 1], restore.ticks
    # the batched rows tick per group in the paste core, the face-less
    # frame ticks in the loop - the sum stays the frame count
    assert paste.ticks == [4, 1], paste.ticks
    assert sum(paste.ticks) == paste.total, paste.ticks
    assert [bar.total for bar in bars] == [5, 5, 5, 5], [bar.total for bar in bars]
    assert [bar.finishes for bar in bars] == [1, 1, 1, 1], [bar.finishes for bar in bars]
    assert [bar.ops[-1] for bar in bars] == [("finish",)] * 4, [bar.ops[-1] for bar in bars]
    # the V3 reporter keeps its exact sequence; the total counts the
    # restore in faces (``num_input_frames * 4 + total_faces``)
    assert progress_events == [
        ("total", 20),  # the frames-only estimate of the reporter (5 * 4)
        ("update", 5),  # the batched convert-in
        ("update", 5),  # the detection
        ("update", 2), ("update", 2), ("update", 1),  # the KEEP chunks
        ("update", 5),  # the paste stage's second convert-in (D-R2.3.1)
        ("update", 1), ("update", 1), ("update", 1), ("update", 1), ("update", 1),  # the paste
    ], progress_events
    assert progress_instances and progress_instances[0].total == 4 * 5 + 5, (
        f"the final total counts the restore in faces, got {progress_instances[0].total}"
    )


# ---------------------------------------------------------------------------
# P14 - the early returns carry the final line
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_all_nan_clip_finishes_the_detection_bar(processor_module, monkeypatch):
    """P14a - the all-NaN clip (single-face) finishes the detection bar.

    The early return carries the final line: exactly one bar exists (the
    detection), it ticks every frame and its last op is the ``finish``
    before the return.
    """
    bars: list = []

    class _RecordingStageBar:
        def __init__(self, label, total, *, unit="frames",
                     interval_s=None, clock=None):
            self.label = label
            self.total = total
            self.unit = unit
            self.ticks: list[int] = []
            self.finishes = 0
            self.ops: list[str] = []
            bars.append(self)

        def tick(self, n=1):
            self.ops.append(("tick", n))
            self.ticks.append(n)

        def finish(self):
            self.ops.append(("finish",))
            self.finishes += 1

    monkeypatch.setattr(processor_module, "StageBar", _RecordingStageBar)

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _BlankFaceHelper())
    )
    frames = torch.rand((2, 16, 16, 3), dtype=torch.float32)
    result = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    assert result is frames, "the all-NaN clip returns the input"
    assert [bar.label for bar in bars] == ["Detecting face landmarks"], bars
    assert bars[0].ticks == [1, 1], bars[0].ticks
    assert bars[0].finishes == 1, bars[0].finishes
    assert bars[0].ops[-1] == ("finish",), bars[0].ops


@pytest.mark.unit
def test_the_zero_faces_clip_finishes_the_detection_bar(
    processor_module, make_helper, monkeypatch
):
    """P14b - the zero-faces clip (multi-face) finishes the detection bar."""
    bars: list = []

    class _RecordingStageBar:
        def __init__(self, label, total, *, unit="frames",
                     interval_s=None, clock=None):
            self.label = label
            self.total = total
            self.unit = unit
            self.ticks: list[int] = []
            self.finishes = 0
            self.ops: list[str] = []
            bars.append(self)

        def tick(self, n=1):
            self.ops.append(("tick", n))
            self.ticks.append(n)

        def finish(self):
            self.ops.append(("finish",))
            self.finishes += 1

    monkeypatch.setattr(processor_module, "StageBar", _RecordingStageBar)

    frames = _sequence_tensor(_smooth_frames(2))
    face_helper = make_helper(use_parse=True)
    _attach_detector(face_helper, _SequenceDetector([[], []]))

    result = _run_sequence(_KeepNet(), face_helper, frames)

    assert result is frames, "the zero-faces clip returns the input"
    assert [bar.label for bar in bars] == ["Detecting face landmarks"], bars
    assert bars[0].ticks == [1, 1], bars[0].ticks
    assert bars[0].finishes == 1, bars[0].finishes
    assert bars[0].ops[-1] == ("finish",), bars[0].ops
