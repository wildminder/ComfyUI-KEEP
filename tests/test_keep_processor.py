"""P3.2 - the face processor of ``src/keep_processor.py``.

The tests drive the processor with **stubs** (a stub face helper and a stub
KEEP net) and assert the things the processor is responsible for:

* the chunk math of the sequence path (``max_clip_length``),
* the paste-back (the output frame count and shape),
* the ``tiled_scale`` wrapper of the optional ComfyUI upscalers,
* the **console routing**: the four steps are logged and the console writes
  of the vendored facelib reach the logger of the pack.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import types
import weakref
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def processor_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_processor")


class _StubKeepNet:
    """A KEEP net that records the clips it is called with."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, frames, need_upscale=False):
        self.calls.append((tuple(frames.shape), need_upscale))
        return torch.zeros_like(frames)


#: The vendored 5-point 512 template (``face_restoration_helper.py:52-53``).
FACE_TEMPLATE_512 = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]]
)


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
        # the template the vendored helper estimates the crop affines with
        self.face_template = FACE_TEMPLATE_512
        self.calls: list[str] = []

    def clean_all(self):
        self.calls.append("clean_all")
        self.cropped_faces = []
        self.restored_faces = []

    # keep-fix5-implement1 (2026-11-03): the vendored ``read_image`` gained
    # the ``check_gray`` flag (plan Step 2.2) - the sequence sites pass it.
    def read_image(self, image, check_gray: bool = True):
        self.calls.append("read_image")
        # the vendored helper keeps the image it read and upscales it when
        # min(h, w) < 512 (``face_restoration_helper.py:179-184``) - the stub
        # reproduces the upscale, or the paste canvas of the small frames
        # would stay at the raw size (the round-1 defect).
        self.input_img = image
        if min(image.shape[:2]) < 512:
            factor = 512.0 / min(image.shape[:2])
            self.input_img = cv2.resize(
                image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

    def align_warp_face(self):
        self.calls.append("align_warp_face")

    def add_restored_face(self, face, **kwargs):
        self.calls.append("add_restored_face")
        self.restored_faces = [face]

    def get_inverse_affine(self, *args, **kwargs):
        self.calls.append("get_inverse_affine")

    def paste_faces_to_input_image(self, *args, **kwargs):
        self.calls.append("paste_faces_to_input_image")
        return np.zeros((512, 512, 3), dtype=np.uint8)


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


@pytest.mark.unit
def test_the_sequence_is_split_into_clips_of_max_clip_length(processor_module):
    """The chunk math of the sequence path (``max_clip_length``)."""
    keep_net = _StubKeepNet()
    face_helper = _StubFaceHelper()
    processor = processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))

    frames = torch.rand((4, 16, 16, 3), dtype=torch.float32)
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=True,
        only_center_face=True,
        draw_box=False,
        max_clip_length=2,
    )

    # 4 frames, clips of 2 -> two calls of two frames each
    assert len(keep_net.calls) == 2, keep_net.calls
    for shape, need_upscale in keep_net.calls:
        assert shape == (1, 2, 3, 512, 512), shape
        assert need_upscale is False
    # the paste-back produced one frame per input frame
    assert tuple(output.shape) == (4, 512, 512, 3), output.shape


@pytest.mark.unit
def test_an_empty_sequence_is_returned_unchanged(processor_module):
    """No frames -> no work, the input is returned."""
    keep_net = _StubKeepNet()
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(keep_net, _StubFaceHelper())
    )
    frames = torch.zeros((0, 16, 16, 3), dtype=torch.float32)
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=True,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )
    assert output is frames
    assert keep_net.calls == []


@pytest.mark.unit
def test_the_upscaler_goes_through_tiled_scale(processor_module):
    """The optional ComfyUI upscaler is called through ``tiled_scale``."""
    calls: list[dict] = []

    class _StubUpscaleModel:
        scale = 2.0

        def __init__(self):
            self.model = self._model

        def _model(self, samples):
            calls.append({"shape": tuple(samples.shape)})
            return samples * 2.0

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _StubFaceHelper())
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = processor._run_upscaler(_StubUpscaleModel(), image)

    assert len(calls) == 1, calls
    assert calls[0]["shape"] == (1, 3, 8, 8), calls
    assert result.shape == (8, 8, 3), result.shape


@pytest.mark.unit
def test_without_an_upscaler_the_image_is_untouched(processor_module):
    """No upscaler -> the image is returned unchanged."""
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _StubFaceHelper())
    )
    image = np.full((4, 4, 3), 7, dtype=np.uint8)
    assert processor._run_upscaler(None, image) is image


@pytest.mark.unit
def test_thirty_frames_are_split_into_clips_of_twenty(processor_module):
    """The chunk math of the sequence path for 30 frames / ``max_clip_length=20``."""
    keep_net = _StubKeepNet()
    face_helper = _StubFaceHelper()
    processor = processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))

    frames = torch.rand((30, 16, 16, 3), dtype=torch.float32)
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=True,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    # 30 frames, clips of 20 -> one call of 20 frames and one of 10
    assert len(keep_net.calls) == 2, keep_net.calls
    shapes = [shape for shape, _ in keep_net.calls]
    assert shapes == [(1, 20, 3, 512, 512), (1, 10, 3, 512, 512)], shapes
    assert tuple(output.shape) == (30, 512, 512, 3), output.shape


# ---------------------------------------------------------------------------
# P4.6 - the VRAM is released through ComfyUI
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_vram_is_released_through_comfyui(processor_module):
    """``_soft_empty_cache`` goes through ``model_management`` (P4.6)."""
    model_management = importlib.import_module("comfy.model_management")
    model_management.calls.clear()

    assert processor_module._soft_empty_cache() is True
    assert ("soft_empty_cache",) in model_management.calls, model_management.calls


@pytest.mark.mock_integration
def test_the_sequence_releases_the_vram_once_per_run(
    processor_module, monkeypatch
):
    """P4.6 (Opt 6) + round 7 (the paste-budget repair): the release cadence.

    30 frames with ``max_clip_length=20`` are two chunks - the chunk
    boundaries do **not** release (the chunk shapes repeat, so the caching
    allocator reuses its blocks instead of accumulating).  A clean run now
    releases **twice**:

    1. the round-7 pre-paste release - after the restore loop's last
       ``del`` and before the paste stage: the paste-mode verdict
       (``device_output_mode``, ``keep_gpu_paste.py``) reads the **free**
       VRAM through ComfyUI's ``get_free_memory(..., torch_free_too=True)``,
       which counts what the torch cache can give back, so the cached
       restore blocks hoard exactly the memory the verdict reads - the
       budget shrinks, the mode flips to ``host`` and the paste takes the
       host-CPU assembly (the round-6 owner's paste-stage stall).  The
       release restores the pre-Opt-6 budget before the verdict, and
    2. the ``finally`` release when the run is done.

    An OOM run still releases twice for a different reason: the ``except``
    handler before the ``raise``, then the ``finally`` - the pre-paste
    site is not reached on OOM (the sibling OOM pin below).

    (The round-6 id is kept for traceability with the run docs; the
    cadence it pins changed in round 7.)
    """
    events: list[str] = []

    class _RecordingKeepNet(_StubKeepNet):
        def __call__(self, frames, need_upscale=False):
            events.append("net")
            return super().__call__(frames, need_upscale=need_upscale)

    monkeypatch.setattr(
        processor_module, "_soft_empty_cache", lambda: events.append("empty_cache")
    )

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_RecordingKeepNet(), _StubFaceHelper())
    )
    frames = torch.rand((30, 16, 16, 3), dtype=torch.float32)
    processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=True,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    assert events == ["net", "net", "empty_cache", "empty_cache"], events


@pytest.mark.mock_integration
def test_an_oom_in_the_restore_loop_releases_the_vram_and_propagates(
    processor_module, monkeypatch
):
    """Opt 6: the OOM handler releases the cached VRAM, then fails loudly.

    The stub net raises ``torch.cuda.OutOfMemoryError`` (constructible
    without CUDA) inside the fused crop + restore loop; the release fires
    and the exception leaves ``process_image_sequence`` unchanged.  The
    recorder fires **twice** (the round-6 owner answer keeps §6.2
    verbatim): once in the ``except`` handler before the ``raise``, once
    in the ``finally`` - both idempotent, and the happy path still fires
    exactly once (the pin above).
    """
    events: list[str] = []

    class _OomKeepNet(_StubKeepNet):
        def __call__(self, frames, need_upscale=False):
            raise torch.cuda.OutOfMemoryError()

    monkeypatch.setattr(
        processor_module, "_soft_empty_cache",
        lambda: events.append("empty_cache"),
    )

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_OomKeepNet(), _StubFaceHelper())
    )
    frames = torch.rand((4, 16, 16, 3), dtype=torch.float32)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        processor.process_image_sequence(
            image_sequence_tensor=frames,
            final_upscale_factor=1.0,
            has_aligned_frames=True,
            only_center_face=True,
            draw_box=False,
            max_clip_length=20,
        )

    assert events == ["empty_cache", "empty_cache"], events


@pytest.mark.mock_integration
def test_the_pre_paste_release_fires_before_the_paste_stage(
    processor_module, monkeypatch
):
    """Round 7 (the paste-budget repair): the release precedes the paste stage.

    On the single-face path the pre-paste release must land **before** the
    paste stage begins: the paste helper (``_batched_paste``) is the first
    paste-stage consult on the default branch (``has_aligned_frames=False``)
    and its mode verdict (``device_output_mode``) reads the cache-inclusive
    free VRAM - a release after the verdict would be too late.  The helper
    records its consult; the release event must precede it (one chunk - 3
    frames, one ``net`` call; the ``finally`` release still closes the run).
    """
    events: list[str] = []

    class _RecordingKeepNet(_StubKeepNet):
        def __call__(self, frames, need_upscale=False):
            events.append("net")
            return super().__call__(frames, need_upscale=need_upscale)

    monkeypatch.setattr(
        processor_module, "_soft_empty_cache", lambda: events.append("empty_cache")
    )
    real_batched_paste = processor_module.KEEPFaceProcessor._batched_paste

    def _recording_batched_paste(self, *args, **kwargs):
        events.append("paste")
        return real_batched_paste(self, *args, **kwargs)

    monkeypatch.setattr(
        processor_module.KEEPFaceProcessor, "_batched_paste", _recording_batched_paste
    )

    landmarks = (FACE_TEMPLATE_512 - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _LandmarkFaceHelper(landmarks)
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_RecordingKeepNet(), face_helper)
    )
    frames = torch.rand((3, 64, 64, 3), dtype=torch.float32)
    processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    assert events == ["net", "empty_cache", "paste", "empty_cache"], events


# ---------------------------------------------------------------------------
# D4 - the console routing of the processor
# ---------------------------------------------------------------------------
class _DetectingFaceHelper(_StubFaceHelper):
    """A face helper that **detects** a face - and prints like the facelib.

    The vendored helper writes to the console (``print()``) while it reads
    the image, while it detects and while it pastes a face back: the
    processor must route those writes to the logger of the pack.

    The zero landmarks of the stub are a degenerate set - the crop of the
    redesigned path estimates the affine with the real vendored call, and
    the ``LMEDS`` failure of that call is what prints the warning of the
    facelib.
    """

    def __init__(self) -> None:
        super().__init__()
        self.landmarks = np.zeros(10, dtype=np.float32)

    def get_face_landmarks_5(self, **kwargs):
        self.calls.append("get_face_landmarks_5")
        print("model_rootpath is None. Using default relative path")
        self.all_landmarks_5 = [self.landmarks]
        return 1

    def paste_faces_to_input_image(self, *args, **kwargs):
        super().paste_faces_to_input_image(*args, **kwargs)
        print("Warning: Mismatch between restored_faces and inverse_affine_matrices.")
        return np.zeros((512, 512, 3), dtype=np.uint8)


@contextlib.contextmanager
def _pack_records():
    """The records the logger of the pack receives while the block runs."""
    pack_logger = importlib.import_module("ComfyUI_KEEP.src.keep_logger").logger
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    pack_logger.addHandler(handler)
    try:
        yield records
    finally:
        pack_logger.removeHandler(handler)


@pytest.mark.unit
def test_the_four_steps_of_the_sequence_are_logged(processor_module):
    """The sequence path logs its four steps - tagged, once each."""
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _DetectingFaceHelper())
    )
    frames = torch.rand((2, 16, 16, 3), dtype=torch.float32)

    with _pack_records() as records:
        processor.process_image_sequence(
            image_sequence_tensor=frames,
            final_upscale_factor=1.0,
            has_aligned_frames=False,
            only_center_face=True,
            draw_box=False,
        )

    steps = [
        "[KEEP] Detecting face landmarks",
        "[KEEP] Cropping and aligning faces",
        "[KEEP] Restoring faces with KEEP",
        "[KEEP] Pasting faces and finalizing frames",
    ]
    messages = [record.getMessage() for record in records]
    # the records of the console routing are in between - the four steps are
    # logged once each and in this order
    assert [message for message in messages if message in steps] == steps


@pytest.mark.unit
def test_the_console_writes_of_the_detection_reach_the_logger(processor_module):
    """A print of the vendored detector becomes a tagged record."""
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _DetectingFaceHelper())
    )
    frames = torch.rand((1, 16, 16, 3), dtype=torch.float32)

    with _pack_records() as records:
        processor.process_image_sequence(
            image_sequence_tensor=frames,
            final_upscale_factor=1.0,
            has_aligned_frames=False,
            only_center_face=True,
            draw_box=False,
        )

    messages = [record.getMessage() for record in records]
    assert (
        "[KEEP] model_rootpath is None. Using default relative path" in messages
    ), messages


@pytest.mark.unit
def test_every_console_write_of_the_facelib_reaches_the_logger(
    processor_module, monkeypatch
):
    """The prints of the detection, of the alignment and of the paste-back.

    Fix 1.1 (2026-10-30) split the corpus's fates: the None-affine frame
    (the LMEDS failure on the stub's degenerate zero landmarks)
    short-circuits to the background-only row since the owner's critical
    fix - it still prints the alignment warning, but it never reaches the
    vendored paste, whose print this pin exists to route.  The corpus
    gained a second frame whose affine estimate is valid (the estimate
    patched to ``[None, valid]``) so the paste-back print keeps firing;
    the stub's paste return is canvas-sized (the background-only row and
    the vendored row must stack).

    Round 11 (2026-10-31): the short-circuit row is canvas-sized since the
    short-circuit resize (the fix-1.1 snippet's raw output cannot join the
    IMAGE batch at a mismatched shape), so the stub's return is the real
    512x512 canvas the vendored paste assembles from ``input_img`` - the
    pin's routing assertions are unchanged.
    """
    real_estimate = processor_module.estimate_affines

    def none_then_valid(landmark_list, template):
        result = list(real_estimate(landmark_list, template))
        result[0] = None
        result[1] = np.float32([[1, 0, 0], [0, 1, 0]])
        return result

    monkeypatch.setattr(processor_module, "estimate_affines", none_then_valid)
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _DetectingFaceHelper())
    )
    face_helper = processor.face_helper
    # the stub's paste returns 512x512 - canvas-sized, so the background-only
    # row and the vendored row stack; the print stays (it is the routing probe)
    def canvas_sized_paste(*args, **kwargs):
        print("Warning: Mismatch between restored_faces and inverse_affine_matrices.")
        return np.zeros((512, 512, 3), dtype=np.uint8)

    face_helper.paste_faces_to_input_image = canvas_sized_paste
    frames = torch.rand((2, 16, 16, 3), dtype=torch.float32)

    with _pack_records() as records:
        processor.process_image_sequence(
            image_sequence_tensor=frames,
            final_upscale_factor=1.0,
            has_aligned_frames=False,
            only_center_face=True,
            draw_box=False,
        )

    messages = [record.getMessage() for record in records]
    for text in (
        "model_rootpath is None. Using default relative path",
        "Warning: Could not estimate affine matrix for landmark set 0 (frame 0). Skipping this face.",
        "Warning: Mismatch between restored_faces and inverse_affine_matrices.",
    ):
        assert f"[KEEP] {text}" in messages, (text, messages)


@pytest.mark.unit
def test_the_console_writes_of_the_image_path_reach_the_logger(processor_module):
    """The single image path routes the detection prints as well."""
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _DetectingFaceHelper())
    )
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    with _pack_records() as records:
        processor.process_image(
            cv2_image_orig=image,
            final_upscale_factor=1.0,
            has_aligned=False,
            only_center_face=True,
            draw_box=False,
        )

    messages = [record.getMessage() for record in records]
    assert (
        "[KEEP] model_rootpath is None. Using default relative path" in messages
    ), messages


# ---------------------------------------------------------------------------
# The GPU video pipeline redesign - the batched sequence path
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_single_frame_chunk_is_duplicated_for_the_net(processor_module):
    """The 1-frame chunk rule of the KEEP contract.

    A chunk of one frame is fed to the net as two copies and the first
    output frame is kept - the chunk boundaries are a quality contract and
    must not move.
    """
    keep_net = _StubKeepNet()
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(keep_net, _StubFaceHelper())
    )

    frames = torch.rand((1, 16, 16, 3), dtype=torch.float32)
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=True,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    assert len(keep_net.calls) == 1, keep_net.calls
    shape, need_upscale = keep_net.calls[0]
    assert shape == (1, 2, 3, 512, 512), shape
    assert need_upscale is False
    assert tuple(output.shape) == (1, 512, 512, 3), output.shape


@pytest.mark.unit
def test_the_sequence_reads_and_measures_every_frame_once(processor_module):
    """Read once, affine once.

    The redesigned path reads every frame a single time (the detection) and
    estimates its affine a single time (the crop) - the paste loop reuses
    the matrix instead of running a second ``align_warp_face``.
    """
    keep_net = _StubKeepNet()
    face_helper = _DetectingFaceHelper()
    processor = processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))

    # The frames are at the 512 floor: the paste fallback's deterministic
    # re-derivation of the vendored upscale (D-R2.3.3) fires only below 512
    # and would add one read per fallback frame - the read-once contract of
    # the detection is pinned at the size where it holds for the whole run.
    frames = torch.rand((3, 512, 640, 3), dtype=torch.float32)
    processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    assert face_helper.calls.count("read_image") == 3, face_helper.calls
    assert "align_warp_face" not in face_helper.calls, face_helper.calls


@pytest.mark.unit
def test_the_progress_reports_four_steps_per_frame(processor_module, monkeypatch):
    """The progress contract of the restructured path.

    The total stays ``num_frames * 4`` and the four families of the updates
    stay: the conversion (one batched update), the detection (one update),
    the KEEP chunks (one update per chunk) and the paste (one update per
    frame).
    """
    events: list = []

    class _RecordingProgress:
        def __init__(self, total):
            events.append(("total", total))

        def update(self, value):
            events.append(("update", value))

    monkeypatch.setattr(processor_module, "ProgressReporter", _RecordingProgress)

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _DetectingFaceHelper())
    )
    frames = torch.rand((3, 16, 16, 3), dtype=torch.float32)
    processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    assert events == [
        ("total", 15),
        ("update", 3),  # the batched convert-in
        ("update", 3),  # the detection
        ("update", 3),  # the single KEEP chunk
        ("update", 3),  # the paste stage's second convert-in (D-R2.3.1)
        ("update", 1),  # the paste, frame by frame
        ("update", 1),
        ("update", 1),
    ], events


class _LandmarkFaceHelper(_StubFaceHelper):
    """A face helper that detects a face at real, non-degenerate landmarks."""

    def __init__(self, landmarks) -> None:
        super().__init__()
        self.landmarks = np.asarray(landmarks, dtype=np.float64).reshape(10)

    def get_face_landmarks_5(self, **kwargs):
        self.calls.append("get_face_landmarks_5")
        self.all_landmarks_5 = [self.landmarks]
        return 1


class _RecordingKeepNet(_StubKeepNet):
    """A KEEP net that keeps the frames it was called with."""

    def __init__(self) -> None:
        super().__init__()
        self.frames: list = []

    def __call__(self, frames, need_upscale=False):
        self.frames.append(frames.detach().clone())
        return super().__call__(frames, need_upscale=need_upscale)


@pytest.mark.unit
def test_the_batched_crop_matches_the_vendored_warp(processor_module):
    """The crop of the redesigned path equals the vendored ``cv2.warpAffine``.

    The reference is the warp of ``face_restoration_helper.py:326-328`` (the
    512 template, the gray border) on the very frame the detector saw,
    followed by the ``img2tensor`` + normalize tail of the processor.  The
    crop itself warps on the device (Opt 1): one batched
    ``crop_faces_batch`` per source-shape group replaced the per-row
    ``cv2.warpAffine``, and the measured residual of that warp against cv2
    is at most **1/255** on this content
    (``tests/test_gpu_batch_equivalence.py``); the bound below is that
    residual in the normalized scale, doubled for headroom.

    The normalize tail is **float32** (Opt 3): its deviation from the
    float64 ``img2tensor`` chain of the reference is one float32 rounding of
    the ``u8 / 255`` quotient (<= 2**-22 in this scale, pinned in
    ``tests/test_gpu_batch_equivalence.py``) - two orders of magnitude below
    the warp bound, so the two changes compose without re-tuning.
    The **smoothed** landmarks Opt 3 feeds the affine estimate add their own
    sub-pixel share (measured ~2e-4 on the matrix, a 1-frame clip's 1-ulp
    mean); the reuse assertion below carries that bound explicitly.

    The test also pins the **affine reuse**: the paste loop receives the
    matrix the crop computed - the second ``align_warp_face`` of the old
    path is gone.
    """
    height, width = 512, 640
    rng = np.random.default_rng(4)
    small = rng.uniform(0.0, 255.0, size=(16, 20, 3)).astype(np.float32)
    frame = np.clip(
        cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC), 0, 255
    ).astype(np.uint8)
    landmarks = FACE_TEMPLATE_512 + np.array([100.0, 50.0])

    face_helper = _LandmarkFaceHelper(landmarks)
    keep_net = _RecordingKeepNet()
    processor = processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))

    frames = torch.from_numpy(frame.astype(np.float32) / 255.0).unsqueeze(0)
    processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    # the crop the net was called with (the 1-frame chunk is duplicated)
    assert len(keep_net.frames) == 1, keep_net.frames
    cropped = keep_net.frames[0][0, 0]
    assert tuple(cropped.shape) == (3, 512, 512), cropped.shape

    # the reference: the vendored warp + the img2tensor tail of the processor
    # the warp source is the BGR conversion of the input (the converter of
    # the pack - the same array the crop warps)
    image_utils = importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")
    frame_bgr = image_utils.comfy_batch_to_bgr_u8(frames)[0]
    affine = cv2.estimateAffinePartial2D(
        landmarks, FACE_TEMPLATE_512, method=cv2.LMEDS
    )[0]
    reference_u8 = cv2.warpAffine(
        frame_bgr, affine, (512, 512),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132),
    )
    reference = (
        torch.from_numpy(np.ascontiguousarray(reference_u8[..., ::-1]) / 255.0)
        .to(torch.float32)
        .permute(2, 0, 1)
        .sub_(0.5)
        .div_(0.5)
    )

    # the crop is not the zero tensor of a no-face frame
    assert cropped.abs().max() > 0.5, "the crop is empty"

    difference = (cropped - reference).abs().max().item()
    # a 1/255 difference of the uint8 crop is 1/32512.5 in the normalized scale
    assert difference <= 2.0 / 255.0 / 127.5, (
        f"the batched crop drifted {difference:.3e} from the vendored warp"
    )

    # the paste loop reused the matrix of the crop
    assert len(face_helper.affine_matrices) == 1, face_helper.affine_matrices
    # The reference ``affine`` is fitted on the **raw** landmarks; the
    # processor fits it on the **smoothed** ones (the single temporal filter
    # of the sequence path).  On a 1-frame clip the symmetric pad makes the
    # smoothing a weighted mean of the same five points, so the result is
    # the input to within one float32 rounding - ~1e-4 px at the landmark
    # magnitude (Opt 3).  The LMEDS fit then translates into a ~2e-4-matrix
    # shift: the comparison is therefore absolute-tolerant in the sub-pixel
    # band rather than the default ``allclose`` relative tolerance, which a
    # near-zero matrix entry would otherwise bound below.
    affine_delta = np.abs(face_helper.affine_matrices[0] - affine).max()
    assert affine_delta <= 1e-3, (
        f"the paste loop did not reuse the affine matrix of the crop "
        f"(delta {affine_delta:.3e})"
    )


class _StubParseNet:
    """A parse net with the vendored contract (``[0]`` are the logits).

    A 1x1 convolution over the color channels - deterministic, and spatial
    enough for the batched paste to produce a non-trivial soft mask.
    """

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(5)
        self.weight = torch.randn(19, 3, 1, 1, generator=generator) * 0.35

    def __call__(self, samples):
        return (F.conv2d(samples, self.weight), None)


@pytest.mark.unit
def test_the_sequence_crop_warps_on_the_device_without_cv2(
    processor_module, monkeypatch
):
    """Opt 1: the fused crop warps on the device - ``cv2.warpAffine`` never runs.

    ``cv2.warpAffine`` is a shared module attribute, so the raising patch
    hits the vendored paste too - the corpus must route **every** frame
    through the batched paste (whose face warp is the torch
    ``warp_face_roi``): every frame carries exactly one detected face with
    a valid affine, all frames color, a parse net present (``use_parse``
    stub), ``draw_box=False`` and no background / face upsampler.  Outside
    this corpus the raising patch fails the run loudly by design - an
    unplanned failure there means the corpus leaked a vendored-fallback
    frame, not that Opt 1 regressed.

    The recorded crop must be the batched device op's output: it is
    recomputed here through ``crop_faces_batch`` +
    ``bgr_u8_batch_to_normalized`` on the same detection frames and
    compared bit for bit (the same op on the same input - tighter than the
    2/255/127.5 warp bound of the pin above).
    """
    height, width = 512, 640
    rng = np.random.default_rng(11)
    small = rng.uniform(0.0, 255.0, size=(16, 20, 3)).astype(np.float32)
    frame = np.clip(
        cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC), 0, 255
    ).astype(np.uint8)
    landmarks = FACE_TEMPLATE_512 + np.array([100.0, 50.0])

    face_helper = _LandmarkFaceHelper(landmarks)
    face_helper.use_parse = True
    face_helper.face_parse = _StubParseNet()
    keep_net = _RecordingKeepNet()
    processor = processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))

    frames = torch.from_numpy(
        np.stack([frame, frame]).astype(np.float32) / 255.0)

    def _no_cv2_warp(*args, **kwargs):
        raise AssertionError("the crop must not warp through cv2.warpAffine")

    monkeypatch.setattr(cv2, "warpAffine", _no_cv2_warp)
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    # every frame went through the batched paste: no vendored fallback (its
    # paste warps with cv2.warpAffine and would have raised above), and the
    # canvas is the raw frame size (the 512 floor - no read_image upscale)
    assert "paste_faces_to_input_image" not in face_helper.calls, face_helper.calls
    assert tuple(output.shape) == (2, height, width, 3), output.shape

    # the recorded crop is the device op's output, bit for bit
    image_utils = importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")
    gpu_ops = importlib.import_module("ComfyUI_KEEP.src.keep_gpu_ops")
    frame_bgr = image_utils.comfy_batch_to_bgr_u8(frames)
    affine = cv2.estimateAffinePartial2D(
        landmarks, FACE_TEMPLATE_512, method=cv2.LMEDS
    )[0]
    expected_u8 = gpu_ops.crop_faces_batch(
        torch.from_numpy(frame_bgr).permute(0, 3, 1, 2),
        [affine, affine], (512, 512))
    expected = image_utils.bgr_u8_batch_to_normalized(expected_u8)
    assert len(keep_net.frames) == 1, keep_net.frames
    assert tuple(keep_net.frames[0].shape) == (1, 2, 3, 512, 512)
    assert torch.equal(keep_net.frames[0][0], expected)


# ---------------------------------------------------------------------------
# P-M1 - the golden fixtures of the image paths (the pre-restructure outputs)
# ---------------------------------------------------------------------------
def _load_golden_capture():
    """The capture tool module - the single source of the golden cases."""
    import importlib.util

    source = (
        Path(__file__).resolve().parent.parent / "tools" / "capture_round4_golden.py"
    )
    spec = importlib.util.spec_from_file_location("_round4_golden_capture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.parametrize("case_name", ["image_multiface", "image_single"])
def test_the_image_golden_fixtures_are_reproduced_bit_for_bit(
    processor_module, monkeypatch, case_name
):
    """P-M1: the image paths re-run their captured cases bitwise.

    The fixtures come from ``tools/capture_round4_golden.py`` (captured
    once from the pre-restructure code, D-R2.0): the input image, the
    returned BGR image, the KEEP call shapes and the parse-call count (the
    vendored paste parses every pasted face itself,
    ``face_restoration_helper.py:462``).  The sub-512 input exercises the
    vendored upscale in all three ``read_image`` stages of the image paths.
    """
    helper_module = importlib.import_module(
        "ComfyUI_KEEP.modules.deps.facelib.utils.face_restoration_helper")
    monkeypatch.setattr(helper_module, "init_detection_model", lambda *a, **k: object())
    monkeypatch.setattr(helper_module, "init_parsing_model", lambda *a, **k: object())
    capture = _load_golden_capture()
    stubs = capture._load_test_stubs()
    case = capture.IMAGE_CASES[case_name]
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    golden = np.load(fixture_dir / f"round4_golden_{case_name}.npz")

    image_bgr = capture.golden_frames(1, case["frame_seed"])[0]
    assert np.array_equal(image_bgr, golden["input_image"]), (
        "the re-run input drifted from the captured one"
    )
    faces = {key: capture.golden_face(stubs, key) for key in capture.FACE_OFFSETS}
    face_helper = capture._make_helper(
        stubs, helper_module,
        [[faces[key] for key in case["faces"]]], capture.PARSE_SEEDS[case_name])
    keep_net = stubs._KeepNet()
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(keep_net, face_helper))
    output = processor.process_image(
        cv2_image_orig=image_bgr,
        final_upscale_factor=1.0,
        has_aligned=False,
        only_center_face=case["only_center_face"],
        draw_box=False,
    )

    assert np.array_equal(np.asarray(output), golden["output_image"]), (
        "the returned image is not bitwise the captured one"
    )
    shapes, flags = capture._keep_records(keep_net)
    assert np.array_equal(shapes, golden["keep_net_shapes"]), (
        f"the KEEP call shapes moved: {shapes} vs {golden['keep_net_shapes']}"
    )
    assert np.array_equal(flags, golden["keep_net_need_upscale"])
    assert face_helper.face_parse.calls == int(golden["parse_calls"]), (
        f"the parse-call count moved: {face_helper.face_parse.calls} "
        f"vs {int(golden['parse_calls'])}"
    )


# ---------------------------------------------------------------------------
# Round 6, Opt 5 - the aligned single-image fast path (design section 5)
# ---------------------------------------------------------------------------
class _IdentityKeepNet(_StubKeepNet):
    """A KEEP net that returns its input and keeps a copy of the call.

    The identity keeps the input contract checkable: the net input stays in
    ``[-1, 1]`` only if the ``/255`` ran *before* the normalize, and the
    output is the denormalized input - so the recorded frames pin both the
    chain and the return contract without a second stub.
    """

    def __init__(self) -> None:
        super().__init__()
        self.frames: list[torch.Tensor] = []

    def __call__(self, frames, need_upscale=False):
        self.calls.append((tuple(frames.shape), need_upscale))
        self.frames.append(frames.detach().clone())
        return frames


class _BlackKeepNet(_StubKeepNet):
    """Every output comes back at the bottom of the range (all black)."""

    def __call__(self, frames, need_upscale=False):
        self.calls.append((tuple(frames.shape), need_upscale))
        return torch.full_like(frames, -1.0)


def _aligned_face_image(seed: int = 3) -> np.ndarray:
    """A "already aligned 512" face" as the node hands it over: uint8 BGR.

    The node's ``comfy_image_to_cv2`` (``keep_image_utils.py:34``) has
    already truncated the IMAGE tensor to uint8, so the fast path takes an
    arbitrary-size uint8 BGR image - the resize is the first thing it does.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(640, 768, 3), dtype=np.uint8)


def _expected_aligned_net_input(image: np.ndarray) -> torch.Tensor:
    """The test's own computation of the corrected chain (design 5.4a).

    bilinear-512 on the float tensor, **``/255`` first**, then the BGR ->
    RGB flip and the ``(t - 0.5) / 0.5`` normalize - the same order the
    standard aligned leg applies (``img2tensor(cropped / 255., bgr2rgb)``
    then the normalize).  A re-computation alone would also pass a wrong
    chain, so the companion range guard (min >= -1, max <= 1) is the
    independent check: a missing ``/255`` lands in ``~(-1, 509)``.
    """
    img = torch.from_numpy(np.ascontiguousarray(image)).to(torch.device("cpu"))
    img512 = F.interpolate(
        img.permute(2, 0, 1).unsqueeze(0).float(),
        size=(512, 512), mode="bilinear", align_corners=False)
    return img512[:, [2, 1, 0]].div(255.0).sub(0.5).div(0.5)


@pytest.mark.unit
def test_the_aligned_fast_path_feeds_the_net_the_corrected_chain(processor_module):
    """Opt 5a: one KEEP call, ``(1, 2, 3, 512, 512)``, the pinned input.

    The recorded net input equals the test's own recomputation of the
    corrected chain **bit for bit**, and it stays inside ``[-1, 1]`` - the
    range guard is what makes the pin independent of the implementation.
    """
    keep_net = _IdentityKeepNet()
    face_helper = _StubFaceHelper()
    processor = processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))
    image = _aligned_face_image()

    output = processor.process_image(
        cv2_image_orig=image,
        final_upscale_factor=1.0,
        has_aligned=True,
        only_center_face=True,
        draw_box=False,
    )

    assert keep_net.calls == [((1, 2, 3, 512, 512), False)], keep_net.calls
    assert tuple(keep_net.frames[0].shape) == (1, 2, 3, 512, 512)
    recorded = keep_net.frames[0]
    # (batch 0, frame 0) of the ``(1, 2, 3, 512, 512)`` KEEP batch - the
    # batch dim is 1 here, so the frame lives on the **time** axis (unlike
    # the sequence stack, where ``[0]`` selects the first face).
    assert torch.equal(recorded[0, 0], _expected_aligned_net_input(image)[0]), (
        "the net input is not the corrected chain (bilinear-512, /255 first)"
    )
    assert recorded.min() >= -1.0, recorded.min()
    assert recorded.max() <= 1.0, recorded.max()
    # the two halves are the same frame (the duplicate-to-2 contract)
    assert torch.equal(recorded[0, 0], recorded[0, 1])
    assert tuple(output.shape) == (512, 512, 3), output.shape


@pytest.mark.unit
def test_the_aligned_fast_path_returns_the_quantized_bgr_image(processor_module):
    """Opt 5b: the return is the 3-D BGR uint8 image, denormalized/quantized.

    With the identity net the output is the recorded net input moved through
    ``clamp(-1, 1) -> +1 -> /2 -> *255 -> round`` and flipped back to BGR -
    the quantization the vendored ``tensor2img`` applied stays.
    """
    keep_net = _IdentityKeepNet()
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(keep_net, _StubFaceHelper()))
    image = _aligned_face_image()

    output = processor.process_image(
        cv2_image_orig=image,
        final_upscale_factor=1.0,
        has_aligned=True,
        only_center_face=True,
        draw_box=False,
    )

    expected_face = _expected_aligned_net_input(image).clamp(-1, 1).add(1).div_(2)
    expected = (expected_face[0].mul(255.0).round().clamp(0, 255).to(torch.uint8)
                .permute(1, 2, 0)[..., [2, 1, 0]].cpu().numpy())
    assert isinstance(output, np.ndarray), type(output)
    assert output.dtype == np.uint8, output.dtype
    assert np.array_equal(output, expected), (
        f"max delta {(output.astype(int) - expected.astype(int)).__abs__().max()}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("factor, expected_hw", [(2.0, (1024, 1024)), (0.5, (256, 256))])
def test_the_aligned_fast_path_upscales_on_the_device(processor_module, factor, expected_hw):
    """Opt 5c: the final upscale is a device resize of the float face.

    Factor 2 goes **up** and factor 0.5 still resizes (the size check mirrors
    ``keep_processor.py:566``), so a factor of 1.0 is the only identity case.
    """
    keep_net = _IdentityKeepNet()
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(keep_net, _StubFaceHelper()))
    image = _aligned_face_image()

    output = processor.process_image(
        cv2_image_orig=image,
        final_upscale_factor=factor,
        has_aligned=True,
        only_center_face=True,
        draw_box=False,
    )

    assert output.shape == (*expected_hw, 3), output.shape
    if factor != 1.0:
        face = _expected_aligned_net_input(image).clamp(-1, 1).add(1).div(2)
        resized = F.interpolate(
            face, size=expected_hw, mode="bicubic", align_corners=False)
        expected = (resized[0].mul(255.0).round().clamp(0, 255).to(torch.uint8)
                    .permute(1, 2, 0)[..., [2, 1, 0]].cpu().numpy())
        assert np.array_equal(output, expected), "the upscale is not the device bicubic"


@pytest.mark.unit
def test_an_all_black_restore_returns_the_original_image(processor_module):
    """Opt 5d: the black-face guard survives the fast path.

    The standard leg checks the restored face after ``tensor2img``
    (``keep_processor.py:540``); the fast path checks the tensor before the
    quantization - the observable contract is the same: the original image.
    """
    keep_net = _BlackKeepNet()
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(keep_net, _StubFaceHelper()))
    image = _aligned_face_image()

    output = processor.process_image(
        cv2_image_orig=image,
        final_upscale_factor=1.0,
        has_aligned=True,
        only_center_face=True,
        draw_box=False,
    )

    assert output is image, "an all-black restore must return the caller's image"
    assert keep_net.calls == [((1, 2, 3, 512, 512), False)], keep_net.calls


@pytest.mark.unit
def test_the_aligned_fast_path_hands_over_to_the_upscaler_configured_pipeline(processor_module):
    """Opt 5e: a configured face upsampler keeps the standard pipeline.

    The precondition (``face_upscale_model is None``) is the pin: with a
    non-``None`` stub the standard aligned leg runs instead.  Its side effect
    proves it - the helper is driven (the ``512`` resize, the crop list, the
    restored face), which the fast path never does, and the output is the
    standard leg's result rather than the fast-path return.  The fast path
    also never reaches the net on this corpus, so the KEEP-call count is the
    second half of the proof.
    """
    keep_net = _IdentityKeepNet()
    face_helper = _StubFaceHelper()

    class _StubUpscaleModel:
        scale = 2.0

        def __init__(self):
            self.model = self._model

        def _model(self, samples):
            return samples

    pack = types.SimpleNamespace(
        keep_net=keep_net,
        face_helper=face_helper,
        bg_upscale_model=None,
        face_upscale_model=_StubUpscaleModel(),
        device=torch.device("cpu"),
        model_type_str="KEEP",
    )
    processor = processor_module.KEEPFaceProcessor(pack)
    image = _aligned_face_image()

    output = processor.process_image(
        cv2_image_orig=image,
        final_upscale_factor=1.0,
        has_aligned=True,
        only_center_face=True,
        draw_box=False,
    )

    # the standard pipeline drove the helper - the fast path leaves it alone
    assert face_helper.calls, face_helper.calls
    # it reached the net with the vendored crop chain, not the fast-path one
    assert keep_net.calls == [((1, 2, 3, 512, 512), False)], keep_net.calls
    # the standard leg's output is the upscaled 512 face, not the fast path's
    # raw resize of the original - the crop of the aligned leg is
    # ``resize_bgr_u8(image, (512, 512))``, so a fast-path run would differ
    assert tuple(output.shape) == (512, 512, 3), output.shape


# ---------------------------------------------------------------------------
# P-M3 / P-M4 / P-M5 - the memory bounds of the redesign (design section 3)
# ---------------------------------------------------------------------------
class _TinyParseNet:
    """A parse net with the vendored contract (``[0]`` are the logits)."""

    def __init__(self, seed: int = 7) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(19, 3, 1, 1, generator=generator) * 0.35

    def __call__(self, samples):
        return (F.conv2d(samples, self.weight), None)


@pytest.mark.unit
def test_the_detection_pass_converts_at_most_a_group_at_a_time(
    processor_module, monkeypatch
):
    """P-M3 (single-face leg): no whole-clip convert before the first KEEP call.

    The streamed detection pass converts a group of at most
    ``max_clip_length`` rows at a time; the whole-clip uint8 stack of the
    old detection pass is gone.  Recording the converter's batch sizes:
    everything before the first KEEP forward stays at or below the clip
    length (3 frames, clips of 2 - a whole-clip convert would be a 3).
    """
    conversions: list[int] = []
    real_convert = processor_module.comfy_batch_to_bgr_u8

    def recording_convert(tensor):
        conversions.append(int(tensor.shape[0]))
        return real_convert(tensor)

    monkeypatch.setattr(processor_module, "comfy_batch_to_bgr_u8", recording_convert)

    keep_at: list[int] = []

    class _ProbeKeepNet(_StubKeepNet):
        def __call__(self, frames, need_upscale=False):
            keep_at.append(len(conversions))
            return super().__call__(frames, need_upscale=need_upscale)

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_ProbeKeepNet(), _DetectingFaceHelper())
    )
    frames = torch.rand((3, 512, 640, 3), dtype=torch.float32)
    processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=2,
    )

    assert keep_at, "the probe net must be called"
    before_the_first_keep = conversions[: keep_at[0]]
    assert before_the_first_keep, (
        "the detection pass must convert through the module function"
    )
    assert max(before_the_first_keep) <= 2, before_the_first_keep
    # The fused crop loop re-derives the chunk's detection images in ONE
    # group conversion per chunk (the grouped D-R2.2.1) - the per-row
    # re-derivations of the first round-4 cut are gone: across the whole
    # run only the detection groups, the crop groups and the paste stage's
    # convert-ins may call the converter.
    #
    # Round 17 (the final stage's RAM) moved the paste stage's convert-in:
    # it was ONE whole-clip conversion (3 rows - the uint8 stack of the
    # clip, ~3.78 GB on the owner's 777-frame clip), and it is now the
    # **fallback groups** (the lazy source, one group of
    # ``max_clip_length`` rows per group).  The count therefore moves UP
    # (3 -> 4 here: the detection groups [2, 1] plus the two fallback
    # groups [2, 1]) while the LARGEST conversion moves DOWN (3 -> 2).
    # The budget follows the movement: the paste stage's convert-ins are
    # the fallback groups now (ceil(3 / 2) = 2 of them), so the ceiling
    # is 2 * ceil(3 / 2) + 2 = 6 (the detection chunks, the crop chunks
    # and the fallback groups).
    assert len(conversions) <= 6, conversions


@pytest.mark.unit
def test_the_paste_stage_never_converts_the_whole_clip(
    processor_module, monkeypatch
):
    """Round 17: the paste stage converts group by group, never the clip.

    The paste stage used to materialize the clip's uint8 BGR stack in one
    conversion (``input_frames_bgr = comfy_batch_to_bgr_u8(
    image_sequence_tensor)``) - ~3.78 GB of host RAM on the owner's
    777-frame 1500x1080 clip.  Round 17 replaced it with the lazy
    :class:`~.keep_image_utils.BgrFrameSource`: the paste core asks the
    source for the rows of the group it is about to paste, and the
    fallback rows arrive in groups of their own.

    The pin drives the **chunked** mode - the production mode of a long
    clip (the full mode needs the whole device canvas and the host mode
    is the no-CUDA fallback; both materialize the eligible rows, the
    disclosed limitation of the design) - with a group size of 2 over a
    4-frame clip, and records the batch size of every conversion of the
    run:

    * no conversion covers the whole clip (4 rows): the stack is never
      built;
    * the paste stage's conversions are group-sized (at most
      ``paste_chunk_size`` rows): it converts the rows it pastes, not the
      clip.

    The measured run (the pin's fixture): the whole run converts
    ``[2, 2, 2, 2, 2, 2, 0]`` - the detection chunks, the crop chunks,
    the paste core's two eligible groups and the **empty** fallback group
    (every row was batched, so the fallback generator runs its defensive
    empty pass).
    """
    paste_mod = importlib.import_module("ComfyUI_KEEP.src.keep_gpu_paste")
    monkeypatch.setattr(
        paste_mod, "device_output_mode", lambda *args, **kwargs: "chunked")
    monkeypatch.setattr(paste_mod, "paste_chunk_size", lambda *args, **kwargs: 2)

    conversions: list[int] = []
    real_convert = processor_module.comfy_batch_to_bgr_u8

    def recording_convert(tensor):
        conversions.append(int(tensor.shape[0]))
        return real_convert(tensor)

    monkeypatch.setattr(processor_module, "comfy_batch_to_bgr_u8", recording_convert)

    # The paste stage's conversions are the ones the core makes (the core
    # is the first consumer of the lazy source) plus the fallback groups
    # the vendored paste pulls afterwards.
    paste_at: list[int] = []
    real_chunked = paste_mod.paste_faces_batch_chunked

    def recording_chunked(*args, **kwargs):
        paste_at.append(len(conversions))
        return real_chunked(*args, **kwargs)

    monkeypatch.setattr(paste_mod, "paste_faces_batch_chunked", recording_chunked)

    landmarks = (FACE_TEMPLATE_512 - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _LandmarkFaceHelper(landmarks)
    face_helper.face_parse = _TinyParseNet()
    face_helper.use_parse = True
    frames = torch.rand((4, 64, 64, 3), dtype=torch.float32)

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), face_helper)
    )
    processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=2,
    )

    assert conversions, "the run must convert through the module function"
    assert paste_at, "the chunked paste core must run"
    assert 4 not in conversions, (
        f"no conversion may cover the whole clip, got {conversions}"
    )
    paste_stage = conversions[paste_at[0]:]
    assert paste_stage, "the paste stage must convert its groups"
    assert max(paste_stage) <= 2, (
        f"the paste stage must convert group by group, got {paste_stage}"
    )


@pytest.mark.unit
def test_the_chunked_mode_keeps_one_tail_buffer(processor_module, monkeypatch):
    """P-M4: the forced chunked mode ships one float32 buffer (fix 1.2 + fix 1.1, 2026-10-30).

    ``device_output_mode`` is patched to "chunked" (on CPU the modes are
    value-identical, the verdict just never selects a device mode by
    itself): 3 frames at 64x64, all batch-eligible.  Fix 1.2 removed the
    gray skip, so the black frame of the old corpus
    (``frames[1] = 0.0``, "the black frame is a gray frame - it leaves
    the batch") is eligible now: all three rows ride the chunked core and
    the paste stage's reuse fast path (fix 1.1, the buffer alias)
    ``final_batch = chunks_out`` aliases the whole-clip IMAGE to the
    chunked core's ``out`` - one float32 allocation, no indexed scatter.
    Recording every ``torch.empty``: exactly one 4-D float32 allocation
    with the whole-clip row count exists, and the returned IMAGE **is**
    that buffer (still alive after the return; the old
    eligible-rows staging tail and its release pin are gone with the
    scatter).
    """
    paste_mod = importlib.import_module("ComfyUI_KEEP.src.keep_gpu_paste")
    monkeypatch.setattr(paste_mod, "device_output_mode", lambda *args, **kwargs: "chunked")

    empties: list[tuple[tuple, torch.dtype, weakref.ref]] = []
    real_empty = torch.empty

    def recording_empty(*args, **kwargs):
        tensor = real_empty(*args, **kwargs)
        empties.append((tuple(tensor.shape), tensor.dtype, weakref.ref(tensor)))
        return tensor

    monkeypatch.setattr(torch, "empty", recording_empty)

    landmarks = (FACE_TEMPLATE_512 - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _LandmarkFaceHelper(landmarks)
    face_helper.face_parse = _TinyParseNet()
    face_helper.use_parse = True
    frames = torch.rand((3, 64, 64, 3), dtype=torch.float32)

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), face_helper)
    )
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    float_rows = [
        (shape, ref) for shape, dtype, ref in empties
        if dtype == torch.float32 and len(shape) == 4
    ]
    final = [ref for shape, ref in float_rows if shape[0] == 3]
    assert len(final) == 1, float_rows
    # the returned IMAGE is the chunked core's buffer itself (the alias)
    assert final[0]() is output, "final_batch must be the chunked core's out"
    assert tuple(output.shape) == (3, 512, 512, 3), output.shape


@pytest.mark.unit
def test_the_mixed_case_holds_one_whole_clip_buffer(processor_module, monkeypatch):
    """Round 18: the mixed case holds **one** whole-clip float32 buffer.

    The paste stage's output used to be assembled in two steps: the paste
    core allocated its own buffer (the eligible rows) and the tail
    allocated a **second** whole-clip buffer and copied the eligible rows
    into it - two whole-clip float32 tensors alive at the same time in
    the mixed case (one frame without a face is enough).  Round 18 hands
    the tail's buffer to the core, so the eligible rows land in their
    final rows and exactly one whole-clip float32 buffer exists.

    The pin drives the **chunked** mode (the production mode of a long
    clip) over a 3-frame clip whose middle frame has no affine (the
    LMEDS failure - a real production mode) - the mixed case, with the
    eligible rows **scattered** (``[0, 2]``).  Every ``torch.empty`` of
    the run is recorded: exactly one 4-D float32 allocation carries the
    canvas's trailing dims (the core's own eligible-sized buffer would
    carry them too - the census sees both), and that tensor **is** the
    returned IMAGE (the weakref identity - the alias, not a copy).
    """
    paste_mod = importlib.import_module("ComfyUI_KEEP.src.keep_gpu_paste")
    monkeypatch.setattr(paste_mod, "device_output_mode", lambda *args, **kwargs: "chunked")

    # the LMEDS failure of the middle frame: its affine stays ``None``, so
    # the frame is not batch-eligible and rides the vendored fallback.
    real_estimate_affines = processor_module.estimate_affines

    def failing_estimate_affines(landmarks, template):
        matrices = list(real_estimate_affines(landmarks, template))
        if len(matrices) > 1:
            matrices[1] = None
        return matrices

    monkeypatch.setattr(processor_module, "estimate_affines", failing_estimate_affines)

    empties: list[tuple[tuple, torch.dtype, weakref.ref]] = []
    real_empty = torch.empty

    def recording_empty(*args, **kwargs):
        tensor = real_empty(*args, **kwargs)
        empties.append((tuple(tensor.shape), tensor.dtype, weakref.ref(tensor)))
        return tensor

    monkeypatch.setattr(torch, "empty", recording_empty)

    landmarks = (FACE_TEMPLATE_512 - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _LandmarkFaceHelper(landmarks)
    face_helper.face_parse = _TinyParseNet()
    face_helper.use_parse = True
    frames = torch.rand((3, 64, 64, 3), dtype=torch.float32)

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), face_helper)
    )
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    canvas_shape = tuple(int(value) for value in output.shape[1:])
    float_buffers = [
        (shape, ref) for shape, dtype, ref in empties
        if dtype == torch.float32 and len(shape) == 4
        and shape[1:] == canvas_shape
    ]
    assert len(float_buffers) == 1, (
        f"the mixed case must hold one canvas-shaped float32 buffer, got "
        f"{[shape for shape, _ in float_buffers]}"
    )
    assert float_buffers[0][0][0] == 3, (
        f"the buffer must carry the whole clip's rows, got {float_buffers[0][0]}"
    )
    # the returned IMAGE is the paste buffer itself (the alias, no copy)
    assert float_buffers[0][1]() is output, (
        "the returned IMAGE must be the paste buffer itself"
    )


@pytest.mark.unit
def test_the_mixed_case_draws_the_boxes_on_the_final_rows(processor_module, monkeypatch):
    """Round 18: the boxes land on the **final** rows of the mixed case.

    ``draw_box`` moved from the eligible-space rows to the final buffer
    (the float32 IMAGE rows - the uint8 colour casts on assignment, the
    documented 0.0 / 255.0 / 0.0).  The translation is the point of the
    pin: the eligible rows of a mixed clip are **scattered** (here
    ``[0, 2]`` - the middle frame has no affine), so a box addressed to
    the eligible-space row would land on the wrong final row.

    The pin compares two runs of the same clip - ``draw_box=True`` and
    ``draw_box=False``: the eligible rows differ (the box is on them) and
    the fallback row is **identical** (a frame without a face never gets
    a box - in either design).
    """
    paste_mod = importlib.import_module("ComfyUI_KEEP.src.keep_gpu_paste")
    monkeypatch.setattr(paste_mod, "device_output_mode", lambda *args, **kwargs: "chunked")

    real_estimate_affines = processor_module.estimate_affines

    def failing_estimate_affines(landmarks, template):
        matrices = list(real_estimate_affines(landmarks, template))
        if len(matrices) > 1:
            matrices[1] = None
        return matrices

    monkeypatch.setattr(processor_module, "estimate_affines", failing_estimate_affines)

    generator = torch.Generator().manual_seed(17)
    frames = torch.rand((3, 64, 64, 3), dtype=torch.float32, generator=generator)
    landmarks = (FACE_TEMPLATE_512 - 256.0) * 0.1 + np.array([48.0, 32.0])

    def run(draw_box):
        face_helper = _LandmarkFaceHelper(landmarks)
        face_helper.face_parse = _TinyParseNet()
        face_helper.use_parse = True
        processor = processor_module.KEEPFaceProcessor(
            _make_pack(_StubKeepNet(), face_helper)
        )
        return processor.process_image_sequence(
            image_sequence_tensor=frames,
            final_upscale_factor=1.0,
            has_aligned_frames=False,
            only_center_face=True,
            draw_box=draw_box,
            max_clip_length=20,
        )

    boxed = run(True)
    plain = run(False)

    assert tuple(boxed.shape) == (3, 512, 512, 3), boxed.shape
    for row in (0, 2):
        assert not torch.equal(boxed[row], plain[row]), (
            f"the eligible row {row} must carry the box"
        )
    assert torch.equal(boxed[1], plain[1]), (
        "the fallback row must not carry a box"
    )


@pytest.mark.unit
@pytest.mark.parametrize("case_name", ["image_single", "image_multiface"])
def test_the_image_paths_release_the_helper_after_the_run(
    processor_module, monkeypatch, case_name
):
    """P-M5 (image legs): no detection image survives the run (D-R2.4).

    The image paths run the real vendored helper (the capture rig): the
    ``finally`` must clear the face lists and the ``input_img`` the same
    way the sequence paths do - the sub-512 golden input leaves a full
    upscaled frame in ``input_img`` unless it is released.
    """
    helper_module = importlib.import_module(
        "ComfyUI_KEEP.modules.deps.facelib.utils.face_restoration_helper")
    monkeypatch.setattr(helper_module, "init_detection_model", lambda *a, **k: object())
    monkeypatch.setattr(helper_module, "init_parsing_model", lambda *a, **k: object())
    capture = _load_golden_capture()
    stubs = capture._load_test_stubs()
    case = capture.IMAGE_CASES[case_name]
    image_bgr = capture.golden_frames(1, case["frame_seed"])[0]
    faces = {key: capture.golden_face(stubs, key) for key in capture.FACE_OFFSETS}
    face_helper = capture._make_helper(
        stubs, helper_module,
        [[faces[key] for key in case["faces"]]], capture.PARSE_SEEDS[case_name])
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(stubs._KeepNet(), face_helper))
    processor.process_image(
        cv2_image_orig=image_bgr,
        final_upscale_factor=1.0,
        has_aligned=False,
        only_center_face=case["only_center_face"],
        draw_box=False,
    )

    assert face_helper.input_img is None, case_name
    for name in (
        "cropped_faces", "restored_faces", "affine_matrices",
        "inverse_affine_matrices", "all_landmarks_5",
    ):
        assert getattr(face_helper, name) == [], (case_name, name)


# ---------------------------------------------------------------------------
# Round 19 (FIX 1) - the mixed case builds no face gather
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_the_mixed_case_gathers_no_face_stack(processor_module, monkeypatch):
    """Round 19 (FIX 1): the mixed case builds no eligible-sized face gather.

    The mixed case used to copy the eligible faces once
    (``np.stack([restored_faces[i] for i in eligible])`` - ~611 MB on the
    owner's 776-eligible clip) although the paste core reads the faces
    per group.  The twin's mixed branch now builds the lazy selected-rows
    source (:class:`~.keep_gpu_paste.LazyFaceSource`) instead, and the
    core serves the group's rows from it - a view when the group is
    contiguous in stack space, a group-sized copy otherwise.

    The pin drives the **chunked** mode (the production mode of a long
    clip) over the round-18 mixed rig - the middle frame has no affine
    (the LMEDS failure), so the eligible rows are scattered (``[0, 2]``)
    - and spies the hand-off plus every ``np.stack`` of the run: the core
    receives the lazy source (never a gathered copy) and no ``np.stack``
    of the run carries the eligible-sized faces shape.
    """
    paste_mod = importlib.import_module("ComfyUI_KEEP.src.keep_gpu_paste")
    monkeypatch.setattr(
        paste_mod, "device_output_mode", lambda *args, **kwargs: "chunked")

    handed: list = []
    real_chunked = paste_mod.paste_faces_batch_chunked

    def recording_chunked(*args, **kwargs):
        handed.append(kwargs.get("faces_bgr"))
        return real_chunked(*args, **kwargs)

    monkeypatch.setattr(paste_mod, "paste_faces_batch_chunked", recording_chunked)

    stacks: list[tuple] = []
    real_stack = np.stack

    def recording_stack(arrays, *args, **kwargs):
        result = real_stack(arrays, *args, **kwargs)
        stacks.append(tuple(int(value) for value in result.shape))
        return result

    monkeypatch.setattr(np, "stack", recording_stack)

    # the LMEDS failure of the middle frame - the mixed case
    real_estimate_affines = processor_module.estimate_affines

    def failing_estimate_affines(landmarks, template):
        matrices = list(real_estimate_affines(landmarks, template))
        if len(matrices) > 1:
            matrices[1] = None
        return matrices

    monkeypatch.setattr(processor_module, "estimate_affines", failing_estimate_affines)

    landmarks = (FACE_TEMPLATE_512 - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _LandmarkFaceHelper(landmarks)
    face_helper.face_parse = _TinyParseNet()
    face_helper.use_parse = True
    frames = torch.rand((3, 64, 64, 3), dtype=torch.float32)

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), face_helper)
    )
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    assert tuple(output.shape) == (3, 512, 512, 3), output.shape
    assert len(handed) == 1 and isinstance(handed[0], paste_mod.LazyFaceSource), (
        "the mixed case must hand the core the lazy face source, got "
        f"{type(handed[0]) if handed else None}"
    )
    canvas_shape = tuple(int(value) for value in output.shape[1:])
    gathers = [shape for shape in stacks if shape == (2, *canvas_shape)]
    assert gathers == [], (
        f"the eligible-sized faces gather is gone from the run, got {gathers}"
    )


@pytest.mark.unit
def test_the_all_eligible_gate_keeps_the_raw_stack_and_serves_slice_views(
    processor_module, monkeypatch
):
    """Round 19 (FIX 1): the all-eligible gate keeps the raw stack, views ride.

    The pass-through gate (round 4, D-R2.3.2) keeps the mixed-case change
    object-wise out of the all-eligible case: the raw restored stack is
    handed to the core **as the ndarray it is** - the lazy source only
    enters through the mixed branch.  And the core's numpy path serves
    the group rows as the slice views it served before
    (``np.shares_memory`` per group row) - no copy is reintroduced on the
    all-eligible path.

    The rig is the round-18 one with the affine failure removed - all
    three frames batch-eligible - and the spies record the hand-off,
    every faces group the numpy provider serves and every ``np.stack``
    of the run (there was no eligible-sized gather before the round
    either; the census pins the non-regression).
    """
    paste_mod = importlib.import_module("ComfyUI_KEEP.src.keep_gpu_paste")
    monkeypatch.setattr(
        paste_mod, "device_output_mode", lambda *args, **kwargs: "chunked")

    handed: list = []
    real_chunked = paste_mod.paste_faces_batch_chunked

    def recording_chunked(*args, **kwargs):
        handed.append(kwargs.get("faces_bgr"))
        return real_chunked(*args, **kwargs)

    monkeypatch.setattr(paste_mod, "paste_faces_batch_chunked", recording_chunked)

    shares: list[bool] = []
    real_group = paste_mod.NumpyFrameProvider.group

    def recording_group(provider, indices):
        rows = real_group(provider, indices)
        if handed and provider.materialize() is handed[-1]:
            shares.append(bool(np.shares_memory(rows, handed[-1])))
        return rows

    monkeypatch.setattr(paste_mod.NumpyFrameProvider, "group", recording_group)

    stacks: list[tuple] = []
    real_stack = np.stack

    def recording_stack(arrays, *args, **kwargs):
        result = real_stack(arrays, *args, **kwargs)
        stacks.append(tuple(int(value) for value in result.shape))
        return result

    monkeypatch.setattr(np, "stack", recording_stack)

    landmarks = (FACE_TEMPLATE_512 - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _LandmarkFaceHelper(landmarks)
    face_helper.face_parse = _TinyParseNet()
    face_helper.use_parse = True
    frames = torch.rand((3, 64, 64, 3), dtype=torch.float32)

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), face_helper)
    )
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )

    assert tuple(output.shape) == (3, 512, 512, 3), output.shape
    assert len(handed) == 1 and isinstance(handed[0], np.ndarray), (
        "the all-eligible gate must hand the raw restored stack, got "
        f"{type(handed[0]) if handed else None}"
    )
    assert shares, "the numpy provider must serve the faces groups"
    assert all(shares), f"every faces group must be a slice view: {shares}"
    canvas_shape = tuple(int(value) for value in output.shape[1:])
    gathers = [shape for shape in stacks if shape == (3, *canvas_shape)]
    assert gathers == [], (
        f"no eligible-sized gather may exist on the all-eligible path, got {gathers}"
    )


# ---------------------------------------------------------------------------
# Round 19 (FIX 2) - the fallback tail converts per gray group
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_the_fallback_rows_convert_per_gray_group_across_the_boundary(
    processor_module, monkeypatch
):
    """Round 19 (FIX 2), pin 4: the fallback rows convert per gray group.

    The chunks_mode tail used to stack **all** fallback frames of the clip
    and convert the whole stack in one call - an O(fallback rows) peak (the
    stack plus the ~10x float32 conversion chain).  The tail now flushes
    every gray group the loop advances past (the pull site) plus the last
    group (at the tail): the group's frames stack (at most
    ``max_clip_length`` rows), convert through the same elementwise chain
    and land in ``final_batch`` with the **indexed write**
    (``final_batch[rows] = value`` - the round-18 ``index_put_`` pattern;
    the read-side would be an advanced-indexing copy).

    The pin crosses a group boundary: a 6-frame clip whose frames 1-4 have
    no affine (the round-18 mixed rig's LMEDS failure, four of them) falls
    back on rows [1, 2, 3, 4]; with ``max_clip_length=2`` the gray groups
    are [1, 2] and [3, 4] - two flushes.  The spy records the frames of
    every canvas-shaped ``np.stack`` of the run; the reference is the
    **one-shot whole-stack conversion** the old tail produced
    (``bgr_u8_batch_to_comfy(np.stack(all four frames))``), computed here
    from the recorded frames.  Every fallback row of ``final_batch`` must
    be byte-equal (``torch.equal``) to its row of the reference - the
    conversion is elementwise per row, so the pixels cannot move.
    """
    image_utils = importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")
    paste_mod = importlib.import_module("ComfyUI_KEEP.src.keep_gpu_paste")
    monkeypatch.setattr(
        paste_mod, "device_output_mode", lambda *args, **kwargs: "chunked")

    # the LMEDS failure of the frames 1-4 - the four fallback rows
    real_estimate_affines = processor_module.estimate_affines

    def failing_estimate_affines(landmarks, template):
        matrices = list(real_estimate_affines(landmarks, template))
        for index in range(1, 5):
            if index < len(matrices):
                matrices[index] = None
        return matrices

    monkeypatch.setattr(
        processor_module, "estimate_affines", failing_estimate_affines)

    # the frames of every canvas-shaped np.stack of the run - the flush's
    # per-group stacks (the old tail's one-shot record would carry all 4)
    groups: list[list[np.ndarray]] = []
    real_stack = np.stack

    def recording_stack(arrays, *args, **kwargs):
        result = real_stack(arrays, *args, **kwargs)
        if result.ndim == 4 and result.shape[1:] == (512, 512, 3):
            groups.append([np.array(frame) for frame in arrays])
        return result

    monkeypatch.setattr(np, "stack", recording_stack)

    landmarks = (FACE_TEMPLATE_512 - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _LandmarkFaceHelper(landmarks)
    face_helper.face_parse = _TinyParseNet()
    face_helper.use_parse = True
    frames = torch.rand((6, 64, 64, 3), dtype=torch.float32)

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), face_helper)
    )
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=2,
    )

    assert tuple(output.shape) == (6, 512, 512, 3), output.shape
    # the boundary was crossed: two group stacks of two frames each - the
    # whole-fallback stack would be one record of four
    assert [len(group) for group in groups] == [2, 2], (
        f"the fallback frames must flush per gray group, got "
        f"{[len(group) for group in groups]}")
    # the reference: the one-shot whole-stack conversion the old tail ran
    # (the snapshot first - the conversion below rides the same spy)
    recorded = [list(group) for group in groups]
    all_frames = [frame for group in recorded for frame in group]
    reference = image_utils.bgr_u8_batch_to_comfy(np.stack(all_frames))
    assert tuple(reference.shape) == (4, 512, 512, 3), reference.shape
    for row, ref_row in zip((1, 2, 3, 4), range(4)):
        assert torch.equal(output[row], reference[ref_row]), (
            f"the fallback row {row} must be byte-equal to the one-shot "
            f"whole-stack conversion's row {ref_row}")


@pytest.mark.unit
def test_the_chunks_mode_tail_builds_no_whole_fallback_stack(
    processor_module, monkeypatch
):
    """Round 19 (FIX 2), pin 5: no whole-fallback stack in the chunks_mode tail.

    The census pin: the same boundary rig as pin 4 (6 frames, the fallback
    rows [1, 2, 3, 4], ``max_clip_length=2`` - two gray groups), with every
    ``np.stack`` and every ``torch.empty`` of the run recorded.  No stack
    may carry more rows than one gray group's fallback rows (2 here - the
    old tail's whole-fallback record would be a 4), and the round-18
    contract stays: exactly **one** canvas-shaped whole-clip float32
    ``torch.empty`` exists and it **is** the returned IMAGE (the eligible
    rows' buffer - the tail writes the fallback rows into it, it does not
    allocate a second one).
    """
    paste_mod = importlib.import_module("ComfyUI_KEEP.src.keep_gpu_paste")
    monkeypatch.setattr(
        paste_mod, "device_output_mode", lambda *args, **kwargs: "chunked")

    real_estimate_affines = processor_module.estimate_affines

    def failing_estimate_affines(landmarks, template):
        matrices = list(real_estimate_affines(landmarks, template))
        for index in range(1, 5):
            if index < len(matrices):
                matrices[index] = None
        return matrices

    monkeypatch.setattr(
        processor_module, "estimate_affines", failing_estimate_affines)

    stacks: list[tuple] = []
    real_stack = np.stack

    def recording_stack(arrays, *args, **kwargs):
        result = real_stack(arrays, *args, **kwargs)
        stacks.append(tuple(int(value) for value in result.shape))
        return result

    monkeypatch.setattr(np, "stack", recording_stack)

    empties: list[tuple[tuple, torch.dtype, weakref.ref]] = []
    real_empty = torch.empty

    def recording_empty(*args, **kwargs):
        tensor = real_empty(*args, **kwargs)
        empties.append((tuple(tensor.shape), tensor.dtype, weakref.ref(tensor)))
        return tensor

    monkeypatch.setattr(torch, "empty", recording_empty)

    landmarks = (FACE_TEMPLATE_512 - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _LandmarkFaceHelper(landmarks)
    face_helper.face_parse = _TinyParseNet()
    face_helper.use_parse = True
    frames = torch.rand((6, 64, 64, 3), dtype=torch.float32)

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), face_helper)
    )
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=2,
    )

    assert tuple(output.shape) == (6, 512, 512, 3), output.shape
    canvas_shape = (512, 512, 3)
    tail_stacks = [
        shape for shape in stacks
        if len(shape) == 4 and shape[1:] == canvas_shape
    ]
    assert tail_stacks, "the flush must stack the fallback frames"
    assert max(shape[0] for shape in tail_stacks) <= 2, (
        f"no tail stack may exceed one gray group's fallback rows "
        f"(2 here), got {tail_stacks}")
    whole_fallback = [shape for shape in tail_stacks if shape[0] == 4]
    assert whole_fallback == [], (
        f"the whole-fallback stack is gone from the tail, got {whole_fallback}")
    # the round-18 conjunction: one whole-clip float32 buffer, and it is
    # the returned IMAGE
    float_buffers = [
        (shape, ref) for shape, dtype, ref in empties
        if dtype == torch.float32 and len(shape) == 4
        and shape[1:] == canvas_shape
    ]
    assert len(float_buffers) == 1, (
        f"the mixed case must hold one canvas-shaped float32 buffer, got "
        f"{[shape for shape, _ in float_buffers]}")
    assert float_buffers[0][0][0] == 6, (
        f"the buffer must carry the whole clip's rows, got {float_buffers[0][0]}")
    assert float_buffers[0][1]() is output, (
        "the returned IMAGE must be the paste buffer itself")
