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

    def read_image(self, image):
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
def test_the_sequence_releases_the_vram_at_every_chunk_boundary(
    processor_module, monkeypatch
):
    """P4.6: every chunk boundary releases the cached VRAM.

    30 frames with ``max_clip_length=20`` are two chunks - the release must
    happen after each of them (the next chunk is allocated right after).
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

    net_positions = [index for index, event in enumerate(events) if event == "net"]
    assert len(net_positions) == 2, events
    for position in net_positions:
        assert events[position + 1] == "empty_cache", (
            f"the chunk boundary did not release the VRAM: {events}"
        )


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
def test_every_console_write_of_the_facelib_reaches_the_logger(processor_module):
    """The prints of the detection, of the alignment and of the paste-back."""
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
    measured residual of the batched warp against cv2 is at most **1/255**
    on this content (``tests/test_gpu_batch_equivalence.py``); the bound
    below leaves the headroom of the float round trip.

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
    assert np.allclose(face_helper.affine_matrices[0], affine), (
        "the paste loop did not reuse the affine matrix of the crop"
    )


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
    # second convert-in may call the converter (2 * ceil(3 / 2) + 1 = 5).
    assert len(conversions) <= 5, conversions


@pytest.mark.unit
def test_the_chunked_mode_keeps_one_tail_buffer(processor_module, monkeypatch):
    """P-M4: the forced chunked mode releases its staging tail (D-R2.3.5).

    ``device_output_mode`` is patched to "chunked" (on CPU the modes are
    value-identical, the verdict just never selects a device mode by
    itself): 3 frames at 64x64 with one black - gray - frame, so 2
    eligible rows.  Recording every ``torch.empty``: exactly one float32
    allocation carries the whole-clip row count (the contractual
    ``final_batch``) and exactly one carries the eligible rows (the
    chunked core's ``out``).  After the return the eligible buffer is
    gone - the processor drops the name AND the row views
    (``batched_outputs`` becomes an index set) - while the returned IMAGE
    is ``final_batch`` itself and keeps it alive.
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
    frames[1] = 0.0  # the black frame is a gray frame - it leaves the batch

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
    eligible = [ref for shape, ref in float_rows if shape[0] == 2]
    assert len(final) == 1, float_rows
    assert len(eligible) == 1, float_rows
    # the returned IMAGE is the contractual whole-clip buffer
    assert final[0]() is not None, "final_batch must stay alive after the return"
    # the chunked core's eligible-rows buffer is released with its row views
    assert eligible[0]() is None, (
        "the chunked staging tail must be released after the indexed scatter"
    )
    assert tuple(output.shape) == (3, 512, 512, 3), output.shape


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
