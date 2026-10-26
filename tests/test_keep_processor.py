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

import cv2
import numpy as np
import pytest
import torch


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

    frames = torch.rand((3, 16, 16, 3), dtype=torch.float32)
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
        ("total", 12),
        ("update", 3),  # the batched convert-in
        ("update", 3),  # the detection
        ("update", 3),  # the single KEEP chunk
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
