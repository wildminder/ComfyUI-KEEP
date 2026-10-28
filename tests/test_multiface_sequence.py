"""The multi-face sequence pipeline (C2, design sections 3.2-3.6).

``only_center_face=False`` on a non-aligned **sequence** restores every
detected face of every frame.  The pins encode the facts the round
contracts:

1. the detector call keeps **all** faces (``only_keep_largest=False`` - the
   second collapse site: the vendored selector reduces to the largest face
   first, ``face_restoration_helper.py:247-255``, so the argument change is
   the fix, PM risk #1) while the ``only_center_face=True`` toggle keeps
   today's exact single-face sequence byte-for-byte;
2. the output of every frame is the **vendored all-faces composite** of that
   frame's faces - a real vendored ``FaceRestoreHelper`` driven by hand over
   the same landmarks and crops (the harness of
   ``tests/test_multiface_image.py``) - for the frames the batched paste
   takes and for the vendored-fallback frames alike, with faces appearing
   and disappearing across the clip;
3. the KEEP batch is **flattened over the faces** (``(1, total_faces, 3,
   512, 512)``), chunked by ``max_clip_length`` faces with the
   duplicate-to-2 trick for the 1-face chunk;
4. the guards: a ``None`` affine drops its face, an all-failed crop set
   returns the input, zero faces in the whole clip return the input;
5. a frame with an ineligible face (gray here) goes **entirely** to the
   vendored fallback with **all** its faces, and ``draw_box=True`` does the
   same for every frame - one box per pasted face, the vendored tail;
6. the progress total counts the restore in faces
   (``num_input_frames * 3 + total_faces``, PM risk #6);
7. the cancel polls fire in the four long loops and the k-th check aborts
   the run with no KEEP work after it;
8. no temporal landmark smoothing runs in the multi-face branch.

The stubs follow the established patterns: the real vendored helper with
stubbed model inits (``tests/test_multiface_image.py:make_helper``), the
``_KeepNet`` call log and the per-frame detector the image-path pins use.
"""

from __future__ import annotations

import importlib

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

#: The vendored 5-point 512 template (``face_restoration_helper.py:52-53``).
FACE_TEMPLATE = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]]
)

#: The frame size of the pins: the minimum side is 512, so the vendored
#: ``read_image`` does not upscale and the paste canvas is the frame size.
FRAME_HW = (512, 640)

#: The two faces of the recurring layout - an **overlapping** pair on the
#: content the sequential per-frame accumulator exists for.
FACE_A = (FACE_TEMPLATE - 256.0) * 0.3 + np.array([300.0, 240.0])
FACE_B = (FACE_TEMPLATE - 256.0) * 0.3 + np.array([308.0, 252.0])
#: The single face of the second frame layout.
FACE_C = (FACE_TEMPLATE - 256.0) * 0.3 + np.array([200.0, 380.0])


@pytest.fixture(scope="module")
def processor_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_processor")


@pytest.fixture(scope="module")
def keep_interrupt():
    return importlib.import_module("ComfyUI_KEEP.src.keep_interrupt")


@pytest.fixture(scope="module")
def helper_module():
    return importlib.import_module(
        "ComfyUI_KEEP.modules.deps.facelib.utils.face_restoration_helper")


@pytest.fixture(scope="module")
def gpu_ops():
    return importlib.import_module("ComfyUI_KEEP.src.keep_gpu_ops")


@pytest.fixture(scope="module")
def image_utils():
    return importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")


@pytest.fixture
def model_management():
    """The interrupt module (the mock of the gate env, the real one headless).

    The flag is session-global (one module): every test starts and ends
    with it cleared, so a cancelled run cannot leak into the next test.
    """
    module = importlib.import_module("comfy.model_management")
    module.interrupt_current_processing(False)
    yield module
    module.interrupt_current_processing(False)


# ---------------------------------------------------------------------------
# The stubs
# ---------------------------------------------------------------------------


class _StubParseNet:
    """A parse net with the vendored contract (``[0]`` are the logits).

    A 1x1 convolution over the color channels: deterministic and identical
    across instances of the same seed, so the processor's helper and the
    reference's helper parse the faces with the very same mask.
    """

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
    """A face detector with per-frame face sets (the appearing clip).

    The vendored detection scales the input to ``resize=640`` and divides
    the bboxes back afterwards (``face_restoration_helper.py:210-229``) -
    the rows here are pre-multiplied with that scale.  Row layout:
    ``[x1, y1, x2, y2, score, lx1, ly1, ..., lx5, ly5]``.

    ``detect_faces`` is called once per ``get_face_landmarks_5`` and the
    detection loop calls that once per frame, in frame order - the call
    index is therefore the frame index.
    """

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


def _make_pack(keep_net, face_helper):
    """A model pack the processor can consume."""
    return type(
        "Pack",
        (),
        {
            "keep_net": keep_net,
            "face_helper": face_helper,
            "bg_upscale_model": None,
            "face_upscale_model": None,
            "device": torch.device("cpu"),
            "model_type_str": "KEEP",
        },
    )()


# ---------------------------------------------------------------------------
# The rig: a real vendored helper with stubbed model inits
# ---------------------------------------------------------------------------


@pytest.fixture
def make_helper(helper_module, monkeypatch):
    """A factory of real ``FaceRestoreHelper`` instances without weights.

    The detection / parsing initializations are stubbed (the weights are
    not on this machine), so every helper runs the real vendored code.
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


def _smooth_frames(num_frames: int, seed: int = 41) -> np.ndarray:
    """A clip of smooth frames (the warp residual stays inside its bound)."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(num_frames):
        small = rng.uniform(0.0, 255.0, size=(32, 32, 3)).astype(np.float32)
        smooth = cv2.resize(small, (FRAME_HW[1], FRAME_HW[0]), interpolation=cv2.INTER_CUBIC)
        frames.append(np.clip(smooth, 0, 255).astype(np.uint8))
    return np.stack(frames)


def _sequence_tensor(image_utils, frames_bgr: np.ndarray) -> torch.Tensor:
    """The BGR frames as the ComfyUI IMAGE tensor the node hands over."""
    return image_utils.bgr_u8_batch_to_comfy(frames_bgr)


def _attach_detector(face_helper, detector):
    face_helper.face_detector = detector


def _record_detection(face_helper) -> dict:
    """Wrap ``get_face_landmarks_5`` to record the call arguments."""
    recorded: dict = {}
    real = face_helper.get_face_landmarks_5

    def recording(**kwargs):
        recorded.update(kwargs)
        return real(**kwargs)

    face_helper.get_face_landmarks_5 = recording
    return recorded


def _run_sequence(processor_module, keep_net, face_helper, frames, **kwargs):
    """One ``process_image_sequence`` run through the multi-face branch by default."""
    processor = processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))
    return processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=kwargs.pop("final_upscale_factor", 1.0),
        has_aligned_frames=kwargs.pop("has_aligned_frames", False),
        only_center_face=kwargs.pop("only_center_face", False),
        draw_box=kwargs.pop("draw_box", False),
        max_clip_length=kwargs.pop("max_clip_length", 20),
    )


def _background(image: np.ndarray, upscale_factor: float) -> np.ndarray:
    """The paste background: the optional bg model (none here), then the size."""
    target_h, target_w = int(image.shape[0] * upscale_factor), int(image.shape[1] * upscale_factor)
    if image.shape[0] != target_h or image.shape[1] != target_w:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return image


def _vendored_frame_reference(
    ref_helper, frame_bgr, landmarks, upscale_factor: float = 1.0,
    draw_box: bool = False,
) -> np.ndarray:
    """The vendored composite of one frame's faces (the pin's reference).

    The helper's own multi-face flow, driven by hand: ``align_warp_face``
    over the frame's landmarks, one restored face appended per crop (the
    KEEP stub restores the crop - the uint8 round trip through
    ``img2tensor`` / ``tensor2img`` is bit-exact), one
    ``get_inverse_affine``, one ``paste_faces_to_input_image``.  With an
    empty landmark list the paste returns the background at the target size
    - the composite of the frames without a face.
    """
    ref_helper.clean_all()
    ref_helper.upscale_factor = upscale_factor
    ref_helper.read_image(frame_bgr)
    ref_helper.all_landmarks_5 = [np.asarray(lm, dtype=np.float64) for lm in landmarks]
    ref_helper.align_warp_face()
    for face in ref_helper.cropped_faces:
        ref_helper.add_restored_face(face.astype("uint8"))
    ref_helper.get_inverse_affine(None)
    return ref_helper.paste_faces_to_input_image(
        upsample_img=_background(frame_bgr, upscale_factor),
        draw_box=draw_box,
        face_upsampler=None,
    )


def _cancel_on_call(model_management, cancel_at: int):
    """A ``check_interrupted`` stand-in that raises on the ``cancel_at``-th call."""
    counter = {"calls": 0}

    def check_interrupted():
        counter["calls"] += 1
        if counter["calls"] >= cancel_at:
            raise model_management.InterruptProcessingException()

    return check_interrupted, counter


# ---------------------------------------------------------------------------
# Pin 1 - the detector call (the second collapse site, PM risk #1)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_multiface_sequence_detector_call_keeps_all_faces(
    processor_module, make_helper
):
    """``only_center_face=False`` detects with ``only_keep_largest=False``.

    The argument change **is** the fix: behind ``only_keep_largest=True``
    the vendored selector reduces every detection to the largest face
    (``face_restoration_helper.py:247-255``) and no downstream indexing
    could ever see the others.  All three faces of the clip reach the KEEP
    net as one flattened batch - the time axis is the faces, not the frames.
    """
    image_utils = importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")
    frames = _sequence_tensor(image_utils, _smooth_frames(3))
    frames_in = image_utils.comfy_batch_to_bgr_u8(frames)
    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet(seed=101)
    _attach_detector(face_helper, _SequenceDetector([[FACE_A, FACE_B], [FACE_C], []]))
    recorded = _record_detection(face_helper)
    keep_net = _KeepNet()

    _run_sequence(processor_module, keep_net, face_helper, frames)

    assert recorded == {
        "only_center_face": False, "resize": 640,
        "eye_dist_threshold": 5, "only_keep_largest": False,
    }, recorded
    assert keep_net.calls == [((1, 3, 3, 512, 512), False)], (
        f"the flattened batch is (1, total_faces, 3, 512, 512), got {keep_net.calls}"
    )
    assert frames_in.shape == (3, *FRAME_HW, 3)


@pytest.mark.unit
def test_the_toggle_keeps_the_single_face_sequence_path(processor_module, make_helper):
    """``only_center_face=True`` keeps today's collapsed detector call.

    The toggle's contract is byte-for-byte today (design section 3.1): the
    largest face wins at the detector and one face per **frame** reaches the
    net - no flattening, and the multi-face branch is not entered.
    """
    image_utils = importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")
    frames = _sequence_tensor(image_utils, _smooth_frames(3))
    layout = [[FACE_A, FACE_B], [FACE_A, FACE_B], [FACE_A, FACE_B]]

    toggle_helper = make_helper(use_parse=True)
    toggle_helper.face_parse = _StubParseNet(seed=103)
    _attach_detector(toggle_helper, _SequenceDetector(layout))
    recorded = _record_detection(toggle_helper)
    toggle_net = _KeepNet()
    toggle = _run_sequence(processor_module, toggle_net, toggle_helper, frames,
                           only_center_face=True)

    assert recorded == {
        "only_center_face": True, "resize": 640,
        "eye_dist_threshold": 5, "only_keep_largest": True,
    }, recorded
    assert toggle_net.calls == [((1, 3, 3, 512, 512), False)], toggle_net.calls

    multi_helper = make_helper(use_parse=True)
    multi_helper.face_parse = _StubParseNet(seed=103)
    _attach_detector(multi_helper, _SequenceDetector(layout))
    multi = _run_sequence(processor_module, _KeepNet(), multi_helper, frames)

    assert not torch.equal(multi, toggle), (
        "the multi-face run must restore more than the single-face toggle"
    )


# ---------------------------------------------------------------------------
# Pin 2 - the end-to-end output is the vendored all-faces composite
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_multiface_sequence_output_is_the_vendored_all_faces_composite(
    processor_module, make_helper, image_utils
):
    """Faces appearing and disappearing across the clip, vendored composite.

    Frame 0 carries two **overlapping** faces - the sequential per-frame
    accumulator the t3 paste core exists for - frame 1 one face and frame 2
    none.  Every frame comes back as the vendored all-faces composite of
    that frame's faces: the batched frames within the t3 residual bound
    (3/255 on smooth content) and the no-face frame bit-exact (the vendored
    paste of zero faces returns the background at the target size).
    """
    frames_bgr = _smooth_frames(3)
    frames = _sequence_tensor(image_utils, frames_bgr)
    frames_in = image_utils.comfy_batch_to_bgr_u8(frames)
    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet(seed=107)
    _attach_detector(face_helper, _SequenceDetector([[FACE_A, FACE_B], [FACE_C], []]))

    output = _run_sequence(processor_module, _KeepNet(), face_helper, frames)

    assert tuple(output.shape) == (3, *FRAME_HW, 3), output.shape
    ref_helper = make_helper(use_parse=True)
    ref_helper.face_parse = _StubParseNet(seed=107)
    references = [
        _vendored_frame_reference(ref_helper, frames_in[0], [FACE_A, FACE_B]),
        _vendored_frame_reference(ref_helper, frames_in[1], [FACE_C]),
        _vendored_frame_reference(ref_helper, frames_in[2], []),
    ]
    expected = image_utils.bgr_u8_batch_to_comfy(np.stack(references))
    for row in (0, 1):
        drift = (output[row] - expected[row]).abs().max().item()
        assert drift <= 3.0 / 255.0 + 1e-6, (
            f"frame {row}: the multi-face sequence drifted {drift * 255:.2f}/255 "
            "from the vendored all-faces composite"
        )
    assert torch.equal(output[2], expected[2]), (
        "the frame without a face must come back as the background, bit-exact"
    )


# ---------------------------------------------------------------------------
# Pin 3 - the flattened KEEP batch
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_flattened_keep_batch_chunks_over_faces(processor_module, make_helper, image_utils):
    """The restore chunks over **faces**, the 1-face chunk duplicated to 2.

    Four faces (two per frame) with ``max_clip_length=2`` are two chunks of
    two faces; three faces with the same clip size are one chunk of two and
    one **single-face** chunk - the duplicate-to-2 trick of the KEEP
    contract, now in face units; without the chunk cap the whole clip is one
    flattened batch.
    """
    frames = _sequence_tensor(image_utils, _smooth_frames(2))

    def run(faces_per_frame, max_clip_length):
        keep_net = _KeepNet()
        face_helper = make_helper(use_parse=True)
        face_helper.face_parse = _StubParseNet(seed=109)
        _attach_detector(face_helper, _SequenceDetector(faces_per_frame))
        _run_sequence(processor_module, keep_net, face_helper, frames,
                      max_clip_length=max_clip_length)
        return keep_net.calls

    assert run([[FACE_A, FACE_B], [FACE_A, FACE_B]], 20) == [
        ((1, 4, 3, 512, 512), False)], "the whole clip is one flattened face batch"
    assert run([[FACE_A, FACE_B], [FACE_A, FACE_B]], 2) == [
        ((1, 2, 3, 512, 512), False), ((1, 2, 3, 512, 512), False)], (
        "the chunk boundaries are counted in faces, not frames"
    )
    assert run([[FACE_A, FACE_B], [FACE_C, FACE_C]], 2) == [
        ((1, 2, 3, 512, 512), False), ((1, 2, 3, 512, 512), False)]
    # three faces, clip of two: the second chunk is the duplicated 1-face one
    assert run([[FACE_A, FACE_B], [FACE_C], []], 2) == [
        ((1, 2, 3, 512, 512), False), ((1, 2, 3, 512, 512), False)], (
        "the 1-face chunk keeps the duplicate-to-2 trick"
    )


# ---------------------------------------------------------------------------
# Pin 4 - the guards
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_none_affine_drops_its_face_from_the_sequence(
    processor_module, make_helper, image_utils, monkeypatch
):
    """A ``None`` affine (the LMEDS failure) drops the face before the crop.

    The failure is forced deterministically for the NaN-marked landmarks (a
    real LMEDS failure mode - the detector can emit them, and the
    eye-distance gate cannot filter a NaN).  The dropped face never reaches
    the batch and the surviving faces keep the vendored composite.
    """
    frames_bgr = _smooth_frames(2)
    frames = _sequence_tensor(image_utils, frames_bgr)
    frames_in = image_utils.comfy_batch_to_bgr_u8(frames)
    broken = FACE_B.copy()
    broken[4] = np.nan  # the LMEDS-killer marker (the eye points stay finite)
    real_estimate = cv2.estimateAffinePartial2D

    def estimate(src, dst, *args, **kwargs):
        if not np.all(np.isfinite(src)):
            return (None, None)
        return real_estimate(src, dst, *args, **kwargs)

    monkeypatch.setattr(cv2, "estimateAffinePartial2D", estimate)

    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet(seed=113)
    _attach_detector(face_helper, _SequenceDetector([[FACE_A, broken], [FACE_C]]))
    keep_net = _KeepNet()

    output = _run_sequence(processor_module, keep_net, face_helper, frames)

    assert keep_net.calls == [((1, 2, 3, 512, 512), False)], (
        f"only the two surviving faces reach the net, got {keep_net.calls}"
    )
    ref_helper = make_helper(use_parse=True)
    ref_helper.face_parse = _StubParseNet(seed=113)
    expected = image_utils.bgr_u8_batch_to_comfy(np.stack([
        _vendored_frame_reference(ref_helper, frames_in[0], [FACE_A]),
        _vendored_frame_reference(ref_helper, frames_in[1], [FACE_C]),
    ]))
    drift = (output - expected).abs().max().item()
    assert drift <= 3.0 / 255.0 + 1e-6, (
        f"the dropped face must not shift the pairing, drifted {drift * 255:.2f}/255"
    )


@pytest.mark.unit
def test_zero_faces_in_the_clip_return_the_input_tensor(
    processor_module, make_helper, image_utils
):
    """No face anywhere in the clip - the input tensor comes back (design 3.2)."""
    frames = _sequence_tensor(image_utils, _smooth_frames(3))
    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet()
    _attach_detector(face_helper, _SequenceDetector([[], [], []]))
    keep_net = _KeepNet()

    output = _run_sequence(processor_module, keep_net, face_helper, frames)

    assert output is frames, "the whole-clip zero-face guard returns the input"
    assert keep_net.calls == [], keep_net.calls


@pytest.mark.unit
def test_an_all_failed_crop_set_returns_the_input_tensor(
    processor_module, make_helper, image_utils
):
    """Every affine failed - there is nothing to crop, restore or paste."""
    frames = _sequence_tensor(image_utils, _smooth_frames(2))
    broken = FACE_A.copy()
    broken[4] = np.nan
    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet()
    _attach_detector(face_helper, _SequenceDetector([[broken], [broken]]))
    keep_net = _KeepNet()

    monkey = cv2.estimateAffinePartial2D

    def estimate(src, dst, *args, **kwargs):
        if not np.all(np.isfinite(src)):
            return (None, None)
        return monkey(src, dst, *args, **kwargs)

    cv2.estimateAffinePartial2D = estimate
    try:
        output = _run_sequence(processor_module, keep_net, face_helper, frames)
    finally:
        cv2.estimateAffinePartial2D = monkey

    assert output is frames, "the all-failed crop set returns the input unchanged"
    assert keep_net.calls == [], keep_net.calls


# ---------------------------------------------------------------------------
# Pin 5 - the ineligible frame goes entirely to the vendored fallback
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_gray_frame_goes_to_the_vendored_fallback_with_all_its_faces(
    processor_module, make_helper, image_utils
):
    """A grayscale frame takes the vendored paste with **all** of its faces.

    The all-or-nothing eligibility (design section 3.4): the gray frame is
    not device-batched, so its two faces composite through one vendored
    ``paste_faces_to_input_image`` call - the reference composite including
    the vendored gray conversion of ``add_restored_face`` - while the color
    frame next to it stays on the batched path.
    """
    gray = np.full((*FRAME_HW, 3), 102, dtype=np.uint8)
    frames_bgr = np.stack([gray, _smooth_frames(1)[0]])
    frames = _sequence_tensor(image_utils, frames_bgr)
    frames_in = image_utils.comfy_batch_to_bgr_u8(frames)
    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet(seed=127)
    _attach_detector(face_helper, _SequenceDetector([[FACE_A, FACE_B], [FACE_A, FACE_B]]))
    paste_calls: list[int] = []
    real_paste = face_helper.paste_faces_to_input_image

    def recording_paste(*args, **kwargs):
        paste_calls.append(len(face_helper.restored_faces))
        return real_paste(*args, **kwargs)

    face_helper.paste_faces_to_input_image = recording_paste

    output = _run_sequence(processor_module, _KeepNet(), face_helper, frames)

    assert paste_calls == [2], (
        f"exactly the gray frame takes the vendored paste, with both faces: {paste_calls}"
    )
    ref_helper = make_helper(use_parse=True)
    ref_helper.face_parse = _StubParseNet(seed=127)
    expected = image_utils.bgr_u8_batch_to_comfy(np.stack([
        _vendored_frame_reference(ref_helper, frames_in[0], [FACE_A, FACE_B]),
        _vendored_frame_reference(ref_helper, frames_in[1], [FACE_A, FACE_B]),
    ]))
    assert torch.equal(output[0], expected[0]), (
        "the gray frame must composite through the vendored paste, bit-exact"
    )
    drift = (output[1] - expected[1]).abs().max().item()
    assert drift <= 3.0 / 255.0 + 1e-6, (
        f"the color frame drifted {drift * 255:.2f}/255 from the vendored composite"
    )


@pytest.mark.unit
def test_draw_box_pastes_every_frame_through_the_vendored_fallback(
    processor_module, make_helper, image_utils
):
    """``draw_box=True`` takes every frame to the vendored paste.

    The global gate of the batched paste (design section 3.4): the box
    needs the vendored masks.  Every frame's output is the vendored
    all-faces composite with one box per pasted face (PM open question 3,
    answered by the vendored tails ``face_restoration_helper.py:428-438``).
    """
    frames_bgr = _smooth_frames(3)
    frames = _sequence_tensor(image_utils, frames_bgr)
    frames_in = image_utils.comfy_batch_to_bgr_u8(frames)
    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet(seed=131)
    _attach_detector(face_helper, _SequenceDetector([[FACE_A, FACE_B], [FACE_C], []]))
    keep_net = _KeepNet()

    output = _run_sequence(processor_module, keep_net, face_helper, frames, draw_box=True)

    assert keep_net.calls == [((1, 3, 3, 512, 512), False)], keep_net.calls
    ref_helper = make_helper(use_parse=True)
    ref_helper.face_parse = _StubParseNet(seed=131)
    expected = image_utils.bgr_u8_batch_to_comfy(np.stack([
        _vendored_frame_reference(ref_helper, frames_in[0], [FACE_A, FACE_B], draw_box=True),
        _vendored_frame_reference(ref_helper, frames_in[1], [FACE_C], draw_box=True),
        _vendored_frame_reference(ref_helper, frames_in[2], [], draw_box=True),
    ]))
    assert torch.equal(output, expected), (
        "every draw_box frame must be the vendored composite with its per-face boxes"
    )


# ---------------------------------------------------------------------------
# Pin 6 - the progress counts the restore in faces
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_multiface_sequence_progress_counts_faces(
    processor_module, make_helper, image_utils, monkeypatch
):
    """The total is ``num_input_frames * 3 + total_faces`` (design section 3.6).

    Three frames, three faces: convert-in + detection spend ``2 * 3``, the
    restore ``3`` (one face-unit chunk) and the paste ``3``.
    """
    events: list[tuple] = []
    created: list = []

    class _RecordingProgress:
        def __init__(self, total):
            events.append(("total", total))
            created.append(self)

        def update(self, value):
            events.append(("update", value))

    monkeypatch.setattr(processor_module, "ProgressReporter", _RecordingProgress)

    frames = _sequence_tensor(image_utils, _smooth_frames(3))
    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet(seed=137)
    _attach_detector(face_helper, _SequenceDetector([[FACE_A, FACE_B], [FACE_C], []]))

    _run_sequence(processor_module, _KeepNet(), face_helper, frames)

    assert events == [
        ("total", 9),  # the frames-only estimate of the reporter
        ("update", 3),  # the batched convert-in
        ("update", 3),  # the detection
        ("update", 3),  # the restore, one chunk of three faces
        ("update", 1),  # the paste, frame by frame
        ("update", 1),
        ("update", 1),
    ], events
    assert created and created[0].total == 3 * 3 + 3, (
        f"the final total counts the restore in faces, got {created[0].total}"
    )


# ---------------------------------------------------------------------------
# Pin 7 - the cancel polls
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_multiface_sequence_polls_the_cancel_flag(
    processor_module, make_helper, image_utils, keep_interrupt, monkeypatch
):
    """The four long loops poll - and a no-op poll changes nothing.

    The detection per frame, the crop per face, the restore per chunk and
    the paste per frame (the paste module's own group polls add
    environment-sized groups on top).  The counting patch run must equal
    the unpatched run: the wiring does no harm.
    """
    frames = _sequence_tensor(image_utils, _smooth_frames(3))

    def run():
        keep_net = _KeepNet()
        face_helper = make_helper(use_parse=True)
        face_helper.face_parse = _StubParseNet(seed=139)
        _attach_detector(
            face_helper, _SequenceDetector([[FACE_A, FACE_B], [FACE_C], []]))
        return _run_sequence(processor_module, keep_net, face_helper, frames), keep_net

    unpatched, unpatched_net = run()

    counter = {"calls": 0}

    def noop_check():
        counter["calls"] += 1

    monkeypatch.setattr(keep_interrupt, "check_interrupted", noop_check)
    patched, patched_net = run()

    assert counter["calls"] >= 10, (
        f"the four loops must poll (3 detection + 3 crop + 1 restore + 3 paste), "
        f"saw {counter['calls']}"
    )
    assert torch.equal(unpatched, patched), "a no-op poll must not change the pixels"
    assert unpatched_net.calls == patched_net.calls, "a no-op poll must not change the KEEP calls"


@pytest.mark.unit
def test_the_kth_cancel_check_aborts_the_multiface_sequence(
    processor_module, make_helper, image_utils, model_management, keep_interrupt,
    monkeypatch
):
    """The k-th check escapes the multi-face branch with no KEEP work after it.

    Four faces over two frames with ``max_clip_length=1``: the checks fire
    2x in the detection loop and 4x in the crop loop (calls 1-6), then once
    per restore chunk (7-10).  Raising on the 9th call lets exactly two
    1-face chunks restore and kills everything after - the third chunk's
    KEEP forward never happens.
    """
    check, counter = _cancel_on_call(model_management, 9)
    monkeypatch.setattr(keep_interrupt, "check_interrupted", check)
    frames = _sequence_tensor(
        image_utils, _smooth_frames(2))
    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet(seed=149)
    _attach_detector(
        face_helper, _SequenceDetector([[FACE_A, FACE_B], [FACE_A, FACE_B]]))
    keep_net = _KeepNet()

    with pytest.raises(model_management.InterruptProcessingException):
        _run_sequence(processor_module, keep_net, face_helper, frames,
                      max_clip_length=1)

    assert counter["calls"] == 9, (
        f"2 detection + 4 crop + 3 restore-chunk polls before the raise, "
        f"saw {counter['calls']}"
    )
    assert keep_net.calls == [
        ((1, 2, 3, 512, 512), False), ((1, 2, 3, 512, 512), False),
    ], f"exactly the first two 1-face chunks may reach the KEEP net, got {keep_net.calls}"


# ---------------------------------------------------------------------------
# Pin 8 - no temporal smoothing
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_multiface_sequence_does_not_smooth_the_landmarks(
    processor_module, make_helper, image_utils, monkeypatch
):
    """The single-track temporal smoothing is not applicable to a face set.

    The ``(frames, 10)`` model ``interpolate_sequence`` /
    ``_gaussian_smooth`` blend the landmarks of *different faces* into each
    other, so the multi-face branch must not run them (per-frame
    independent detection, design section 3.2).
    """
    smoothed = {"count": 0}
    interpolated = {"count": 0}

    def counting_smooth(*args, **kwargs):
        smoothed["count"] += 1
        return processor_module._gaussian_smooth(*args, **kwargs)

    def counting_interpolate(*args, **kwargs):
        interpolated["count"] += 1
        return processor_module.interpolate_sequence(*args, **kwargs)

    monkeypatch.setattr(processor_module, "_gaussian_smooth", counting_smooth)
    monkeypatch.setattr(processor_module, "interpolate_sequence", counting_interpolate)

    frames = _sequence_tensor(image_utils, _smooth_frames(3))
    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet(seed=151)
    _attach_detector(face_helper, _SequenceDetector([[FACE_A, FACE_B], [FACE_C], []]))

    _run_sequence(processor_module, _KeepNet(), face_helper, frames)

    assert smoothed["count"] == 0 and interpolated["count"] == 0, (
        "the multi-face branch must not run the temporal landmark smoothing"
    )
