"""The multi-face single-image path (C2, design section 3.5).

``only_center_face=False`` on a non-aligned image restores **every**
detected face.  The pins encode the facts the round contracts:

1. the detector call keeps **all** faces (``only_keep_largest=False`` - the
   second collapse site: the vendored selector reduces to the largest face
   first, ``face_restoration_helper.py:247-255``, so the argument change is
   the fix, PM risk #1) while the ``only_center_face=True`` toggle keeps
   today's exact arguments byte-for-byte;
2. the output equals the **vendored reference composite** - a real vendored
   ``FaceRestoreHelper`` driven by hand over the same landmarks and faces
   (the harness of ``tests/test_gpu_paste_equivalence.py:91-104``), with
   both paste branches (the eroded soft mask and the production
   ``use_parse=True`` parse mask);
3. the guards: a black restore drops its face (the survivors' landmarks
   drive the paste - a dropped face must not shift the pairing), an
   all-black restore returns the original, a ``None`` affine drops its
   face, zero faces return the original;
4. the aligned path and the single-face toggle stay out of the branch;
5. the multi-face path polls the cancel flag at the three stage boundaries
   (after detection, after the KEEP restore, before the paste).

The stubs follow the established patterns: the real vendored helper with
stubbed model inits (``tests/test_gpu_paste_equivalence.py:91-104``), the
``_StubKeepNet`` call log (``:1323-1331``) and the ``_make_pack`` namespace
(``tests/test_keep_processor.py:97-106``).
"""

from __future__ import annotations

import importlib
import types

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


class _BlackSecondHalfKeepNet(_KeepNet):
    """The duplicated half of the batch comes back all black.

    The processor must keep the **first** ``F`` outputs
    (``net_output[:, 0:F]``): were it to take the duplicated half, every
    face would restore black and drop - and the reference pin below would
    fail with the original image.
    """

    def __call__(self, frames, need_upscale=False):
        self.calls.append((tuple(frames.shape), need_upscale))
        out = frames.clone()
        out[:, frames.shape[1] // 2:] = -1.0
        return out


class _BlackFaceKeepNet(_KeepNet):
    """One face's slots come back all black (the per-face all-black guard)."""

    def __init__(self, black_face: int) -> None:
        super().__init__()
        self._black_face = black_face

    def __call__(self, frames, need_upscale=False):
        self.calls.append((tuple(frames.shape), need_upscale))
        out = frames.clone()
        half = frames.shape[1] // 2
        out[:, self._black_face] = -1.0
        out[:, half + self._black_face] = -1.0
        return out


class _AllBlackKeepNet(_KeepNet):
    """Every face comes back all black (the drop-all guard)."""

    def __call__(self, frames, need_upscale=False):
        self.calls.append((tuple(frames.shape), need_upscale))
        return torch.full_like(frames, -1.0)


class _StubDetector:
    """A face detector with precomputed faces (the vendored bbox contract).

    The vendored detection scales the input to ``resize=640`` and divides
    the bboxes back afterwards (``face_restoration_helper.py:210-229``) -
    the rows here are pre-multiplied with that scale.  Row layout:
    ``[x1, y1, x2, y2, score, lx1, ly1, ..., lx5, ly5]``.
    """

    def __init__(self, faces: list[np.ndarray], frame_hw: tuple[int, int]) -> None:
        scale = max(1, 640 / min(frame_hw))
        rows = []
        for landmarks in faces:
            (x1, y1), (x2, y2) = landmarks.min(axis=0), landmarks.max(axis=0)
            rows.append(np.concatenate(([x1, y1, x2, y2, 0.99], landmarks.reshape(-1))))
        self._bboxes = np.stack(rows) * scale if rows else None

    def detect_faces(self, image, confidence):
        return self._bboxes


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


def _face(scale: float, center: tuple[float, float]) -> np.ndarray:
    """One face: the 512 template scaled and moved inside the frame."""
    return (FACE_TEMPLATE - 256.0) * scale + np.array(center)


def _noise_image(seed: int = 7) -> np.ndarray:
    """Per-pixel noise: the two composites cannot round onto each other."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(768, 1024, 3), dtype=np.uint8)


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


def _detect_all_faces(det_helper, image) -> list[np.ndarray]:
    """The standalone multi-face detection (the reference's face source)."""
    det_helper.clean_all()
    det_helper.read_image(image)
    det_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5,
        only_keep_largest=False)
    return list(det_helper.all_landmarks_5)


def _run(processor_module, keep_net, face_helper, image, **kwargs):
    """One ``process_image`` run through the multi-face branch by default."""
    processor = processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))
    return processor.process_image(
        cv2_image_orig=image,
        final_upscale_factor=kwargs.pop("final_upscale_factor", 1.0),
        has_aligned=kwargs.pop("has_aligned", False),
        only_center_face=kwargs.pop("only_center_face", False),
        draw_box=kwargs.pop("draw_box", False),
    )


def _background(image: np.ndarray, upscale_factor: float) -> np.ndarray:
    """The paste background: the optional bg model (none here), then the size."""
    target_h, target_w = int(image.shape[0] * upscale_factor), int(image.shape[1] * upscale_factor)
    if image.shape[0] != target_h or image.shape[1] != target_w:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return image


def _vendored_reference(
    ref_helper, image, landmarks, upscale_factor,
    draw_box: bool = False,
) -> np.ndarray:
    """The vendored composite of the given faces (the pin's reference).

    The helper's own multi-face flow, driven by hand: ``align_warp_face``
    over the landmarks, one restored face appended per crop (the KEEP stub
    restores the crop - the uint8 round trip through ``img2tensor`` /
    ``tensor2img`` is bit-exact), one ``get_inverse_affine`` and **one**
    ``paste_faces_to_input_image``.
    """
    ref_helper.clean_all()
    ref_helper.upscale_factor = upscale_factor
    ref_helper.read_image(image)
    ref_helper.all_landmarks_5 = list(landmarks)
    ref_helper.align_warp_face()
    for face in ref_helper.cropped_faces:
        ref_helper.add_restored_face(face.astype("uint8"))
    ref_helper.get_inverse_affine(None)
    return ref_helper.paste_faces_to_input_image(
        upsample_img=_background(image, upscale_factor),
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
# Pins 1 - the detector call (the second collapse site, PM risk #1)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_multiface_detector_call_keeps_all_faces(processor_module, make_helper):
    """``only_center_face=False`` detects with ``only_keep_largest=False``.

    The argument change **is** the fix: behind ``only_keep_largest=True``
    the vendored selector reduces every detection to the largest face
    (``face_restoration_helper.py:247-255``) and no downstream indexing
    could ever see the others.  Two detected faces must reach the KEEP net
    as two crops.
    """
    image = _noise_image()
    faces = [_face(0.4, (280, 250)), _face(0.22, (700, 520))]
    face_helper = make_helper()
    _attach_detector(face_helper, _StubDetector(faces, image.shape[:2]))
    recorded = _record_detection(face_helper)
    keep_net = _KeepNet()

    _run(processor_module, keep_net, face_helper, image)

    assert recorded == {
        "only_center_face": False, "resize": 640,
        "eye_dist_threshold": 5, "only_keep_largest": False,
    }, recorded
    assert keep_net.calls == [((1, 4, 3, 512, 512), False)], (
        f"two faces go in as one duplicated batch of 2F=4, got {keep_net.calls}"
    )


@pytest.mark.unit
def test_the_single_face_toggle_keeps_its_detector_arguments(processor_module, make_helper):
    """``only_center_face=True`` keeps today's exact arguments.

    The toggle's contract is byte-for-byte today: the largest face wins at
    the detector (the documented reality of the vendored selector), one
    crop reaches the net.
    """
    image = _noise_image()
    faces = [_face(0.4, (280, 250)), _face(0.22, (700, 520))]
    face_helper = make_helper()
    _attach_detector(face_helper, _StubDetector(faces, image.shape[:2]))
    recorded = _record_detection(face_helper)
    keep_net = _KeepNet()

    _run(processor_module, keep_net, face_helper, image, only_center_face=True)

    assert recorded == {
        "only_center_face": True, "resize": 640,
        "eye_dist_threshold": 5, "only_keep_largest": True,
    }, recorded
    assert keep_net.calls == [((1, 2, 3, 512, 512), False)], keep_net.calls


# ---------------------------------------------------------------------------
# Pin 2 - the output is the vendored reference composite
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_multiface_output_is_the_vendored_all_faces_composite(
    processor_module, make_helper,
):
    """Two faces, one vendored paste - the reference composite by construction.

    The KEEP stub blacks the duplicated half of the batch, so the pin also
    proves the first ``F`` outputs are taken (``net_output[:, 0:F]``): the
    wrong half would restore black and drop, and the original would come
    back instead of the composite.
    """
    image = _noise_image()
    faces = [_face(0.4, (280, 250)), _face(0.22, (700, 520))]
    face_helper = make_helper()
    _attach_detector(face_helper, _StubDetector(faces, image.shape[:2]))
    keep_net = _BlackSecondHalfKeepNet()

    output = _run(processor_module, keep_net, face_helper, image)

    ref_helper = make_helper()
    expected = _vendored_reference(ref_helper, image, faces, 1.0)
    assert np.array_equal(output, expected), "the multi-face output must be the vendored composite of all faces"
    assert keep_net.calls == [((1, 4, 3, 512, 512), False)], keep_net.calls


@pytest.mark.unit
def test_the_multiface_composite_with_the_production_parse_net(
    processor_module, make_helper,
):
    """The same reference under ``use_parse=True`` (the pack's real config).

    The vendored paste branch of the parse mask (the two Gaussian passes
    and the warped ``parse_mask_final``, ``face_restoration_helper.py:450-478``)
    must hold for every face of the composite as well.
    """
    image = _noise_image()
    faces = [_face(0.4, (280, 250)), _face(0.22, (700, 520))]
    face_helper = make_helper(use_parse=True)
    face_helper.face_parse = _StubParseNet()
    _attach_detector(face_helper, _StubDetector(faces, image.shape[:2]))
    keep_net = _KeepNet()

    output = _run(processor_module, keep_net, face_helper, image)

    ref_helper = make_helper(use_parse=True)
    ref_helper.face_parse = _StubParseNet()
    expected = _vendored_reference(ref_helper, image, faces, 1.0)
    assert np.array_equal(output, expected), "the parse-mask branch must composite all faces identically"


@pytest.mark.unit
def test_the_multiface_output_exceeds_the_largest_face_only(processor_module, make_helper):
    """The owner's regression: more than the largest face is restored.

    The single-face run (the toggle) restores the largest face only; the
    multi-face run restores both - the two composites must differ, and the
    single one must equal the vendored one-face reference (the harness
    agrees with the untouched single-face path).
    """
    image = _noise_image()
    faces = [_face(0.4, (280, 250)), _face(0.22, (700, 520))]
    detector = _StubDetector(faces, image.shape[:2])

    single_helper = make_helper()
    _attach_detector(single_helper, detector)
    single = _run(processor_module, _KeepNet(), single_helper, image, only_center_face=True)

    multi_helper = make_helper()
    _attach_detector(multi_helper, detector)
    multi = _run(processor_module, _KeepNet(), multi_helper, image)

    ref_helper = make_helper()
    assert np.array_equal(single, _vendored_reference(ref_helper, image, [faces[0]], 1.0)), (
        "the toggle run must stay the vendored largest-face composite"
    )
    assert not np.array_equal(multi, single), (
        "the second face must change the output - the one-face tracking is the bug being fixed"
    )


@pytest.mark.unit
def test_one_detected_face_matches_the_single_face_composite(processor_module, make_helper):
    """One face through the multi-face path degenerates to today's output.

    ``only_keep_largest=False`` with a single detection is the same face -
    the composite must equal the vendored one-face reference, and the
    duplicate-axis trick keeps the net contract at ``2F=2``.
    """
    image = _noise_image()
    faces = [_face(0.4, (280, 250))]
    face_helper = make_helper()
    _attach_detector(face_helper, _StubDetector(faces, image.shape[:2]))
    keep_net = _KeepNet()

    output = _run(processor_module, keep_net, face_helper, image)

    ref_helper = make_helper()
    expected = _vendored_reference(ref_helper, image, faces, 1.0)
    assert np.array_equal(output, expected), "a single face must composite exactly like the single-face path"
    assert keep_net.calls == [((1, 2, 3, 512, 512), False)], keep_net.calls


# ---------------------------------------------------------------------------
# Pin 3 - the guards
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_black_restore_drops_its_face_and_keeps_the_pairing(processor_module, make_helper):
    """A black restore drops with its landmarks - the pairing must not shift.
    Face 0 restores black and drops.  The surviving face 1 must paste
    against **its own** affine (the reference of face 1 alone): pasting it
    against face 0's matrix - the all-landmarks misalignment - would put
    its content at the wrong position and fail the pin.
    """
    image = _noise_image()
    faces = [_face(0.4, (280, 250)), _face(0.22, (700, 520))]
    face_helper = make_helper()
    _attach_detector(face_helper, _StubDetector(faces, image.shape[:2]))
    keep_net = _BlackFaceKeepNet(black_face=0)

    output = _run(processor_module, keep_net, face_helper, image)

    ref_helper = make_helper()
    expected = _vendored_reference(ref_helper, image, [faces[1]], 1.0)
    assert np.array_equal(output, expected), "the survivor must composite against its own affine"
    assert keep_net.calls == [((1, 4, 3, 512, 512), False)], (
        f"the guard runs after the restore - both crops reach the net, got {keep_net.calls}"
    )


@pytest.mark.unit
def test_all_black_restores_return_the_original_image(processor_module, make_helper):
    """Every face restored black - the original image comes back."""
    image = _noise_image()
    faces = [_face(0.4, (280, 250)), _face(0.22, (700, 520))]
    face_helper = make_helper()
    _attach_detector(face_helper, _StubDetector(faces, image.shape[:2]))
    keep_net = _AllBlackKeepNet()

    output = _run(processor_module, keep_net, face_helper, image)

    assert np.array_equal(output, image), "with every face dropped the original must be returned"


@pytest.mark.unit
def test_a_failed_affine_drops_its_face(processor_module, make_helper, monkeypatch):
    """A ``None`` affine (the LMEDS failure) drops the face from the batch.

    The vendored ``align_warp_face`` appends a ``None`` to **both**
    ``cropped_faces`` and ``affine_matrices``
    (``face_restoration_helper.py:317-318``); the face must drop out of the
    KEEP batch and out of the paste.  The failure is forced deterministically
    for the NaN-marked landmarks (a real LMEDS failure mode - the detector
    can emit them, and the eye-distance gate cannot filter a NaN).
    """
    image = _noise_image()
    good = _face(0.4, (280, 250))
    broken = _face(0.22, (700, 520))
    broken[4] = np.nan  # the LMEDS-killer marker (the eye points stay finite)
    real_estimate = cv2.estimateAffinePartial2D

    def estimate(src, dst, *args, **kwargs):
        if not np.all(np.isfinite(src)):
            return (None, None)
        return real_estimate(src, dst, *args, **kwargs)

    monkeypatch.setattr(cv2, "estimateAffinePartial2D", estimate)

    face_helper = make_helper()
    _attach_detector(face_helper, _StubDetector([good, broken], image.shape[:2]))
    keep_net = _KeepNet()

    output = _run(processor_module, keep_net, face_helper, image)

    ref_helper = make_helper()
    expected = _vendored_reference(ref_helper, image, [good], 1.0)
    assert np.array_equal(output, expected), "the LMEDS-failed face must not reach the batch or the paste"
    assert keep_net.calls == [((1, 2, 3, 512, 512), False)], keep_net.calls


@pytest.mark.unit
def test_no_detected_faces_return_the_original_image(processor_module, make_helper):
    """An empty detection returns the original (the zero-face guard)."""
    image = _noise_image()
    face_helper = make_helper()
    _attach_detector(face_helper, _StubDetector([], image.shape[:2]))
    keep_net = _KeepNet()

    output = _run(processor_module, keep_net, face_helper, image)

    assert np.array_equal(output, image), "no faces - the original comes back"
    assert keep_net.calls == [], keep_net.calls


# ---------------------------------------------------------------------------
# Pin 4 - the branch boundaries
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_aligned_path_stays_out_of_the_multiface_branch(processor_module, make_helper):
    """``has_aligned=True`` keeps the single-face semantics (out of scope).

    The aligned path never detects - the recording wrapper stays empty -
    and the net contract stays the aligned one (``2F=2`` for the one face).
    """
    image = _noise_image()
    face_helper = make_helper()
    recorded = _record_detection(face_helper)
    keep_net = _KeepNet()

    output = _run(processor_module, keep_net, face_helper, image, has_aligned=True)

    assert recorded == {}, f"the aligned path must not detect, saw {recorded}"
    assert keep_net.calls == [((1, 2, 3, 512, 512), False)], keep_net.calls
    assert output.shape == (512, 512, 3), output.shape


@pytest.mark.unit
def test_the_box_reaches_every_face_of_the_composite(processor_module, make_helper):
    """``draw_box=True`` draws one box per pasted face (the vendored tails).

    PM open question 3, answered by the vendored paste itself: the box
    masks accumulate per face (``face_restoration_helper.py:428-438``) and
    draw after the convert tail (``:509-521``).
    """
    image = _noise_image()
    faces = [_face(0.4, (280, 250)), _face(0.22, (700, 520))]
    face_helper = make_helper()
    _attach_detector(face_helper, _StubDetector(faces, image.shape[:2]))
    keep_net = _KeepNet()

    output = _run(processor_module, keep_net, face_helper, image, draw_box=True)

    ref_helper = make_helper()
    expected = _vendored_reference(ref_helper, image, faces, 1.0, draw_box=True)
    assert np.array_equal(output, expected), "the boxes must be the vendored per-face ones"


# ---------------------------------------------------------------------------
# Pin 5 - the cancel polls (the stage boundaries of design section 2.2)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_first_check_of_the_multiface_path_aborts_before_keep_work(
    processor_module, make_helper, model_management, keep_interrupt, monkeypatch,
):
    """The post-detection check raises before the KEEP net runs."""
    check, _ = _cancel_on_call(model_management, 1)
    monkeypatch.setattr(keep_interrupt, "check_interrupted", check)

    image = _noise_image()
    faces = [_face(0.4, (280, 250)), _face(0.22, (700, 520))]
    face_helper = make_helper()
    _attach_detector(face_helper, _StubDetector(faces, image.shape[:2]))
    keep_net = _KeepNet()

    with pytest.raises(model_management.InterruptProcessingException):
        _run(processor_module, keep_net, face_helper, image)

    assert keep_net.calls == [], f"no KEEP work after the cancel, got {keep_net.calls}"


@pytest.mark.unit
def test_the_multiface_path_polls_the_three_stage_boundaries(
    processor_module, make_helper, keep_interrupt, monkeypatch,
):
    """A clean run polls exactly the three boundaries of the image path.

    After detection, after the KEEP restore, before the vendored paste -
    the same boundaries the single-face path polls
    (``process_image`` / design section 2.2).
    """
    counter = {"calls": 0}

    def noop_check():
        counter["calls"] += 1

    monkeypatch.setattr(keep_interrupt, "check_interrupted", noop_check)

    image = _noise_image()
    faces = [_face(0.4, (280, 250)), _face(0.22, (700, 520))]
    face_helper = make_helper()
    _attach_detector(face_helper, _StubDetector(faces, image.shape[:2]))
    keep_net = _KeepNet()

    _run(processor_module, keep_net, face_helper, image)

    assert counter["calls"] == 3, (
        f"the multi-face image path polls 3 stage boundaries, saw {counter['calls']}"
    )
