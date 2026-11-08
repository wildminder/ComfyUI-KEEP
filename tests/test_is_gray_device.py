"""The device gray flags of the paste stage (the ``is_gray`` port).

The last host-side item of the paste stage was the ``is_gray``
computation: a full-canvas numpy pass per frame (41.02 ms of process CPU
per frame on CUDA - the paste-stage CPU optimization round, section 4.2).
The port computes the three channel variances in float64 **on the device**
and falls back to the numpy computation for the frames whose result lands
on the threshold.

The pins here are **flag agreements**, not tolerances: the device flags
have to be the flags of the numpy path, frame by frame.  The borderline
frame - the one whose variance sum is exactly the threshold (10.0) - is
built here and its construction is **proved** (the vendored PIL path
measures the very same 10.0 and the vendored comparison is a ``<=``), not
assumed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
from PIL import Image

from src.keep_vendored import vendor  # the pack-private namespace (Round 21)

#: The size of the borderline frame - 64x64 is divisible by 8, so the
#: three-eighths split of the construction is exact.
BORDERLINE_HW = (64, 64)


@pytest.fixture(scope="module")
def gpu_paste():
    return importlib.import_module("ComfyUI_KEEP.src.keep_gpu_paste")


@pytest.fixture(scope="module")
def misc_module():
    return vendor("facelib.utils.misc")


@pytest.fixture(scope="module")
def processor_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_processor")


@pytest.fixture(scope="module")
def image_utils():
    return importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")


def vendored_diff_sum(frame_bgr_u8: np.ndarray) -> float:
    """The variance sum of the vendored ``is_gray`` (``misc.py:146-160``).

    The vendored helper, through PIL: the three channel differences and
    their variances - the value the vendored flag compares with the
    threshold.
    """
    image = Image.fromarray(frame_bgr_u8)
    img1 = np.asarray(image.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(image.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(image.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    return float((diff1 + diff2 + diff3) / 3.0)


def borderline_frame(height: int = BORDERLINE_HW[0], width: int = BORDERLINE_HW[1]) -> np.ndarray:
    """A frame whose variance sum is **exactly** the threshold (10.0).

    The construction: ``blue == green`` everywhere (so the first variance
    is zero), ``red = green + d``, and ``d`` takes the value 8 on three
    eighths of the pixels and 0 on the rest.  The variance of ``d`` is
    then ``3/8 * 5/8 * 8**2 = 15``, so the three variances are ``0``,
    ``15`` and ``15`` and their mean is exactly ``10.0`` - the threshold.
    The vendored comparison is a ``<=``, so the frame stays gray.

    Every value of the construction is an exact float64 (the sums are
    integers below ``2**53``), so the vendored measurement is exactly
    ``10.0`` - the test asserts it.
    """
    pixels = height * width
    assert pixels % 8 == 0, "the three-eighths split has to be exact"
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 1] = 100  # the green channel
    difference = np.zeros(pixels, dtype=np.int16)
    difference[: (3 * pixels) // 8] = 8
    frame[..., 2] = (100 + difference.reshape(height, width)).astype(np.uint8)
    return frame


def _random_frames(seed: int, num_frames: int = 6) -> np.ndarray:
    """A batch of random frames with one gray frame in the middle."""
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(num_frames, 48, 64, 3), dtype=np.uint8)
    frames[num_frames // 2] = np.repeat(
        frames[num_frames // 2][..., :1], 3, axis=2
    )
    return frames


#: The 5-point 512 template (``face_restoration_helper.py:52-53``).
FACE_TEMPLATE = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]]
)


class _StubFaceHelper:
    """A face helper stub with the vendored contract (no weight models)."""

    def __init__(self, landmarks: np.ndarray) -> None:
        self.face_template = FACE_TEMPLATE
        self.all_landmarks_5: list = []
        self.restored_faces: list = []
        self.cropped_faces: list = []
        self.affine_matrices: list = []
        self.input_img = None
        self.is_gray = False
        self.upscale_factor = 1.0
        self._landmarks = landmarks

    def clean_all(self) -> None:
        self.cropped_faces = []
        self.restored_faces = []

    # keep-fix5-implement1 (2026-11-03): the vendored ``read_image`` gained
    # the ``check_gray`` flag (plan Step 2.2) - the sequence sites pass it.
    def read_image(self, image, check_gray: bool = True) -> None:
        self.input_img = image

    def get_face_landmarks_5(self, **kwargs) -> int:
        self.all_landmarks_5 = [self._landmarks]
        return 1

    def add_restored_face(self, face, **kwargs) -> None:
        self.restored_faces = [face]

    def get_inverse_affine(self, *args, **kwargs) -> None:
        pass

    def paste_faces_to_input_image(self, *args, **kwargs) -> np.ndarray:
        return np.zeros((64, 96, 3), dtype=np.uint8)


class _StubKeepNet:
    """A KEEP net that returns its input (the restored face is the crop)."""

    def __call__(self, frames, need_upscale=False):
        return frames


def _run_sequence(processor_module, device: torch.device) -> None:
    """Run the sequence path of the processor on the given device."""
    landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
    pack = type(
        "Pack",
        (),
        {
            "keep_net": _StubKeepNet(),
            "face_helper": _StubFaceHelper(landmarks),
            "bg_upscale_model": None,
            "face_upscale_model": None,
            "device": device,
            "model_type_str": "KEEP",
        },
    )()
    processor = processor_module.KEEPFaceProcessor(pack)
    processor.process_image_sequence(
        image_sequence_tensor=torch.rand((1, 64, 96, 3), dtype=torch.float32),
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )


class _RecordingGrayFaceHelper(_StubFaceHelper):
    """The stub helper that records the gray flags the fallback paste reads.

    The fallback loop sets ``face_helper.is_gray`` from the clip's flag
    array right before the vendored paste, so the value the paste sees is
    the flag the row was given.
    """

    def __init__(self, landmarks: np.ndarray) -> None:
        super().__init__(landmarks)
        self.gray_flags_read: list[bool] = []

    def paste_faces_to_input_image(self, *args, **kwargs) -> np.ndarray:
        self.gray_flags_read.append(bool(self.is_gray))
        return super().paste_faces_to_input_image(*args, **kwargs)


def _run_sequence_with(processor_module, device: torch.device, helper,
                        frames: torch.Tensor, max_clip_length: int = 20):
    """Run the sequence path of the processor with the given helper."""
    pack = type(
        "Pack",
        (),
        {
            "keep_net": _StubKeepNet(),
            "face_helper": helper,
            "bg_upscale_model": None,
            "face_upscale_model": None,
            "device": device,
            "model_type_str": "KEEP",
        },
    )()
    processor = processor_module.KEEPFaceProcessor(pack)
    return processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=False,
        max_clip_length=max_clip_length,
    )


def _gray_and_colour_frames(seed: int = 17, num_frames: int = 4,
                            gray_rows: tuple[int, ...] = (0, 3)) -> torch.Tensor:
    """A clip of IMAGE rows: the named rows are gray, the others colour.

    A gray row has all three channels equal - the vendored ``is_gray``
    measures the variance of the channel differences, so the row's flag
    is True and the pins are not vacuous.
    """
    rng = np.random.default_rng(seed)
    rows = rng.uniform(0.0, 1.0, size=(num_frames, 64, 96, 3)).astype(np.float32)
    for index in gray_rows:
        rows[index] = rows[index, :, :, :1].copy()
    return torch.from_numpy(rows)


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device for the device pin"
)
def test_the_device_flags_are_the_flags_of_the_numpy_path(gpu_paste):
    """The device flags == the numpy flags, frame by frame (random frames)."""
    frames = _random_frames(97)

    device_flags = gpu_paste.is_gray_batch(frames, device=torch.device("cuda"))
    numpy_flags = gpu_paste.is_gray_batch(frames)

    assert list(device_flags) == list(numpy_flags), (list(device_flags), list(numpy_flags))
    assert device_flags.dtype == np.bool_
    assert device_flags.shape == (len(frames),)
    # the batch has both flags - the pin is not vacuous
    assert list(numpy_flags).count(True) == 1, list(numpy_flags)
    assert list(numpy_flags).count(False) == len(frames) - 1, list(numpy_flags)


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device for the device pin"
)
def test_the_device_flags_are_the_flags_of_the_numpy_path_on_flat_frames(gpu_paste):
    """The flat (gray) frames: the three variances are zero, the flags gray."""
    height, width = 32, 48
    flat = np.zeros((3, height, width, 3), dtype=np.uint8)
    flat[0] = 0
    flat[1] = 128
    flat[2] = 255
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    gradient = np.repeat(gradient[..., None], 3, axis=2)
    frames = np.concatenate([flat, gradient[None]])

    device_flags = gpu_paste.is_gray_batch(frames, device=torch.device("cuda"))
    numpy_flags = gpu_paste.is_gray_batch(frames)

    assert list(device_flags) == list(numpy_flags)
    assert list(device_flags) == [True] * len(frames)


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device for the device pin"
)
def test_the_device_flags_are_the_flags_of_the_numpy_path_on_saturated_frames(gpu_paste):
    """The saturated frames: the extremes of the uint8 range."""
    height, width = 32, 48
    checker = (np.indices((height, width)).sum(axis=0) % 2).astype(np.uint8)
    saturated = np.repeat((checker * 255)[..., None], 3, axis=2)
    colour = np.zeros((height, width, 3), dtype=np.uint8)
    colour[..., 0] = 255
    colour[..., 1] = checker * 255
    colour[..., 2] = 0
    frames = np.stack([saturated, colour])

    device_flags = gpu_paste.is_gray_batch(frames, device=torch.device("cuda"))
    numpy_flags = gpu_paste.is_gray_batch(frames)

    assert list(device_flags) == list(numpy_flags)
    assert list(device_flags) == [True, False]


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device for the device pin"
)
def test_the_borderline_frame_keeps_its_gray_flag(
    gpu_paste, misc_module, monkeypatch
):
    """The frame whose variance sum is exactly the threshold stays gray.

    The construction is proved here: the vendored PIL path measures the
    very same ``10.0`` (and the vendored comparison is a ``<=``), the
    device result lands in the borderline band, and the numpy fallback
    decides the flag.
    """
    frame = borderline_frame()
    frames = frame[None]

    # the construction: the vendored path measures exactly the threshold
    assert vendored_diff_sum(frame) == 10.0
    assert misc_module.is_gray(frame, threshold=10) is True
    numpy_flags = gpu_paste.is_gray_batch(frames)
    assert list(numpy_flags) == [True]

    # the device result lands on the threshold -> the fallback decides it
    values = gpu_paste._is_gray_diff_sums_device(frames, torch.device("cuda"))
    assert abs(values[0] - 10.0) <= gpu_paste.GRAY_BORDERLINE_EPSILON

    calls: list[np.ndarray] = []
    original = gpu_paste._is_gray_diff_sum_numpy

    def counting(frame):
        calls.append(frame)
        return original(frame)

    monkeypatch.setattr(gpu_paste, "_is_gray_diff_sum_numpy", counting)

    device_flags = gpu_paste.is_gray_batch(frames, device=torch.device("cuda"))

    assert list(device_flags) == list(numpy_flags) == [True]
    assert len(calls) == 1, calls


@pytest.mark.unit
def test_the_cpu_device_keeps_the_numpy_path(gpu_paste):
    """A CPU device (and no device at all) keeps the host computation."""
    frames = _random_frames(101, num_frames=4)

    reference = gpu_paste.is_gray_batch(frames)
    on_cpu = gpu_paste.is_gray_batch(frames, device=torch.device("cpu"))
    without_device = gpu_paste.is_gray_batch(frames, device=None)

    assert list(on_cpu) == list(reference)
    assert list(without_device) == list(reference)


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device for the device pin"
)
def test_an_empty_batch_has_no_flags(gpu_paste):
    """An empty batch is an empty flag array (the device path included)."""
    frames = np.zeros((0, 16, 16, 3), dtype=np.uint8)

    flags = gpu_paste.is_gray_batch(frames, device=torch.device("cuda"))

    assert flags.shape == (0,)
    assert flags.dtype == np.bool_


@pytest.mark.unit
def test_a_bad_shape_is_rejected(gpu_paste):
    """The shape check of the batch is unchanged."""
    with pytest.raises(ValueError):
        gpu_paste.is_gray_batch(np.zeros((4, 16, 16), dtype=np.uint8))


@pytest.mark.unit
def test_the_processor_keeps_the_numpy_flags_on_the_cpu_device(
    processor_module, monkeypatch
):
    """On the CPU device the processor calls the numpy path (no device)."""
    calls: list = []
    original = processor_module.is_gray_batch

    def spy(frames, device=None):
        calls.append(device)
        return original(frames, device=device)

    monkeypatch.setattr(processor_module, "is_gray_batch", spy)

    _run_sequence(processor_module, torch.device("cpu"))

    assert calls, "the processor did not call is_gray_batch"
    assert calls[0] is None, calls


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device for the device pin"
)
def test_the_processor_passes_the_cuda_device(processor_module, monkeypatch):
    """On CUDA the processor hands the device to ``is_gray_batch``."""
    calls: list = []
    original = processor_module.is_gray_batch

    def spy(frames, device=None):
        calls.append(device)
        return original(frames, device=device)

    monkeypatch.setattr(processor_module, "is_gray_batch", spy)

    _run_sequence(processor_module, torch.device("cuda"))

    assert calls, "the processor did not call is_gray_batch"
    assert torch.device(calls[0]).type == "cuda", calls


# ---------------------------------------------------------------------------
# Round 17 - the fallback rows' gray flags (the whole-clip gray pass is gone)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_fallback_gray_flags_are_the_whole_clip_flags(
    processor_module, gpu_paste, image_utils, monkeypatch
):
    """The fallback rows' gray flags are the whole-clip flags, exactly.

    Round 17 (the final stage's RAM) replaced the whole-clip gray pass
    with a **per fallback group** pass: the flags are computed with the
    group's frames, one group at a time, so the whole-clip uint8 stack
    (and the whole-clip gray pass) are gone.  The values do not move -
    ``is_gray_batch`` decides **per frame**, so a group is the same set of
    decisions as the whole clip.

    The pin runs the sequence path over a 4-frame clip whose helper has
    no parse net: the batched paste takes no row, so every row falls
    back to the vendored paste - the rows that read the flags.  The
    captured flags of the fallback groups must equal the flags of the
    whole clip, exactly, and the flags the fallback paste reads must be
    the same values.
    """
    calls: list[tuple[np.ndarray, object, np.ndarray]] = []
    original = processor_module.is_gray_batch

    def spy(frames, device=None):
        flags = original(frames, device=device)
        calls.append((np.asarray(frames), device, np.asarray(flags)))
        return flags

    monkeypatch.setattr(processor_module, "is_gray_batch", spy)

    frames = _gray_and_colour_frames()
    helper = _RecordingGrayFaceHelper(
        (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0]))
    _run_sequence_with(
        processor_module, torch.device("cpu"), helper, frames, max_clip_length=2)

    whole = image_utils.comfy_batch_to_bgr_u8(frames)
    whole_flags = gpu_paste.is_gray_batch(whole)

    assert calls, "the processor did not call is_gray_batch"
    # the CPU device contract: the host (numpy) computation, no device
    assert calls[0][1] is None, [call[1] for call in calls]
    # the pin is not vacuous: the clip has both flags
    assert list(whole_flags) == [True, False, False, True], list(whole_flags)
    # the flags of the fallback groups are the flags of the whole clip
    captured = np.concatenate([flags for _, _, flags in calls])
    assert list(captured) == list(whole_flags), (
        list(captured), list(whole_flags)
    )
    # the flags the fallback paste reads are the same values
    assert helper.gray_flags_read == list(whole_flags), helper.gray_flags_read
    # the gray pass runs in groups (max_clip_length rows), never the clip
    assert [len(group) for group, _, _ in calls] == [2, 2], calls
    # every group holds whole-clip rows, byte for byte
    whole_rows = {row.tobytes() for row in whole}
    for group, _, _ in calls:
        for row in group:
            assert row.tobytes() in whole_rows


@pytest.mark.unit
def test_the_gray_pass_covers_only_the_fallback_rows(
    processor_module, gpu_paste, image_utils, monkeypatch
):
    """The eligible rows' flags are never computed - they were never read.

    The fallback set is the rows the batched paste did **not** take (the
    twin's returned keys), so the gray pass - and the conversion of the
    rows it needs - covers the fallback rows only.  The eligible rows'
    flags are never computed: they were never read (the batched rows do
    not use them), so they stay False.

    The pin stubs the twin with a mapping for rows 1 and 3, so the
    fallback rows are exactly 0 and 2 - the gray pass must see those two
    rows (and only those), with the flags of the whole clip at those
    rows.
    """
    calls: list[tuple[np.ndarray, object, np.ndarray]] = []
    original = processor_module.is_gray_batch

    def spy(frames, device=None):
        flags = original(frames, device=device)
        calls.append((np.asarray(frames), device, np.asarray(flags)))
        return flags

    monkeypatch.setattr(processor_module, "is_gray_batch", spy)

    def fake_batched_paste(self, *args, **kwargs):
        """The twin takes rows 1 and 3 - the other rows fall back."""
        return {
            1: np.zeros((64, 96, 3), dtype=np.uint8),
            3: np.zeros((64, 96, 3), dtype=np.uint8),
        }

    monkeypatch.setattr(
        processor_module.KEEPFaceProcessor, "_batched_paste", fake_batched_paste)

    frames = _gray_and_colour_frames()
    helper = _RecordingGrayFaceHelper(
        (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0]))
    _run_sequence_with(
        processor_module, torch.device("cpu"), helper, frames, max_clip_length=2)

    whole = image_utils.comfy_batch_to_bgr_u8(frames)
    whole_flags = gpu_paste.is_gray_batch(whole)
    fallback_rows = [0, 2]

    assert calls, "the processor did not call is_gray_batch"
    assert calls[0][1] is None, [call[1] for call in calls]
    # the gray pass saw the fallback rows only - the eligible rows' flags
    # are never computed
    assert [len(group) for group, _, _ in calls] == [2], calls
    assert sum(len(group) for group, _, _ in calls) == len(fallback_rows), calls
    # the flags are the whole-clip flags at exactly the fallback rows
    captured = np.concatenate([flags for _, _, flags in calls])
    assert list(captured) == list(whole_flags[fallback_rows]), (
        list(captured), list(whole_flags[fallback_rows])
    )
    # the flags the fallback paste reads are the same values
    assert helper.gray_flags_read == list(whole_flags[fallback_rows]), (
        helper.gray_flags_read
    )
