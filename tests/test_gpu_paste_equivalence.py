"""The measured old-vs-new equivalence of the batched paste-back.

The video pipeline redesign (design section 7.7 of
``.dev/docs/2026-10-25-keep-gpu-video-pipeline.md``) replaces the per-frame
vendored paste (``face_restoration_helper.py:355-528``) with a batched
paste on the device: the chunked ParseNet forward, the batched parse mask,
the batched inverse warp and the batched blend.

Every op is pinned here against the vendored reference it replaces, with
the tolerance the runs measured - measured, not assumed (the pattern of
``tests/test_image_op_equivalence.py`` and of t1's
``tests/test_gpu_batch_equivalence.py``).

Bound provenance:

- the parse mask is t1's batched postprocess (the colormap assignment, the
  two blurs and the border zeroing) - the drift against the vendored chain
  is the float summation order of the two blurs;
- the inverse warp of the faces replaces ``cv2.warpAffine`` on a uint8
  image: cv2 samples in 1/32 px fixed point, torch in float32 - the
  residual is bounded by ``local gradient / 64`` plus the quantization of
  the uint8 grid;
- the parse mask warp replaces the float32 ``cv2.warpAffine`` - cv2
  quantizes the affine coefficients to 1/1024, so the position error grows
  with the output coordinate;
- the blend and its quantizing tail are the elementwise chain of
  ``:495-507`` - bit-exact against the vendored arithmetic, the residual
  of the paste comes from the two warps above.
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


@pytest.fixture(scope="module")
def gpu_paste():
    return importlib.import_module("ComfyUI_KEEP.src.keep_gpu_paste")


@pytest.fixture(scope="module")
def gpu_ops():
    return importlib.import_module("ComfyUI_KEEP.src.keep_gpu_ops")


@pytest.fixture(scope="module")
def helper_module():
    return importlib.import_module("ComfyUI_KEEP.modules.deps.facelib.utils.face_restoration_helper")


@pytest.fixture(scope="module")
def misc_module():
    return importlib.import_module("ComfyUI_KEEP.modules.deps.facelib.utils.misc")


@pytest.fixture()
def face_helper(helper_module, monkeypatch):
    """A vendored ``FaceRestoreHelper`` without its weight-backed models.

    The detection / parsing initializations are stubbed (the weights are not
    on this machine), so the reference paste runs the real vendored code.
    """
    monkeypatch.setattr(helper_module, "init_detection_model", lambda *a, **k: object())
    monkeypatch.setattr(helper_module, "init_parsing_model", lambda *a, **k: object())
    return helper_module.FaceRestoreHelper(
        upscale_factor=1, face_size=512, crop_ratio=(1, 1),
        det_model="retinaface_resnet50", save_ext="png", use_parse=False,
        device="cpu", model_rootpath="unused",
    )


class _StubParseNet:
    """A parse net with the vendored contract (``[0]`` are the logits).

    A 1x1 convolution over the color channels: the class map is a function
    of the pixels alone, so the forward is batch invariant by construction -
    the chunk size of the paste stays a pure performance knob.
    """

    def __init__(self, seed: int = 5) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(19, 3, 1, 1, generator=generator) * 0.35

    def __call__(self, samples):
        return (F.conv2d(samples, self.weight), None)


def synthetic_landmarks(num_frames: int, frame_hw: tuple[int, int], seed: int) -> np.ndarray:
    """One face per frame: the 512 template at a jittered in-frame position."""
    rng = np.random.default_rng(seed)
    height, width = frame_hw
    base_scale = min(height, width) / 512.0 * 0.72
    landmarks = np.zeros((num_frames, 5, 2), dtype=np.float64)
    for i in range(num_frames):
        scale = base_scale + rng.uniform(-0.005, 0.005)
        center = np.array(
            [width * 0.5 + rng.uniform(-8, 8), height * 0.46 + rng.uniform(-6, 6)]
        )
        landmarks[i] = (FACE_TEMPLATE - 256.0) * scale + center
    return landmarks


def smooth_frames_u8(num_frames: int, frame_hw: tuple[int, int], seed: int) -> np.ndarray:
    """Low-frequency content: the realistic case for the warp residual."""
    rng = np.random.default_rng(seed)
    height, width = frame_hw
    small = rng.uniform(0.0, 255.0, size=(num_frames, 12, 22, 3)).astype(np.float32)
    frames = np.stack(
        [cv2.resize(small[i], (width, height), interpolation=cv2.INTER_CUBIC) for i in range(num_frames)]
    )
    return np.clip(frames, 0, 255).astype(np.uint8)


def noise_frames_u8(num_frames: int, frame_hw: tuple[int, int], seed: int) -> np.ndarray:
    """Per-pixel noise: the adversarial case for the fixed-point sampling."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(num_frames, *frame_hw, 3), dtype=np.uint8)


def _vendored_paste(
    helper,
    frame_bgr: np.ndarray,
    face_bgr: np.ndarray,
    affine: np.ndarray,
    upscale_factor: float,
    parse_net,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """The vendored paste of one frame (``face_restoration_helper.py:355-528``)."""
    helper.clean_all()
    helper.use_parse = True
    helper.face_parse = parse_net
    helper.input_img = frame_bgr
    helper.is_gray = False
    helper.restored_faces = [face_bgr]
    helper.affine_matrices = [affine]
    helper.upscale_factor = upscale_factor
    helper.get_inverse_affine(None)
    height, width = frame_bgr.shape[:2]
    target_h, target_w = int(height * upscale_factor), int(width * upscale_factor)
    if background is None:
        background = frame_bgr
    if background.shape[0] != target_h or background.shape[1] != target_w:
        background = cv2.resize(
            background, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
        )
    return helper.paste_faces_to_input_image(
        upsample_img=background, draw_box=False, face_upsampler=None
    )


# ---------------------------------------------------------------------------
# The batched parse mask (design section 7.7, steps 3-4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_batched_parse_masks_match_the_vendored_parse_chain(gpu_paste):
    """The chunked parse mask vs the vendored chain of ``:452-472``.

    The reference is the vendored input chain (``cv2.resize`` to 512 - an
    identity for the production face size -, the ``img2tensor`` BGR -> RGB
    flip, the float32 ``/255`` and the ``(t - 0.5) / 0.5`` normalize), the
    ``argmax``, the colormap assignment, the two Gaussian blurs, the border
    zeroing and the ``/255``.
    """
    rng = np.random.default_rng(3)
    faces = rng.integers(0, 256, size=(3, 512, 512, 3), dtype=np.uint8)
    parse_net = _StubParseNet(seed=7)

    faces_t = torch.from_numpy(faces).permute(0, 3, 1, 2)
    masks = gpu_paste.parse_soft_masks_batched(
        parse_net, faces_t, torch.device("cpu"), chunk_size=2
    ).numpy()[:, 0]

    reference = []
    for i in range(len(faces)):
        face_input = faces[i].astype("float32") / 255.0
        tensor = torch.from_numpy(
            cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        ).float()
        tensor = tensor.sub_(0.5).div_(0.5)
        with torch.no_grad():
            out_parse = parse_net(tensor.unsqueeze(0))[0].argmax(dim=1).squeeze().numpy()
        raw = np.zeros(out_parse.shape, dtype=np.float32)
        colormap = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                    255, 255, 255, 255, 0, 255, 0, 0, 0]
        for map_idx, color_val in enumerate(colormap):
            raw[out_parse == map_idx] = color_val
        blurred = cv2.GaussianBlur(raw, (101, 101), 11)
        blurred = cv2.GaussianBlur(blurred, (101, 101), 11)
        blurred[:10, :] = 0
        blurred[-10:, :] = 0
        blurred[:, :10] = 0
        blurred[:, -10:] = 0
        reference.append(blurred / 255.0)
    reference = np.stack(reference)

    assert masks.shape == reference.shape
    diff = np.abs(reference - masks).max()
    assert diff <= 1.0 / 255.0, f"the parse masks drifted {diff:.2e} from the vendored chain"


@pytest.mark.unit
def test_the_parse_forward_is_batch_invariant(gpu_paste):
    """The chunk size of the parse forward is a pure performance knob."""
    rng = np.random.default_rng(11)
    faces = rng.integers(0, 256, size=(5, 512, 512, 3), dtype=np.uint8)
    parse_net = _StubParseNet(seed=13)
    faces_t = torch.from_numpy(faces).permute(0, 3, 1, 2)

    per_frame = gpu_paste.parse_soft_masks_batched(
        parse_net, faces_t, torch.device("cpu"), chunk_size=1
    )
    chunked = gpu_paste.parse_soft_masks_batched(
        parse_net, faces_t, torch.device("cpu"), chunk_size=4
    )
    whole = gpu_paste.parse_soft_masks_batched(
        parse_net, faces_t, torch.device("cpu"), chunk_size=5
    )

    assert torch.equal(per_frame, chunked)
    assert torch.equal(per_frame, whole)


# ---------------------------------------------------------------------------
# The batched paste (design section 7.7, steps 2-5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_batched_paste_matches_the_vendored_paste(gpu_paste, gpu_ops, face_helper):
    """The batched paste vs the vendored paste: smooth content.

    The reference is the real vendored ``paste_faces_to_input_image`` with
    the same parse net, the same affine and the same background.  The
    residual comes from the two warps (cv2's 1/32 px fixed point against
    torch's float32 sampling) and from the float summation order of the
    two parse blurs.

    Measured on this content: **4/255** at worst, 13% of the pixels differ
    at all.
    """
    frames = smooth_frames_u8(3, (360, 640), seed=17)
    landmarks = synthetic_landmarks(3, (360, 640), seed=18)
    rng = np.random.default_rng(19)
    faces = rng.integers(0, 256, size=(3, 512, 512, 3), dtype=np.uint8)
    parse_net = _StubParseNet(seed=23)
    affines = gpu_ops.estimate_affines(landmarks, face_helper.face_template)

    device = torch.device("cpu")
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)
    pasted = gpu_paste.paste_faces_batch(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=frames,
        device=device,
        parse_net=parse_net,
    )

    assert len(pasted) == 3
    for i in range(3):
        reference = _vendored_paste(
            face_helper, frames[i], faces[i], affines[i], 1.0, parse_net
        )
        diff = np.abs(reference.astype(int) - pasted[i].astype(int))
        assert diff.max() <= 4, (
            f"frame {i}: the batched paste drifted {diff.max()}/255 from the vendored paste"
        )


@pytest.mark.unit
def test_the_batched_paste_matches_the_vendored_paste_on_noise_content(
    gpu_paste, gpu_ops, face_helper
):
    """The batched paste vs the vendored paste: adversarial content.

    Per-pixel noise is the worst case for cv2's fixed-point sampling (the
    gradients reach 255 per pixel); the bound is the sum of the two warp
    residuals (the face warp on noise, the mask warp on the blurred mask).

    Measured: **8/255** at worst, 16% of the pixels differ at all - the
    same bound t1 measured for the inverse warp alone.
    """
    frames = noise_frames_u8(3, (360, 640), seed=29)
    landmarks = synthetic_landmarks(3, (360, 640), seed=30)
    rng = np.random.default_rng(31)
    faces = rng.integers(0, 256, size=(3, 512, 512, 3), dtype=np.uint8)
    parse_net = _StubParseNet(seed=37)
    affines = gpu_ops.estimate_affines(landmarks, face_helper.face_template)

    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)
    pasted = gpu_paste.paste_faces_batch(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=frames,
        device=torch.device("cpu"),
        parse_net=parse_net,
    )

    for i in range(3):
        reference = _vendored_paste(
            face_helper, frames[i], faces[i], affines[i], 1.0, parse_net
        )
        diff = np.abs(reference.astype(int) - pasted[i].astype(int))
        assert diff.max() <= 8, (
            f"frame {i}: the batched paste drifted {diff.max()}/255 on noise"
        )


@pytest.mark.unit
def test_the_batched_paste_matches_the_vendored_paste_at_upscale_two(
    gpu_paste, gpu_ops, face_helper
):
    """The paste at an upscale factor of two.

    The inverse warp of the paste is scaled by the upscale factor
    (``get_inverse_affine``), so the canvas is the double one and the
    background is the LANCZOS4 resize of the frame - the resize stays on
    cv2 (there is no torch lanczos).
    """
    frames = smooth_frames_u8(2, (240, 320), seed=41)
    landmarks = synthetic_landmarks(2, (240, 320), seed=42)
    rng = np.random.default_rng(43)
    faces = rng.integers(0, 256, size=(2, 512, 512, 3), dtype=np.uint8)
    parse_net = _StubParseNet(seed=47)
    affines = gpu_ops.estimate_affines(landmarks, face_helper.face_template)

    background = gpu_paste.batched_bg_frames(
        frames, 2.0, None, torch.device("cpu")
    )
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=2.0)
    pasted = gpu_paste.paste_faces_batch(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
    )

    assert background.shape == (2, 480, 640, 3), background.shape
    for i in range(2):
        reference = _vendored_paste(
            face_helper, frames[i], faces[i], affines[i], 2.0, parse_net,
            background=background[i],
        )
        diff = np.abs(reference.astype(int) - pasted[i].astype(int))
        assert diff.max() <= 5, (
            f"frame {i}: the upscaled batched paste drifted {diff.max()}/255"
        )


@pytest.mark.unit
def test_the_batched_paste_rejects_a_wrong_face_size(gpu_paste):
    """A face that is not of the ParseNet size fails loudly.

    The vendored paste resizes the face to the ParseNet size and the parse
    mask back to the face size; the batched path keeps the production size
    and leaves the other sizes to the vendored paste.
    """
    with pytest.raises(ValueError):
        gpu_paste.paste_faces_batch(
            faces_bgr=np.zeros((1, 256, 256, 3), dtype=np.uint8),
            inverse_affines=[np.eye(2, 3)],
            background_bgr=np.zeros((1, 64, 64, 3), dtype=np.uint8),
            device=torch.device("cpu"),
            parse_net=_StubParseNet(),
        )


# ---------------------------------------------------------------------------
# The batched background (design section 7.7, step 1)
# ---------------------------------------------------------------------------


class _StubUpscaleModel:
    """A background upscaler with the ComfyUI model contract."""

    scale = 2.0

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def model(self, samples):
        self.calls.append(tuple(samples.shape))
        return samples * 2.0


@pytest.mark.unit
def test_the_batched_background_is_pixel_identical_to_the_per_frame_calls(
    gpu_paste, processor_module
):
    """One ``tiled_scale`` call over the batch == N per-frame calls.

    ``comfy.utils.tiled_scale`` loops over the images of the batch
    (``comfy/utils.py:1205-1206``) and processes every image on its own,
    so the batched call produces the same pixels - with one call instead
    of N.  The reference is the **per-frame path of the processor**
    (``_run_upscaler`` + the LANCZOS4 resize to the target size).
    """
    frames = smooth_frames_u8(3, (64, 96), seed=53)
    model = _StubUpscaleModel()
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _ParseFaceHelper(), bg_upscale_model=model)
    )

    # the upscale factor of 2 makes the target size the double one
    batched = gpu_paste.batched_bg_frames(frames, 2.0, model, torch.device("cpu"))

    reference = []
    for frame in frames:
        upscaled = processor._run_upscaler(model, frame)
        reference.append(
            cv2.resize(upscaled, (192, 128), interpolation=cv2.INTER_LANCZOS4)
        )

    assert batched.shape == (3, 128, 192, 3), batched.shape
    assert np.array_equal(batched, np.stack(reference))


# ---------------------------------------------------------------------------
# The gray flags
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_gray_batch_agrees_with_the_vendored_is_gray(gpu_paste, misc_module):
    """The batched gray flags are the flags of the vendored ``is_gray``.

    The vendored call measures the variance of the three channel
    differences through PIL; this form measures the same variances with
    numpy, so the flags agree.
    """
    rng = np.random.default_rng(59)
    color = rng.integers(0, 256, size=(4, 32, 48, 3), dtype=np.uint8)
    gray = np.repeat(rng.integers(0, 256, size=(4, 32, 48, 1), dtype=np.uint8), 3, axis=3)
    near_gray = color.copy()
    near_gray[..., 1] = near_gray[..., 0]
    frames = np.concatenate([color, gray, near_gray])

    flags = gpu_paste.is_gray_batch(frames)
    reference = [misc_module.is_gray(frames[i], threshold=10) for i in range(len(frames))]

    assert list(flags) == reference, (list(flags), reference)
    assert not flags[0]
    assert flags[len(color)]


# ---------------------------------------------------------------------------
# The VRAM-aware chunk sizes (design section 7.7, step 3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_parse_chunk_size_follows_the_free_memory(gpu_paste, monkeypatch):
    """The chunk cap is the free VRAM, bounded by the measured sweet spot."""
    device = torch.device("cpu")
    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 0)
    assert gpu_paste.parse_chunk_size(device) == gpu_paste.PARSE_CHUNK_MAX

    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 64 << 30)
    assert gpu_paste.parse_chunk_size(device) == gpu_paste.PARSE_CHUNK_MAX

    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 2 << 30)
    assert gpu_paste.parse_chunk_size(device) == 1

    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 6 << 30)
    assert gpu_paste.parse_chunk_size(device) == 3


@pytest.mark.unit
def test_the_paste_chunk_size_follows_the_free_memory(gpu_paste, monkeypatch):
    """The staging of the paste is bounded by the free VRAM."""
    device = torch.device("cpu")
    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 0)
    assert gpu_paste.paste_chunk_size(720, 1280, device) == gpu_paste.PASTE_CHUNK_MAX

    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 64 << 30)
    assert gpu_paste.paste_chunk_size(720, 1280, device) == gpu_paste.PASTE_CHUNK_MAX

    # 1 MiB of free memory - the 50% share leaves 512 KiB of staging, which
    # fits 4 frames of a 64x64 canvas (31 bytes per pixel)
    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 1 << 20)
    assert gpu_paste.paste_chunk_size(64, 64, device) == 4


# ---------------------------------------------------------------------------
# The processor: the batched paste and the vendored fallbacks
# ---------------------------------------------------------------------------


class _StubKeepNet:
    """A KEEP net that returns its input (the restored face is the crop)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, frames, need_upscale=False):
        self.calls.append((tuple(frames.shape), need_upscale))
        return frames


class _ParseFaceHelper:
    """A face helper stub with a parse net (the batched paste branch)."""

    def __init__(self, parse_net=None, landmarks=None) -> None:
        self.device = torch.device("cpu")
        self.upscale_factor = 1.0
        self.is_gray = False
        self.cropped_faces: list = []
        self.restored_faces: list = []
        self.all_landmarks_5: list = []
        self.affine_matrices: list = []
        self.inverse_affine_matrices: list = []
        self.input_img = None
        self.face_template = FACE_TEMPLATE
        self.calls: list[str] = []
        self.face_parse = parse_net
        self.use_parse = parse_net is not None
        self._landmarks = landmarks

    def clean_all(self):
        self.calls.append("clean_all")
        self.cropped_faces = []
        self.restored_faces = []

    def read_image(self, image):
        self.calls.append("read_image")
        self.input_img = image

    def get_face_landmarks_5(self, **kwargs):
        self.calls.append("get_face_landmarks_5")
        self.all_landmarks_5 = [self._landmarks]
        return 1

    def add_restored_face(self, face, **kwargs):
        self.calls.append("add_restored_face")
        self.restored_faces = [face]

    def get_inverse_affine(self, *args, **kwargs):
        self.calls.append("get_inverse_affine")

    def paste_faces_to_input_image(self, *args, **kwargs):
        self.calls.append("paste_faces_to_input_image")
        return np.zeros((64, 96, 3), dtype=np.uint8)


def _make_pack(keep_net, face_helper, bg_upscale_model=None):
    return type(
        "Pack",
        (),
        {
            "keep_net": keep_net,
            "face_helper": face_helper,
            "bg_upscale_model": bg_upscale_model,
            "face_upscale_model": None,
            "device": torch.device("cpu"),
            "model_type_str": "KEEP",
        },
    )()


def _run_sequence(processor_module, face_helper, frames, **kwargs):
    keep_net = _StubKeepNet()
    processor = processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=kwargs.pop("final_upscale_factor", 1.0),
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=kwargs.pop("draw_box", False),
        max_clip_length=kwargs.pop("max_clip_length", 20),
    )
    return output, keep_net


@pytest.fixture(scope="module")
def processor_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_processor")


@pytest.mark.unit
def test_the_default_branch_of_the_paste_runs_on_the_device(processor_module):
    """The default paste is the batched one - the vendored paste is not called."""
    landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _ParseFaceHelper(parse_net=_StubParseNet(), landmarks=landmarks)
    frames = torch.rand((2, 64, 96, 3), dtype=torch.float32)

    output, keep_net = _run_sequence(processor_module, face_helper, frames)

    assert "paste_faces_to_input_image" not in face_helper.calls, face_helper.calls
    assert tuple(output.shape) == (2, 64, 96, 3), output.shape
    # the KEEP chunk boundaries did not move (the paste stages after them)
    assert [shape for shape, _ in keep_net.calls] == [(1, 2, 3, 512, 512)], keep_net.calls


@pytest.mark.unit
def test_the_vendored_paste_stays_reachable_for_the_box(processor_module):
    """``draw_box=True`` keeps the vendored paste (the box needs its masks)."""
    landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _ParseFaceHelper(parse_net=_StubParseNet(), landmarks=landmarks)
    frames = torch.rand((1, 64, 96, 3), dtype=torch.float32)

    output, _ = _run_sequence(processor_module, face_helper, frames, draw_box=True)

    assert "paste_faces_to_input_image" in face_helper.calls, face_helper.calls
    assert tuple(output.shape) == (1, 64, 96, 3), output.shape


@pytest.mark.unit
def test_the_vendored_paste_stays_reachable_for_the_grayscale_frames(processor_module):
    """A grayscale frame keeps the vendored paste (the gray face conversion)."""
    landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _ParseFaceHelper(parse_net=_StubParseNet(), landmarks=landmarks)
    gray = torch.full((1, 64, 96, 3), 0.4, dtype=torch.float32)
    gray[..., 1] = 0.5

    output, _ = _run_sequence(processor_module, face_helper, gray)

    assert "paste_faces_to_input_image" in face_helper.calls, face_helper.calls
    assert tuple(output.shape) == (1, 64, 96, 3), output.shape


@pytest.mark.unit
def test_the_vendored_paste_stays_reachable_without_a_parse_net(processor_module):
    """A face helper without a parse net keeps the vendored paste."""
    landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _ParseFaceHelper(parse_net=None, landmarks=landmarks)
    frames = torch.rand((1, 64, 96, 3), dtype=torch.float32)

    output, _ = _run_sequence(processor_module, face_helper, frames)

    assert "paste_faces_to_input_image" in face_helper.calls, face_helper.calls
    assert tuple(output.shape) == (1, 64, 96, 3), output.shape
