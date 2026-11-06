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

The **ROI paste** (the paste-stage CPU optimization round) is pinned here
as well: the ROI face warp (cv2's fixed-point bilinear, re-implemented in
int32 on the device), the ROI mask warp (the full-canvas grid, sliced),
the ROI blend, the support of the warped parse mask, the transfer
accounting and the frame-global maximum of the quantizing tail.  Those
pins were **bit-exact** - the ROI path had to reproduce the crop of the
full-canvas result byte for byte.  Fix 2.1 (2026-10-30, the owner's
``keep-fix3.txt``) vectorized the shipped single-face chunk
(``_paste_chunk_roi``): the full-canvas ``grid_sample`` face warp
replaced the cv2 fixed-point warp + ROI slicing, so the whole-paste pins
against the unchanged full-canvas reference hold the **measured
residual** now (the 8/255 drift class, documented per pin).  The owner's
fix 4 (2026-10-31) went further: both ROI chunks warp each face straight
into its canvas ROI (``_compute_roi_theta`` + ``F.affine_grid`` /
``F.grid_sample``) and quantize with the plain ``round_().clamp_(0, 255)``
tail - the near-black 255-factor scaling is gone from the ROI paths
(owner-authorized pixel change), so the dark-frame pin holds the
divergence instead of a residual.  The warp-op pins (``warp_face_roi``
and friends) stay bit-exact - the ops outlived their production call
site (the fix-4 chunks stopped calling them) and are pinned as the
reference implementations they are.

The **device-output assembly** (the GPU-resident-output round) is pinned
here as well: the VRAM budget of the device canvas, the three-way mode
selector, the sub-chunked convert-out, the factor-4 device canvas and the
chunked device assembly.  The pins are **bit-exact** and run on the CPU
device only: every op of the paste is batch-invariant there (measured -
the parse forward, the warps, the blend and the elementwise convert
chain), so the chunked mode equals the full mode equals the host stack
for any chunk boundary.
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


@pytest.fixture(scope="module")
def image_utils():
    return importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")


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


@pytest.mark.unit
def test_the_face_upscaler_rides_the_batched_core(gpu_paste):
    """The face upscaler (keep-fix5, 2026-11-03): B=1, per-face, bit-exact.

    The pin of :func:`upscale_faces_batch_gpu` against the per-face
    reference re-derived below - the BGR -> RGB flip, the /255, the model
    call, the clamp, the round quantize and the BGR flip back.  The
    per-face loop of the production form means the batched result is the
    per-face one, byte for byte; a ``None`` model returns the input
    object.
    """
    class _SpatialUpscaleModule(torch.nn.Module):
        """The stub's inner ``.model`` - the same nearest x2 as a module.

        keep-fix5-implement2 (2026-11-03): the re-entrant
        ``_spandrel_on_device`` probes ``next(m.model.parameters()).device``,
        so the inner model is a real ``nn.Module`` (a bound method has no
        ``parameters()``); the forward is the nearest x2 the former method
        body ran - the pin's math is unchanged.
        """

        def __init__(self, owner) -> None:
            super().__init__()
            self.owner = owner
            self.mark = torch.nn.Parameter(torch.zeros(1))

        def forward(self, samples):
            return F.interpolate(
                samples, scale_factor=self.owner.scale, mode="nearest"
            )

    class _SpatialUpscaleStub:
        """A face upscaler whose output really is ``scale`` x the input.

        The descriptor form (keep-fix5-implement2, 2026-11-03): the
        ``.to`` the lifecycle manager calls moves the inner module, like
        spandrel's own descriptor ``.to`` does.
        """

        scale = 2.0

        def __init__(self) -> None:
            self.model = _SpatialUpscaleModule(self)

        def to(self, device, *args, **kwargs):
            self.model.to(device, *args, **kwargs)
            return self

    rng = np.random.default_rng(17)
    faces = torch.from_numpy(
        rng.integers(0, 256, size=(3, 8, 12, 3), dtype=np.uint8)
    ).permute(0, 3, 1, 2)
    model = _SpatialUpscaleStub()

    upscaled = gpu_paste.upscale_faces_batch_gpu(faces, model, torch.device("cpu"))
    assert tuple(upscaled.shape) == (3, 3, 16, 24), upscaled.shape

    for index in range(3):
        face_rgb = faces[index : index + 1, [2, 1, 0]].to(torch.float32).div_(255.0)
        scaled = model.model(face_rgb)
        reference = (
            scaled.float()[:, [2, 1, 0]]
            .clamp_(0.0, 1.0)
            .mul_(255.0)
            .round_()
            .to(torch.uint8)
        )
        assert torch.equal(upscaled[index : index + 1], reference)

    assert gpu_paste.upscale_faces_batch_gpu(faces, None, torch.device("cpu")) is faces


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
# The ROI paste (the paste-stage CPU optimization round)
#
# The ROI paste replaces the full-canvas paste of the same pipeline: the
# face warp, the mask warp and the blend run on the **ROI** of the
# transformed face and only the ROI patch travels between the host and the
# device.  The pins below are the acceptance criteria of the round:
#
# 1. the ROI face warp is bit-exact against ``cv2.warpAffine`` (the smooth
#    and the noise corpus, at an interior and at an edge-touching ROI),
# 2. the ROI mask warp is bit-exact against the crop of the full-canvas
#    mask warp,
# 3. the ROI blend is bit-exact against the crop of the full-canvas blend,
# 4. the support of the warped parse mask stays inside the ROI (no seam),
# 5. the transfer counters are ROI-sized, not canvas-sized,
# 6. the frame-global maximum branch of the quantizing tail: a near-black
#    frame falls back to the full-canvas path, and a near-black ROI inside
#    a bright frame does not.
# ---------------------------------------------------------------------------

#: The canvas of the ROI pins - big enough for a face and small enough for
#: a fast test.
ROI_CANVAS_HW = (360, 640)


def _face_u8(corpus: str, seed: int) -> np.ndarray:
    """One 512x512 BGR face of a corpus (the smooth and the noise one).

    The face is the input of the paste's face warp; the corpus is the
    smooth + noise pair the other pins of this file use - the noise face is
    the adversarial case for the fixed-point sampling (its gradients reach
    255 per pixel).
    """
    rng = np.random.default_rng(seed)
    if corpus == "noise":
        return rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    small = rng.uniform(0.0, 255.0, size=(32, 32, 3)).astype(np.float32)
    smooth = cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)
    return np.clip(smooth, 0, 255).astype(np.uint8)


def _face_to_canvas(
    canvas_hw: tuple[int, int], position: str = "interior", scale: float = 0.25
) -> np.ndarray:
    """The face -> canvas matrix of the paste, at a controlled ROI position.

    The paste warps the face onto the canvas with the matrix of
    ``get_inverse_affine`` (``cv2.invertAffineTransform(affine) *
    upscale_factor``) - the map is the **forward** one (face -> canvas),
    because ``cv2.warpAffine`` inverts it internally.

    The face is scaled down (a quarter of the face size) and rotated,
    so the ROI is a small part of the canvas - the transfer claim needs
    one - and the fixed-point sampling has a non-trivial fraction.
    ``interior`` puts the face in the middle of the canvas, ``edge`` hangs
    it over the top-left corner, so the ROI is clipped by the canvas.
    """
    height, width = canvas_hw
    angle = np.deg2rad(7.0)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    ) * scale
    if position == "interior":
        center = np.array([width * 0.55, height * 0.45], dtype=np.float64)
    else:
        center = np.array([26.0, 20.0], dtype=np.float64)
    matrix[:, 2] = center - matrix[:, :2] @ np.array([256.0, 256.0])
    return matrix


def _soft_masks(gpu_paste, faces: np.ndarray, parse_net) -> torch.Tensor:
    """The soft parse masks of the faces (the product's own chain)."""
    faces_t = torch.from_numpy(np.ascontiguousarray(faces)).permute(0, 3, 1, 2)
    return gpu_paste.parse_soft_masks_batched(
        parse_net, faces_t, torch.device("cpu"), chunk_size=2
    )


@pytest.mark.unit
@pytest.mark.parametrize("corpus", ["smooth", "noise"])
@pytest.mark.parametrize("position", ["interior", "edge"])
def test_the_roi_face_warp_is_bit_exact_against_cv2(
    gpu_paste, corpus, position
):
    """The device ROI face warp == the crop of ``cv2.warpAffine``.

    The ROI warp is cv2's own bilinear, re-implemented in int32 on the
    device (the 1/32 sampling grid, the four taps, the ``(+512) >> 10``
    rounding and the ``BORDER_CONSTANT`` rule).  The pin is **bit-exact**:
    the device warp and the ROI crop of the full-canvas ``cv2.warpAffine``
    are the same bytes.

    The two corpora are the two sides of the fixed-point sampling (a
    smooth face hides a wrong rounding, a noise face does not) and the two
    positions are the two sides of the ROI (an interior ROI and one the
    canvas clips, where the border taps fall outside the source).
    """
    height, width = ROI_CANVAS_HW
    face = _face_u8(corpus, seed=61 + len(corpus))
    matrix = _face_to_canvas(ROI_CANVAS_HW, position)
    roi = gpu_paste.face_roi_box(matrix, ROI_CANVAS_HW)
    terms = gpu_paste.warp_fixed_point_terms(matrix, width, height)

    warped = gpu_paste.warp_face_roi(
        torch.from_numpy(face).permute(2, 0, 1).contiguous(), terms, roi
    ).numpy()
    reference = cv2.warpAffine(face, matrix, (width, height))
    x0, y0, x1, y1 = roi

    assert np.array_equal(warped, reference[y0:y1, x0:x1].transpose(2, 0, 1)), (
        f"the ROI face warp drifted from cv2 on the {corpus} corpus "
        f"at the {position} ROI"
    )


@pytest.mark.unit
def test_the_roi_face_warp_uploads_the_terms_once(gpu_paste, monkeypatch):
    """The four term uploads happen once, at the top of the body.

    ``warp_face_roi`` needs exactly four host arrays per call - the ROI
    slices of the two column and two row terms - uploaded before the
    sampling arithmetic starts.  The pin freezes that hoisted shape
    against a regression to per-tap transfers: a counting
    ``torch.from_numpy`` wrapper must see exactly four calls per
    invocation, and each of them one of the four term slices.
    """
    height, width = ROI_CANVAS_HW
    face = _face_u8("noise", seed=103)
    matrix = _face_to_canvas(ROI_CANVAS_HW, "edge")
    roi = gpu_paste.face_roi_box(matrix, ROI_CANVAS_HW)
    terms = gpu_paste.warp_fixed_point_terms(matrix, width, height)
    source = torch.from_numpy(face).permute(2, 0, 1).contiguous()

    uploaded: list[np.ndarray] = []
    real_from_numpy = torch.from_numpy

    def counting_from_numpy(array):
        uploaded.append(np.asarray(array))
        return real_from_numpy(array)

    monkeypatch.setattr(torch, "from_numpy", counting_from_numpy)
    gpu_paste.warp_face_roi(source, terms, roi)

    x0, y0, x1, y1 = roi
    expected = [
        terms[0][x0:x1].astype(np.int32),
        terms[1][x0:x1].astype(np.int32),
        terms[2][y0:y1].astype(np.int32),
        terms[3][y0:y1].astype(np.int32),
    ]
    assert len(uploaded) == 4, f"{len(uploaded)} term uploads for one ROI warp"
    remaining = list(expected)
    for piece in uploaded:
        match = next(
            (
                index
                for index, want in enumerate(remaining)
                if np.array_equal(piece, want)
            ),
            None,
        )
        assert match is not None, "an upload that is not one of the term slices"
        remaining.pop(match)
    assert not remaining, "a term slice was never uploaded"


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device for the GPU pin"
)
def test_the_roi_face_warp_is_bit_exact_on_the_gpu(gpu_paste):
    """The ROI face warp on the GPU == the CPU warp == cv2's crop.

    The emulation is pure int32 (the 1/32 grid, the four taps, the
    ``(+512) >> 10`` rounding), so it is **device independent** - and the
    production path runs it on the GPU.  The pin asserts that the two
    devices and cv2 agree byte for byte on the adversarial corpus.
    """
    height, width = ROI_CANVAS_HW
    face = _face_u8("noise", seed=67)
    matrix = _face_to_canvas(ROI_CANVAS_HW, "interior")
    roi = gpu_paste.face_roi_box(matrix, ROI_CANVAS_HW)
    terms = gpu_paste.warp_fixed_point_terms(matrix, width, height)
    source = torch.from_numpy(face).permute(2, 0, 1).contiguous()

    on_gpu = gpu_paste.warp_face_roi(source.cuda(), terms, roi).cpu().numpy()
    on_cpu = gpu_paste.warp_face_roi(source, terms, roi).numpy()
    x0, y0, x1, y1 = roi
    reference = cv2.warpAffine(face, matrix, (width, height))[
        y0:y1, x0:x1
    ].transpose(2, 0, 1)

    assert np.array_equal(on_gpu, on_cpu), "the GPU face warp differs from the CPU one"
    assert np.array_equal(on_gpu, reference), "the GPU face warp differs from cv2"


@pytest.mark.unit
def test_the_roi_mask_warp_is_bit_exact_against_the_full_canvas_warp(
    gpu_paste, gpu_ops
):
    """The ROI mask warp == the crop of the full-canvas mask warp.

    The mask warp is exact by construction: ``grid_sample``'s sampling
    position depends only on the grid value and the source size, so the
    ROI's grid is the full-canvas grid **sliced** - the same taps, the same
    weights, the same zero padding.  The pin is bit-exact on the float32
    mask (there is no quantisation in between).
    """
    height, width = ROI_CANVAS_HW
    faces = np.stack([_face_u8("noise", 71), _face_u8("smooth", 73)])
    matrices = [
        _face_to_canvas(ROI_CANVAS_HW, "interior"),
        _face_to_canvas(ROI_CANVAS_HW, "edge"),
    ]
    masks = _soft_masks(gpu_paste, faces, _StubParseNet(seed=79))
    rois = [gpu_paste.face_roi_box(matrix, ROI_CANVAS_HW) for matrix in matrices]

    full = gpu_ops.warp_affine_batch(masks, matrices, (width, height))
    cropped = gpu_ops.warp_affine_batch(
        masks, matrices, (width, height), rois=rois
    )

    for index, (x0, y0, x1, y1) in enumerate(rois):
        assert torch.equal(cropped[index][0], full[index][:, y0:y1, x0:x1]), (
            f"frame {index}: the ROI mask warp drifted from the full-canvas warp"
        )


@pytest.mark.unit
@pytest.mark.parametrize("corpus", ["smooth", "noise"])
def test_the_roi_blend_is_bit_exact_against_the_full_canvas_blend(
    gpu_paste, gpu_ops, corpus
):
    """The ROI blend == the crop of the full-canvas blend.

    The blend is ``mask * face + (1 - mask) * background`` - an elementwise
    float32 chain, so the ROI form is bit-exact when its three inputs are
    (the face warp pin, the mask warp pin and the untouched background
    patch).
    """
    height, width = ROI_CANVAS_HW
    face = _face_u8(corpus, seed=83)
    matrix = _face_to_canvas(ROI_CANVAS_HW, "interior")
    roi = gpu_paste.face_roi_box(matrix, ROI_CANVAS_HW)
    x0, y0, x1, y1 = roi
    rng = np.random.default_rng(89)
    background = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    masks = _soft_masks(gpu_paste, face[None], _StubParseNet(seed=97))
    terms = gpu_paste.warp_fixed_point_terms(matrix, width, height)

    # the full-canvas reference: cv2's face warp, the device mask warp
    warped_face = torch.from_numpy(
        cv2.warpAffine(face, matrix, (width, height))
    ).permute(2, 0, 1).float()
    warped_mask = gpu_ops.warp_affine_batch(masks, [matrix], (width, height))
    background_t = torch.from_numpy(background).permute(2, 0, 1).float()
    full = warped_mask * warped_face + (1.0 - warped_mask) * background_t

    # the ROI form: the device face warp, the sliced mask, the ROI patch
    roi_face = gpu_paste.warp_face_roi(
        torch.from_numpy(face).permute(2, 0, 1).contiguous(), terms, roi
    ).float()
    roi_mask = gpu_ops.warp_affine_batch(
        masks, [matrix], (width, height), rois=[roi]
    )[0][0]
    patch = torch.from_numpy(
        np.ascontiguousarray(background[y0:y1, x0:x1])
    ).permute(2, 0, 1).float()
    blended = roi_mask * roi_face + (1.0 - roi_mask) * patch

    assert np.array_equal(blended.numpy(), full[0][:, y0:y1, x0:x1].numpy()), (
        f"the ROI blend drifted from the full-canvas blend on the {corpus} corpus"
    )


@pytest.mark.unit
def test_the_warped_parse_mask_stays_inside_the_roi(gpu_paste, gpu_ops):
    """The support of the warped parse mask stays inside the ROI.

    The parse mask is zeroed in a 10 px border of the face, so its support
    is ``[10, 501]²`` and the bilinear can only spread it to ``[9, 502]²`` -
    strictly inside the transformed footprint of the face, hence inside the
    ROI.  The pin asserts it instead of assuming it: every pixel outside
    the ROI is exactly zero (no seam at the ROI's border) and the ROI's
    border ring is zero on the sides the canvas does not clip.
    """
    height, width = ROI_CANVAS_HW
    faces = np.stack([_face_u8("noise", 101), _face_u8("smooth", 103)])
    matrices = [
        _face_to_canvas(ROI_CANVAS_HW, "interior"),
        _face_to_canvas(ROI_CANVAS_HW, "edge"),
    ]
    masks = _soft_masks(gpu_paste, faces, _StubParseNet(seed=107))
    rois = [gpu_paste.face_roi_box(matrix, ROI_CANVAS_HW) for matrix in matrices]
    full = gpu_ops.warp_affine_batch(masks, matrices, (width, height))

    for index, (x0, y0, x1, y1) in enumerate(rois):
        warped = full[index][0]
        outside = warped.clone()
        outside[y0:y1, x0:x1] = 0
        assert outside.abs().max().item() == 0.0, (
            f"frame {index}: the warped parse mask reaches outside the ROI"
        )
        # the ROI's border ring - zero on the sides the canvas does not clip
        ring = []
        if x0 > 0:
            ring.append(warped[y0:y1, x0])
        if x1 < width:
            ring.append(warped[y0:y1, x1 - 1])
        if y0 > 0:
            ring.append(warped[y0, x0:x1])
        if y1 < height:
            ring.append(warped[y1 - 1, x0:x1])
        for pixels in ring:
            assert pixels.abs().max().item() == 0.0, (
                f"frame {index}: the ROI's border ring of the warped mask is not zero"
            )


@pytest.mark.unit
@pytest.mark.parametrize("corpus", ["smooth", "noise"])
@pytest.mark.parametrize("position", ["interior", "edge"])
def test_the_roi_paste_matches_the_full_canvas_paste_within_the_measured_residual(
    gpu_paste, corpus, position
):
    """The vectorized ROI paste vs the full-canvas reference: the measured residual.

    Fix 2.1 (2026-10-30, the owner's ``keep-fix3.txt``) vectorized the ROI
    chunk: the full-canvas ``grid_sample`` face warp replaced the cv2
    fixed-point warp + ROI slicing, so the ROI path is no longer
    byte-identical to the full-canvas reference (``_paste_chunk``,
    unchanged).  The pin holds the measured residual instead: per
    corpus/position, the max diff against the reference - measured
    **4/255** on the noise corpus and **1/255** on the smooth one, the
    un-amplified drift class of the batched warps, pinned at the **8/255**
    class bound of the batched-paste noise pin.

    The two frames of the chunk still have **different** ROIs (a different
    scale), so the interior and the edge-clipped grid positions of the old
    bit-exact pin stay covered.
    """
    height, width = ROI_CANVAS_HW
    rng = np.random.default_rng(109)
    faces = np.stack([_face_u8(corpus, seed=113), _face_u8(corpus, seed=127)])
    matrices = [
        _face_to_canvas(ROI_CANVAS_HW, position),
        _face_to_canvas(ROI_CANVAS_HW, position, scale=0.3),
    ]
    background = rng.integers(
        0, 256, size=(2, height, width, 3), dtype=np.uint8
    )
    parse_net = _StubParseNet(seed=131)
    kwargs = dict(
        faces_bgr=faces,
        inverse_affines=matrices,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        frame_chunk=2,
    )

    roi_paste = gpu_paste.paste_faces_batch(use_roi=True, **kwargs)
    full_paste = gpu_paste.paste_faces_batch(use_roi=False, **kwargs)

    for index, (roi_frame, full_frame) in enumerate(zip(roi_paste, full_paste)):
        diff = np.abs(roi_frame.astype(int) - full_frame.astype(int)).max()
        assert diff <= 8, (
            f"frame {index}: the vectorized ROI paste drifted {diff}/255 from the "
            f"full-canvas reference on the {corpus} corpus at the {position} ROI"
        )


@pytest.mark.unit
def test_the_device_output_stack_is_bit_identical_to_the_host_stack(gpu_paste):
    """F1: the device-output canvas is the host stack, byte for byte.

    The device-output mode (``device_output=True``) moves the assembly of
    the output stack - the per-frame download, the canvas copy, the
    ``np.stack`` and the convert-out's host math - onto the device, where
    the whole stack is one uint8 tensor.  It must not change a pixel: on a
    CPU device the two modes run the same arithmetic on the same data, so
    the canvas equals the list of the host frames exactly, on every pixel,
    for every ROI position of the corpus.

    Fix 2.1 (2026-10-30) retired the ROI staging counters of the chunk
    (``roi_staged_bytes`` counted the ROI patches that no longer travel) -
    the pin asserts the retired zero, the bit-equality itself survives
    (both modes share the vectorized chunk).
    """
    height, width = ROI_CANVAS_HW
    rng = np.random.default_rng(109)
    faces = np.stack([_face_u8("smooth", seed=113), _face_u8("noise", seed=127)])
    matrices = [
        _face_to_canvas(ROI_CANVAS_HW, "interior"),
        _face_to_canvas(ROI_CANVAS_HW, "interior", scale=0.3),
    ]
    background = rng.integers(0, 256, size=(2, height, width, 3), dtype=np.uint8)
    kwargs = dict(
        faces_bgr=faces,
        inverse_affines=matrices,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=_StubParseNet(seed=131),
        frame_chunk=2,
    )

    host_frames = gpu_paste.paste_faces_batch(use_roi=True, **kwargs)
    device_canvas = gpu_paste.paste_faces_batch(use_roi=True, device_output=True, **kwargs)

    assert isinstance(device_canvas, torch.Tensor), type(device_canvas)
    assert tuple(device_canvas.shape) == (2, height, width, 3), device_canvas.shape
    for index, host_frame in enumerate(host_frames):
        assert np.array_equal(device_canvas[index].numpy(), host_frame), (
            f"frame {index}: the device-output canvas drifted from the host stack"
        )
    stats = gpu_paste.PASTE_TRANSFER_STATS
    assert stats["h2d_bytes"] == 0 and stats["d2h_bytes"] == 0, stats
    assert stats["roi_staged_bytes"] == 0, stats
    assert stats["roi_frames"] == 2 and stats["fallback_frames"] == 0, stats


@pytest.mark.unit
def test_the_device_canvas_keeps_the_near_black_full_canvas_fallback(gpu_paste):
    """F1: the near-black branch works in the device-output mode.

    A frame that is near black **everywhere** (not just in its ROI) takes
    the vendored tail's full-canvas fallback, which scales the whole canvas.
    In the device-output mode that fallback writes into the same canvas -
    the pin runs the branch and holds it bit-identical to the host stack,
    so the rare path is covered by construction and not by luck.
    """
    height, width = ROI_CANVAS_HW
    matrix = _face_to_canvas(ROI_CANVAS_HW, "interior")
    rng = np.random.default_rng(179)
    face = rng.integers(0, 2, size=(512, 512, 3), dtype=np.uint8)
    background = np.zeros((1, height, width, 3), dtype=np.uint8)
    background[0, 0, 0] = 200  # bright enough to reach the frame-global test
    kwargs = dict(
        faces_bgr=face[None],
        inverse_affines=[matrix],
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=_StubParseNet(seed=181),
        frame_chunk=1,
    )

    host_frames = gpu_paste.paste_faces_batch(use_roi=True, **kwargs)
    stats = gpu_paste.PASTE_TRANSFER_STATS
    assert stats["fallback_frames"] == 0 or stats["fallback_frames"] == 1, stats
    gpu_paste.reset_paste_transfer_stats()

    canvas = gpu_paste.paste_faces_batch(use_roi=True, device_output=True, **kwargs)
    stats = gpu_paste.PASTE_TRANSFER_STATS
    assert stats["fallback_frames"] in (0, 1), stats
    assert stats["roi_frames"] == 1, stats
    assert np.array_equal(canvas.numpy(), np.stack(host_frames)), (
        "the near-black canvas drifted from the host stack"
    )


@pytest.mark.unit
def test_the_roi_transfers_are_roi_sized(gpu_paste):
    """The transfer counters of the vectorized paste (fix 2.1, 2026-10-30).

    Fix 2.1 vectorized the ROI chunk (the owner's ``keep-fix3.txt``): no
    ROI patch is staged any more, so the old body's counters are retired -
    ``h2d_bytes`` / ``d2h_bytes`` (the ROI patch upload/download),
    ``roi_staged_bytes`` (the device-output staging) and ``canvas_bytes``
    (the canvas size the ROI claim was measured against);
    ``fallback_frames`` counted the near-black full-canvas fallback, whose
    explicit branch the vectorized body deleted (the 255 factor lives in
    the whole-canvas maximum).  The keys stay in ``PASTE_TRANSFER_STATS``
    and report 0; the survivors keep their meaning: ``face_h2d_bytes``
    (the restored faces upload whole) and ``roi_frames`` (the chunk's rows).
    """
    height, width = ROI_CANVAS_HW
    rng = np.random.default_rng(137)
    faces = np.stack([_face_u8("noise", 139), _face_u8("smooth", 149)])
    matrices = [
        _face_to_canvas(ROI_CANVAS_HW, "interior"),
        _face_to_canvas(ROI_CANVAS_HW, "edge"),
    ]
    background = rng.integers(
        0, 256, size=(2, height, width, 3), dtype=np.uint8
    )

    gpu_paste.paste_faces_batch(
        faces_bgr=faces,
        inverse_affines=matrices,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=_StubParseNet(seed=151),
        frame_chunk=2,
    )
    stats = gpu_paste.PASTE_TRANSFER_STATS

    assert stats["frames"] == 2
    assert stats["roi_frames"] == 2
    assert stats["fallback_frames"] == 0
    # the restored faces upload whole - they are not ROI-sized by nature
    assert stats["face_h2d_bytes"] == 2 * 512 * 512 * 3, stats
    # retired by fix 2.1 (2026-10-30): the vectorized chunk stages no ROI
    # patch and takes no explicit near-black fallback branch
    assert stats["canvas_bytes"] == 0, stats
    for key in ("h2d_bytes", "d2h_bytes", "roi_staged_bytes"):
        assert stats[key] == 0, (key, stats)


@pytest.mark.unit
def test_a_dark_frame_keeps_the_whole_canvas_255_factor(gpu_paste, face_helper):
    """A near-black frame stays dark on the batched path (round 10, the owner's fix 4).

    The vendored tail multiplies the **whole canvas** by 255 when the
    blended frame is near black (``face_restoration_helper.py:504-505``).
    The owner's fix 4 (2026-10-31) replaces the ROI chunk's quantizing
    tail with the plain ``round_().clamp_(0, 255)`` - the near-black
    255-factor scaling (:func:`_quantize_and_clip`) is gone from this
    path (owner-authorized pixel change), so a near-black frame no longer
    amplifies: the blend of the 0/1 corpus quantizes back to 0/1 and the
    frame stays dark, while the vendored paste still scales its canvas by
    255.  The pin holds exactly that contrast - the batched frame's
    maximum stays at the raw blend level and the vendored reference
    amplifies - the measured behavior replacing the fix-2.1 residual pin
    (which bound the drift at 133/255 while the factor survived).
    """
    height, width = ROI_CANVAS_HW
    matrix = _face_to_canvas(ROI_CANVAS_HW, "interior")
    rng = np.random.default_rng(157)
    # a near-black frame: the blend stays <= 1, so the vendored tail takes
    # its "values are likely 0-1 float" branch and scales the canvas by 255.
    # The frame is not *all* zero: the vendored paste treats an all-zero
    # input as "no image loaded" (``face_restoration_helper.py:356``).
    face = rng.integers(0, 2, size=(512, 512, 3), dtype=np.uint8)
    background = rng.integers(0, 2, size=(1, height, width, 3), dtype=np.uint8)
    parse_net = _StubParseNet(seed=163)
    kwargs = dict(
        faces_bgr=face[None],
        inverse_affines=[matrix],
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        frame_chunk=1,
    )

    roi_paste = gpu_paste.paste_faces_batch(use_roi=True, **kwargs)
    stats = gpu_paste.PASTE_TRANSFER_STATS
    assert stats["fallback_frames"] == 0, stats
    assert stats["roi_frames"] == 1, stats

    # the plain tail keeps the near-black frame dark: the 0/1 blend
    # quantizes back to 0/1 - no whole-canvas 255 factor on this path
    assert int(roi_paste[0].max()) <= 1, int(roi_paste[0].max())

    reference = _vendored_paste(
        face_helper, background[0], face, cv2.invertAffineTransform(matrix),
        1.0, parse_net,
    )
    # the vendored paste still takes its near-black branch: the whole
    # canvas is scaled by 255 - the documented divergence of the fix
    assert int(reference.max()) == 255, int(reference.max())


@pytest.mark.unit
def test_a_dark_roi_in_a_bright_frame_keeps_the_frame_global_branch(gpu_paste):
    """The near-black branch is decided on the **whole frame**.

    A frame whose ROI is near black but whose background outside the ROI
    is bright must **not** take the near-black branch: the factor of the
    vendored tail comes from the frame-global maximum, so the ROI keeps its
    values.  A ROI-local maximum would scale the ROI by 255 and ruin the
    frame - the pin catches exactly that.

    Fix 2.1 (2026-10-30) note: with the vectorized chunk the ROI path is
    no longer bit-identical to the full-canvas reference - the pin holds
    the measured residual instead (measured **1/255** on this corpus,
    pinned at the 8/255 class bound) and keeps the outside-ROI background
    preservation, which the zero mask outside the footprint guarantees.
    """
    height, width = ROI_CANVAS_HW
    matrix = _face_to_canvas(ROI_CANVAS_HW, "interior")
    x0, y0, x1, y1 = gpu_paste.face_roi_box(matrix, ROI_CANVAS_HW)
    rng = np.random.default_rng(167)
    # a near-black ROI inside a bright frame
    face = rng.integers(0, 2, size=(512, 512, 3), dtype=np.uint8)
    background = np.full((1, height, width, 3), 200, dtype=np.uint8)
    background[0, y0:y1, x0:x1] = 0
    kwargs = dict(
        faces_bgr=face[None],
        inverse_affines=[matrix],
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=_StubParseNet(seed=173),
        frame_chunk=1,
    )

    roi_paste = gpu_paste.paste_faces_batch(use_roi=True, **kwargs)
    stats = gpu_paste.PASTE_TRANSFER_STATS
    assert stats["fallback_frames"] == 0, stats
    assert stats["roi_frames"] == 1, stats

    full_paste = gpu_paste.paste_faces_batch(use_roi=False, **kwargs)
    diff = np.abs(roi_paste[0].astype(int) - full_paste[0].astype(int)).max()
    assert diff <= 8, (
        f"the dark ROI's paste drifted {diff}/255 from the full-canvas paste"
    )
    # the pixels outside the ROI keep their background value
    outside = np.ones((height, width), dtype=bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(roi_paste[0][outside], background[0][outside]), (
        "the pixels outside the ROI did not keep their background value"
    )


# ---------------------------------------------------------------------------
# The device-output assembly (the GPU-resident-output round)
#
# The round that keeps the paste's output stack on the device past the F1
# budget: the VRAM budget of the device canvas (``device_canvas_budget``),
# the three-way mode selector (``device_output_mode``), the sub-chunked
# convert-out (``comfy_convert_out``) and the chunked device assembly
# (``paste_faces_batch_chunked``).  The pins are **bit-exact** and run on
# the CPU device only: every op of the paste is batch-invariant there
# (measured - the parse forward, the warps, the blend and the elementwise
# convert chain), so the chunked mode equals the full mode equals the host
# stack for any chunk boundary.  On CUDA the production group size is the
# full mode's own ``paste_chunk_size``, which keeps the batch compositions
# equal by construction; the Lanczos background resize stays the one
# owner-gated residual.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_device_canvas_budget_follows_the_free_memory(gpu_paste, monkeypatch):
    """The canvas budget: the free-VRAM share under the F1 hard cap.

    The mirror of the chunk-size pins above: ``free_memory_bytes`` is
    patched for determinism (with a ComfyUI host it returns the **host**
    RAM for a CPU device - ``comfy/model_management.py:1816-1818`` - so a
    CPU budget is the cap without the patch).  No number keeps the fixed
    cap (the F1 engagement, unchanged); a quarter of 2 GiB degrades the
    budget to 512 MiB (the small-free card); 64 GiB of free memory is
    capped back at 1 GiB (the share never engages past the F1 one).
    """
    device = torch.device("cpu")
    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 0)
    assert gpu_paste.device_canvas_budget(device) == gpu_paste.PASTE_DEVICE_STACK_MAX_BYTES

    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 2 << 30)
    assert gpu_paste.device_canvas_budget(device) == 1 << 29, (
        "a quarter of 2 GiB is 512 MiB - below the cap, so the share decides"
    )

    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 64 << 30)
    assert gpu_paste.device_canvas_budget(device) == gpu_paste.PASTE_DEVICE_STACK_MAX_BYTES


@pytest.mark.unit
def test_the_device_output_mode_selects_full_chunked_or_host(gpu_paste, monkeypatch):
    """The three-way selector: full inside the budget, chunked past it.

    A ``torch.device("cuda")`` **object** drives the CUDA branch - the
    selector only reads ``.type`` and the patched memory, so no CUDA is
    touched (the CPU-only rule of the round).  The factor-4 numbers are
    the owner's complaint: 100 frames on a 2880x5120 canvas are ~4.1 GiB
    of uint8, past the 1 GiB cap, so the full mode disengages; one chunk
    of 4 frames (the 2 GiB staging over ~436 MiB per frame) still fits, so
    the chunked mode engages.  Fix 2.2 (2026-10-30, the owner's
    ``keep-fix3.txt``): a starved card no longer keeps the host assembly -
    a CUDA device never gets ``"host"``; the chunked mode takes it with
    the group floored at one frame (:func:`paste_chunk_size`), the OOM
    handler stays the net.  The last two assertions pin the boundary
    itself: ``num_frames * canvas_bytes == budget`` stays ``full`` (the
    ``<=``), one frame past it falls to ``chunked``.
    """
    cuda = torch.device("cuda")
    # a CPU device keeps the host output (the F1 behavior, unpatched)
    assert gpu_paste.device_output_mode(torch.device("cpu"), 4, 256, 384) == "host"

    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 8 << 30)
    assert gpu_paste.device_output_mode(cuda, 4, 256, 384) == "full"
    assert gpu_paste.device_output_mode(cuda, 100, 2880, 5120) == "chunked"

    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 64 << 20)
    assert gpu_paste.device_output_mode(cuda, 100, 2880, 5120) == "chunked"

    # the exact budget boundary: four frames of a 256x384 canvas are
    # exactly a quarter of 4,718,592 bytes of "free" memory
    monkeypatch.setattr(gpu_paste, "free_memory_bytes", lambda _device: 4_718_592)
    assert gpu_paste.device_output_mode(cuda, 4, 256, 384) == "full"
    assert gpu_paste.device_output_mode(cuda, 5, 256, 384) == "chunked"


@pytest.mark.unit
def test_the_convert_out_sub_chunks_without_changing_a_pixel(gpu_paste, image_utils):
    """The budget-sized sub-chunks of the convert-out are bit-exact.

    The full mode's convert-out peaks at about 9x the canvas bytes of
    device transient (the out-of-place RGB gather, the float32 cast and
    the ``/255`` - the design's D4 finding), so both device tails convert
    in sub-chunks of ``budget // (9 * frame_bytes)`` frames.  The chain is
    elementwise - batch-invariant on the CPU device like every op of the
    paste - so the sub-chunked result is the single
    :func:`~.keep_image_utils.bgr_u8_batch_to_comfy` call, byte for byte.
    The budget is forced to one frame and to two (a non-tail-aligned 2/2/1
    split over the five frames), and the preallocated ``out`` receives the
    result in place.
    """
    rng = np.random.default_rng(191)
    canvas = torch.from_numpy(
        rng.integers(0, 256, size=(5, 48, 72, 3), dtype=np.uint8))
    single = image_utils.bgr_u8_batch_to_comfy(canvas)
    transient = 9 * 48 * 72 * 3  # the per-frame peak the budget bounds

    assert torch.equal(gpu_paste.comfy_convert_out(canvas), single)
    assert torch.equal(gpu_paste.comfy_convert_out(canvas, budget=transient), single)
    assert torch.equal(
        gpu_paste.comfy_convert_out(canvas, budget=2 * transient), single)

    out = torch.empty((5, 48, 72, 3), dtype=torch.float32)
    assert gpu_paste.comfy_convert_out(canvas, out=out, budget=transient) is out
    assert torch.equal(out, single)


@pytest.mark.unit
def test_the_device_output_stack_is_bit_identical_to_the_host_stack_at_upscale_four(
    gpu_paste, gpu_ops
):
    """AC 4: the device-output canvas at the upscale factor of four.

    The factor-4 canvas is where the host assembly revives today (16x the
    pixels), so the pin runs the cheap stand-in: 64x96 frames paste onto a
    256x384 canvas.  The 4x background is produced **once** and feeds both
    modes, so the pin isolates the assembly of the output stack - exactly
    the F1 pin above, at the factor the budget bites.  (The processor's
    factor-4 host background is the cv2 Lanczos while the device modes use
    the owner-accepted device Lanczos - a tolerance route the end-to-end
    pin stays clear of by comparing at one factor.)
    """
    frames = smooth_frames_u8(2, (64, 96), seed=193)
    landmarks = synthetic_landmarks(2, (64, 96), seed=197)
    rng = np.random.default_rng(199)
    faces = rng.integers(0, 256, size=(2, 512, 512, 3), dtype=np.uint8)
    affines = gpu_ops.estimate_affines(landmarks, FACE_TEMPLATE)
    background = gpu_paste.batched_bg_frames(frames, 4.0, None, torch.device("cpu"))
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=4.0)
    kwargs = dict(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=_StubParseNet(seed=211),
        frame_chunk=2,
    )

    host_frames = gpu_paste.paste_faces_batch(**kwargs)
    device_canvas = gpu_paste.paste_faces_batch(device_output=True, **kwargs)

    assert background.shape == (2, 256, 384, 3), background.shape
    assert tuple(device_canvas.shape) == (2, 256, 384, 3), device_canvas.shape
    for index, host_frame in enumerate(host_frames):
        assert np.array_equal(device_canvas[index].numpy(), host_frame), (
            f"frame {index}: the factor-4 device canvas drifted from the host stack"
        )


@pytest.mark.unit
def test_the_chunked_assembly_matches_the_host_stack_and_the_full_canvas(
    gpu_paste, image_utils
):
    """AC 3a: the chunked helper's IMAGE rows == the host stack == the canvas.

    The helper pastes in ``frame_chunk=1`` groups - the smallest possible -
    while the full mode runs the whole batch and the host stack is the
    shipped list return.  On the CPU device every op of the paste is
    batch-invariant (measured, the design's D3), so all three assemblies
    come out as the same float32 IMAGE batch, byte for byte.
    """
    rng = np.random.default_rng(223)
    faces = np.stack([_face_u8("smooth", seed=227), _face_u8("noise", seed=229)])
    matrices = [
        _face_to_canvas(ROI_CANVAS_HW, "interior"),
        _face_to_canvas(ROI_CANVAS_HW, "interior", scale=0.3),
    ]
    background = rng.integers(0, 256, size=(2, *ROI_CANVAS_HW, 3), dtype=np.uint8)
    kwargs = dict(
        inverse_affines=matrices,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=_StubParseNet(seed=233),
    )

    host_frames = gpu_paste.paste_faces_batch(faces_bgr=faces, frame_chunk=2, **kwargs)
    full_canvas = gpu_paste.paste_faces_batch(
        faces_bgr=faces, frame_chunk=2, device_output=True, **kwargs)
    chunked = gpu_paste.paste_faces_batch_chunked(
        faces_bgr=faces,
        upscale_factor=1.0,
        bg_upscale_model=None,
        canvas_hw=None,
        frame_chunk=1,
        **kwargs,
    )

    assert chunked.dtype == torch.float32 and chunked.device.type == "cpu"
    assert tuple(chunked.shape) == (2, *ROI_CANVAS_HW, 3), chunked.shape
    assert torch.equal(
        chunked, image_utils.bgr_u8_batch_to_comfy(np.stack(host_frames))
    ), "the chunked helper drifted from the host stack"
    assert torch.equal(chunked, gpu_paste.comfy_convert_out(full_canvas)), (
        "the chunked helper drifted from the full device mode"
    )


@pytest.mark.unit
def test_the_chunked_assembly_is_independent_of_the_group_boundaries(
    gpu_paste, image_utils
):
    """AC 3b: 17 faces in groups of 2 == the full mode's one whole batch.

    The group boundaries (2, with the 8 x 2 + 1 remainder group) do not
    align with the full mode's internal chunk (the whole batch in one), so
    every op sees a different batch composition - and the IMAGE rows still
    come out identical on the CPU device, where the parse forward, the
    warps, the blend and the convert chain are batch-invariant (measured,
    the design's D3).  On CUDA the production group size **is** the full
    mode's own :func:`paste_chunk_size`, so the batch compositions are
    equal by construction; the Lanczos background resize (whole batch in
    the full mode, groups in this one) stays the one owner-gated residual.
    """
    rng = np.random.default_rng(239)
    faces = rng.integers(0, 256, size=(17, 512, 512, 3), dtype=np.uint8)
    matrices = [
        _face_to_canvas(
            ROI_CANVAS_HW,
            "interior" if index % 2 == 0 else "edge",
            scale=0.25 if index % 3 else 0.3,
        )
        for index in range(17)
    ]
    background = rng.integers(0, 256, size=(17, *ROI_CANVAS_HW, 3), dtype=np.uint8)
    kwargs = dict(
        inverse_affines=matrices,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=_StubParseNet(seed=241),
    )

    full_canvas = gpu_paste.paste_faces_batch(
        faces_bgr=faces, frame_chunk=17, device_output=True, **kwargs)
    chunked = gpu_paste.paste_faces_batch_chunked(
        faces_bgr=faces,
        upscale_factor=1.0,
        bg_upscale_model=None,
        canvas_hw=None,
        frame_chunk=2,
        **kwargs,
    )

    assert torch.equal(chunked, gpu_paste.comfy_convert_out(full_canvas)), (
        "the group boundaries changed the pasted pixels"
    )


@pytest.mark.unit
def test_the_chunked_assembly_accepts_the_lazy_source_and_matches_the_numpy_stack(
    gpu_paste, image_utils
):
    """The lazy background source: the same pixels, asked group by group.

    The round-17 source (:class:`~.keep_image_utils.BgrFrameSource`) wraps
    the caller's float32 IMAGE tensor and converts the rows **on demand**,
    so the paste never holds the whole-clip uint8 stack (~3.78 GB on the
    owner's 777-frame clip).  The pin is **bit-exact**: the chunked
    assembly over the source equals the one over the numpy stack, byte for
    byte - the converter is elementwise, so a row converted alone is the
    row of the whole-clip conversion.

    The second half is the memory discipline: the source is asked **per
    group** (the spy records the batch shapes), so no conversion ever
    covers the whole clip.
    """
    num_frames = 3
    rng = np.random.default_rng(251)
    faces = rng.integers(0, 256, size=(num_frames, 512, 512, 3), dtype=np.uint8)
    matrices = [
        _face_to_canvas(
            ROI_CANVAS_HW, "interior" if index % 2 == 0 else "edge", scale=0.3
        )
        for index in range(num_frames)
    ]
    # the node's float32 IMAGE batch - the payload of the lazy source
    image_batch = torch.rand(
        (num_frames, *ROI_CANVAS_HW, 3),
        generator=torch.Generator().manual_seed(257),
    )
    whole = image_utils.comfy_batch_to_bgr_u8(image_batch)

    kwargs = dict(
        faces_bgr=faces,
        inverse_affines=matrices,
        upscale_factor=1.0,
        bg_upscale_model=None,
        device=torch.device("cpu"),
        canvas_hw=None,
        parse_net=_StubParseNet(seed=259),
        frame_chunk=2,
    )

    numpy_rows = gpu_paste.paste_faces_batch_chunked(background_bgr=whole, **kwargs)

    calls: list[tuple[int, ...]] = []

    def _recording_convert(batch):
        calls.append(tuple(batch.shape))
        return image_utils.comfy_batch_to_bgr_u8(batch)

    source = image_utils.BgrFrameSource(image_batch, convert=_recording_convert)
    source_rows = gpu_paste.paste_faces_batch_chunked(background_bgr=source, **kwargs)

    assert torch.equal(source_rows, numpy_rows), (
        "the lazy source drifted from the numpy stack"
    )
    assert calls == [(2, *ROI_CANVAS_HW, 3), (1, *ROI_CANVAS_HW, 3)], (
        f"the source was not asked group by group: {calls}"
    )


@pytest.mark.unit
def test_the_core_writes_the_indexed_destination_byte_for_byte(gpu_paste):
    """The indexed destination: the core's own bytes, in the named rows.

    Round 18 hands the core a **caller-owned** buffer: the processor
    allocates the final whole-clip float32 tensor and the core writes the
    eligible rows straight into it, by index - no second buffer alive at
    the same time, no copy of the rows.  The pin is **bit-exact**: the
    destination rows carry the core's own-buffer bytes (the same run
    without ``out=``), byte for byte, and the rows the map does **not**
    name keep the sentinel fill.

    The sentinel is the point of the pin: a slice write
    (``out[start:stop] = ...``) would land the rows on the wrong indices,
    and the read-side ``out[rows]`` (advanced indexing - a **copy**) would
    write into a temporary and lose the rows.  Both fail this pin loudly
    instead of shipping a corrupted buffer.

    The map ``[0, 2, 5]`` is scattered on purpose, and the group size of
    two splits it across the groups (the first group owns the rows 0 and
    2, the second one the row 5) - the indexed write has to survive the
    group boundary.
    """
    num_rows = 3
    rng = np.random.default_rng(401)
    faces = np.stack([
        _face_u8("smooth", 403),
        _face_u8("noise", 405),
        _face_u8("smooth", 407),
    ])
    matrices = [
        _face_to_canvas(ROI_CANVAS_HW, "interior"),
        _face_to_canvas(ROI_CANVAS_HW, "edge", scale=0.3),
        _face_to_canvas(ROI_CANVAS_HW, "interior", scale=0.22),
    ]
    background = rng.integers(
        0, 256, size=(num_rows, *ROI_CANVAS_HW, 3), dtype=np.uint8)
    kwargs = dict(
        faces_bgr=faces,
        inverse_affines=matrices,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=_StubParseNet(seed=409),
    )
    core_kwargs = dict(
        upscale_factor=1.0,
        bg_upscale_model=None,
        canvas_hw=None,
        frame_chunk=2,
    )
    destination_rows = [0, 2, 5]

    own = gpu_paste.paste_faces_batch_chunked(**core_kwargs, **kwargs)
    assert tuple(own.shape) == (num_rows, *ROI_CANVAS_HW, 3), own.shape

    sentinel = -7.5
    given = torch.full((6, *ROI_CANVAS_HW, 3), sentinel, dtype=torch.float32)
    returned = gpu_paste.paste_faces_batch_chunked(
        **core_kwargs, **kwargs, out=given, out_rows=destination_rows)

    assert returned is given, "the core did not return the destination it was given"
    for canvas_row, destination_row in enumerate(destination_rows):
        assert torch.equal(given[destination_row], own[canvas_row]), (
            f"the destination row {destination_row} does not carry the "
            f"canvas row {canvas_row}'s bytes"
        )
    untouched = torch.full((3, *ROI_CANVAS_HW, 3), sentinel, dtype=torch.float32)
    for offset, row in enumerate((1, 3, 4)):
        assert torch.equal(given[row], untouched[offset]), (
            f"the row {row} was written although the map does not name it - "
            "the write was a slice, not an indexed assignment"
        )


@pytest.mark.unit
def test_the_indexed_destination_is_validated_loudly(gpu_paste):
    """A broken destination map fails loudly instead of losing a row.

    The indexed destination is a **pair**: ``out`` names the tensor and
    ``out_rows`` names its rows, so one without the other cannot be
    honoured.  The map has to cover exactly the canvas rows, name
    distinct rows of the destination and stay inside it, and the
    destination has to be the float32 **host** tensor of the canvas shape
    - the core's own canvas derivation is the authority of the shape, so a
    divergence between the caller's canvas and the core's is a loud
    failure, not a silent corruption.
    """
    num_rows = 3
    rng = np.random.default_rng(411)
    faces = np.stack([
        _face_u8("smooth", 413),
        _face_u8("noise", 415),
        _face_u8("smooth", 417),
    ])
    matrices = [
        _face_to_canvas(ROI_CANVAS_HW, "interior"),
        _face_to_canvas(ROI_CANVAS_HW, "edge", scale=0.3),
        _face_to_canvas(ROI_CANVAS_HW, "interior", scale=0.22),
    ]
    background = rng.integers(
        0, 256, size=(num_rows, *ROI_CANVAS_HW, 3), dtype=np.uint8)
    kwargs = dict(
        faces_bgr=faces,
        inverse_affines=matrices,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=_StubParseNet(seed=419),
        upscale_factor=1.0,
        bg_upscale_model=None,
        canvas_hw=None,
    )
    canvas = torch.zeros((num_rows, *ROI_CANVAS_HW, 3), dtype=torch.uint8)

    def destination(rows=6, dtype=torch.float32, shape=None):
        if shape is None:
            shape = (rows, *ROI_CANVAS_HW, 3)
        return torch.zeros(shape, dtype=dtype)

    # the pair: the destination without its map, and the map without its
    # destination
    with pytest.raises(ValueError):
        gpu_paste.paste_faces_batch_chunked(
            **kwargs, out=destination())
    with pytest.raises(ValueError):
        gpu_paste.paste_faces_batch_chunked(
            **kwargs, out_rows=[0, 1, 2])
    # the map: the wrong length, a repeated row, a row outside the buffer
    with pytest.raises(ValueError):
        gpu_paste.paste_faces_batch_chunked(
            **kwargs, out=destination(), out_rows=[0, 2])
    with pytest.raises(ValueError):
        gpu_paste.paste_faces_batch_chunked(
            **kwargs, out=destination(), out_rows=[0, 0, 5])
    with pytest.raises(ValueError):
        gpu_paste.paste_faces_batch_chunked(
            **kwargs, out=destination(), out_rows=[0, 2, 6])
    # the destination: the wrong dtype, the wrong canvas, too few rows
    with pytest.raises(ValueError):
        gpu_paste.paste_faces_batch_chunked(
            **kwargs, out=destination(dtype=torch.uint8), out_rows=[0, 2, 5])
    with pytest.raises(ValueError):
        gpu_paste.paste_faces_batch_chunked(
            **kwargs, out=destination(shape=(6, 64, 64, 3)), out_rows=[0, 2, 5])
    with pytest.raises(ValueError):
        gpu_paste.paste_faces_batch_chunked(
            **kwargs, out=destination(rows=2), out_rows=[0, 1, 2])
    # the convert-out: the map names the rows of a destination, so it
    # cannot allocate one
    with pytest.raises(ValueError):
        gpu_paste.comfy_convert_out(canvas, out_rows=[0, 1, 2])


@pytest.mark.unit
def test_the_numpy_background_provider_serves_the_same_rows_without_copying(gpu_paste):
    """The numpy adapter: the rows the core used to slice, as views.

    The paste core consumes its background through one provider protocol,
    so the lazy source and a numpy stack share a single code path.  This
    adapter is the numpy side: the ``group`` of a contiguous range is the
    **view** the core took before (``background[start:stop]``), the shape
    is the array's shape and the ``materialize`` is the array itself.  The
    zero-copy claim is the pin - a copy here would rebuild the whole
    eligible stack.
    """
    frames = np.arange(3 * 4 * 5 * 3, dtype=np.uint8).reshape(3, 4, 5, 3)

    provider = gpu_paste.frame_provider(frames)
    assert provider.shape == (3, 4, 5, 3)
    assert len(provider) == 3

    group = provider.group(slice(1, 3))
    assert np.array_equal(group, frames[1:3])
    assert np.shares_memory(group, frames), "the group copied the rows"

    assert np.array_equal(provider.group(range(1, 3)), frames[1:3])
    assert np.array_equal(provider.group([0, 2]), frames[[0, 2]])

    # the materialized stack is the array itself - no copy
    assert provider.materialize() is frames

    # the gather of the eligible rows: the copy the numpy path always made
    selected = provider.select([0, 2])
    assert selected.shape == (2, 4, 5, 3)
    assert np.array_equal(selected.materialize(), frames[[0, 2]])


# ---------------------------------------------------------------------------
# The batched background (design section 7.7, step 1)
# ---------------------------------------------------------------------------


class _DoublingUpscaleModule(torch.nn.Module):
    """The inner ``.model`` of ``_StubUpscaleModel`` - records and doubles.

    keep-fix5-implement2 (2026-11-03): the re-entrant
    ``_spandrel_on_device`` probes ``next(m.model.parameters()).device``,
    so the inner model is a real ``nn.Module`` (a bound method has no
    ``parameters()``); the forward is the same doubling the former method
    body ran - the pin's math is unchanged.
    """

    def __init__(self, owner) -> None:
        super().__init__()
        self.owner = owner
        self.mark = torch.nn.Parameter(torch.zeros(1))

    def forward(self, samples):
        self.owner.calls.append(tuple(samples.shape))
        return samples * 2.0


class _StubUpscaleModel:
    """A background upscaler with the ComfyUI model contract.

    The descriptor form (keep-fix5-implement2, 2026-11-03): the ``.to``
    the lifecycle manager calls moves the inner module, like spandrel's
    own descriptor ``.to`` does.
    """

    scale = 2.0

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.model = _DoublingUpscaleModule(self)

    def to(self, device, *args, **kwargs):
        self.model.to(device, *args, **kwargs)
        return self


@pytest.mark.unit
def test_the_batched_background_is_pixel_identical_to_the_per_frame_calls(
    gpu_paste, processor_module
):
    """The model branch of ``batched_bg_frames`` == N per-frame upscales.

    Re-authored for keep-fix5-implement1 (Phase 1, step 1.1.2, 2026-11-03):
    the model branch runs ``_tiled_scale_and_resize_batch`` - one frame per
    ``tiled_scale`` call in FP16 autocast on CUDA (the CPU gate runs FP32),
    the canvas resize happens **on the float32 scaled output**
    (``F.interpolate`` bicubic antialias, the plan's kernel - the Lanczos
    reference of the former pin is superseded) and the quantization rounds
    (the truncating cast of the former ``_tiled_scale_batch`` is
    superseded).  The reference is the per-frame chain re-derived
    independently below - the same model work, the same resize, the same
    round quantize - so the pin stays **bit-exact** and now covers the
    canvas resize (a target off the model's own scale) as well.

    The former pin compared against ``_run_upscaler`` + LANCZOS4 on the
    identity-size case; that reference encodes the superseded kernel and
    quantization, so it is replaced, not loosened: the bound stays
    ``np.array_equal``.
    """
    from comfy.utils import tiled_scale  # the conftest mock passthrough

    frames = smooth_frames_u8(3, (64, 96), seed=53)
    model = _StubUpscaleModel()

    # the upscale factor of 2 makes the target size the double one; the
    # canvas_hw variant gives the target a size the model's own scale does
    # not produce, so the float32 bicubic resize runs
    batched = gpu_paste.batched_bg_frames(frames, 2.0, model, torch.device("cpu"))
    batched_canvas = gpu_paste.batched_bg_frames(
        frames, 2.0, model, torch.device("cpu"), canvas_hw=(96, 144)
    )
    batched_device = gpu_paste.batched_bg_frames(
        frames, 2.0, model, torch.device("cpu"), return_device=True
    )

    def reference_for(target_hw):
        target_h, target_w = (int(value) for value in target_hw)
        output = []
        for frame in frames:
            sample = (
                torch.from_numpy(frame[..., [2, 1, 0]])
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(torch.float32)
                .div_(255.0)
            )
            scaled = tiled_scale(
                sample,
                lambda a: model.model(a),
                tile_x=512,
                tile_y=512,
                overlap=64,
                upscale_amount=model.scale,
                output_device=torch.device("cpu"),
            )
            scaled = scaled.float()
            if scaled.shape[-2] != target_h or scaled.shape[-1] != target_w:
                scaled = F.interpolate(
                    scaled,
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            output.append(
                scaled.squeeze(0)
                .clamp_(0.0, 1.0)
                .mul_(255.0)
                .round_()
                .to(torch.uint8)
                .permute(1, 2, 0)[..., [2, 1, 0]]
                .numpy()
            )
        return np.stack(output)

    assert batched.shape == (3, 128, 192, 3), batched.shape
    assert np.array_equal(batched, reference_for((128, 192)))
    # the canvas target is (192, 288) - the model's own 2x output (128, 192)
    # is smaller, so the float32 bicubic resize actually runs
    assert batched_canvas.shape == (3, 192, 288, 3), batched_canvas.shape
    assert np.array_equal(batched_canvas, reference_for((192, 288)))
    assert batched_device.dtype == torch.uint8
    assert torch.equal(
        batched_device,
        torch.from_numpy(np.ascontiguousarray(reference_for((128, 192)))),
    )


# ---------------------------------------------------------------------------
# The bg-upscale frame groups of the former round 12 were removed with
# keep-fix5 (2026-11-03): ``bg_chunk_size`` lost its production caller to
# ``_tiled_scale_and_resize_batch`` (a B=1 loop, no groups to pin), so the
# group-equality pin and the ``bg_chunk_size`` free-memory pin went with
# the dead machinery - a removal, not a loosening.
# ---------------------------------------------------------------------------


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


@pytest.mark.unit
def test_the_device_gray_pass_never_uploads_more_than_one_group(
    gpu_paste, monkeypatch
):
    """Round 16: the gray-flag pass uploads one pixel-budget group at a time.

    The former implementation moved the whole (N, H, W, 3) stack to the
    device (3.78 GB on the 777-frame clip) before the pixel-chunk loop in
    ``_is_gray_diff_sums_device``.  With the pixel budget smaller than one
    frame, the pass must split the batch; the values must still be the
    numpy ones (the CPU device exercises the same float64 reduction on the
    host, because ``is_gray_batch`` only routes CUDA devices to it).
    """
    rng = np.random.default_rng(63)
    frames = np.concatenate(
        [
            rng.integers(0, 256, size=(4, 32, 48, 3), dtype=np.uint8),
            np.repeat(
                rng.integers(0, 256, size=(4, 32, 48, 1), dtype=np.uint8), 3, axis=3
            ),
        ]
    )

    device = torch.device("cpu")
    sizes: list[tuple[int, ...]] = []

    real_from_numpy = torch.from_numpy

    def _recording_from_numpy(array):
        result = real_from_numpy(array)
        sizes.append(tuple(result.shape))
        return result

    monkeypatch.setattr(torch, "from_numpy", _recording_from_numpy)
    monkeypatch.setattr(gpu_paste, "GRAY_CHUNK_PIXELS", 1024)  # < 1 frame
    values = gpu_paste._is_gray_diff_sums_device(frames, device)

    budget = 1024 // (32 * 48)
    assert budget < 1
    assert sizes, "the pass must upload through torch.from_numpy"
    assert max(size[0] for size in sizes) == 1, sizes
    assert max(size[0] for size in sizes) <= len(frames)

    reference = np.array(
        [
            gpu_paste._is_gray_diff_sum_numpy(frames[i])
            for i in range(len(frames))
        ]
    )
    assert np.allclose(values, reference), (values, reference)


# ---------------------------------------------------------------------------
# The VRAM-aware chunk sizes (design section 7.7, step 3)
# ---------------------------------------------------------------------------


# The ``bg_chunk_size`` pin that followed here (the x4-intermediate group
# budget) went with the function - keep-fix5 (2026-11-03) replaced the
# bg frame groups with the per-frame loop, the budget machinery is dead.
# The parse / paste chunk pins below keep the free-VRAM sizing pinned.


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

    # keep-fix5-implement1 (2026-11-03): the vendored ``read_image`` gained
    # the ``check_gray`` flag (plan 2.2) - the sequence sites pass it.
    def read_image(self, image, check_gray: bool = True):
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
def test_the_box_rides_the_batched_core(processor_module, gpu_paste, gpu_ops, monkeypatch):
    """``draw_box=True`` rides the batched core (round 10, the owner's fix 4).

    The box no longer disqualifies the batch (the owner's fix 4,
    2026-10-31): the vendored paste is not called and the boxes are drawn
    on the device-output canvases (:meth:`_draw_boxes_on_canvas`) - the
    green 2-px ``face_roi_box`` border on the forced-``"full"`` canvas.
    The host branch draws none (the disclosed CPU gap of the fix): the
    unpatched CPU run's output is bit-identical to the ``draw_box=False``
    run - the measured behavior replacing the old
    ``test_the_vendored_paste_stays_reachable_for_the_box`` pin (which
    routed the box through the vendored paste).
    """
    landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _ParseFaceHelper(parse_net=_StubParseNet(), landmarks=landmarks)
    frames = torch.rand((1, 64, 96, 3), dtype=torch.float32)

    output, _ = _run_sequence(processor_module, face_helper, frames, draw_box=True)

    assert "paste_faces_to_input_image" not in face_helper.calls, face_helper.calls
    assert tuple(output.shape) == (1, 64, 96, 3), output.shape

    # the host branch of the same run draws no box: the output is the
    # draw_box=False run bit for bit
    plain, _ = _run_sequence(processor_module, face_helper, frames, draw_box=False)
    assert torch.equal(output, plain), "the CPU draw_box run drew on the host path"

    # the device canvas carries the green 2-px border of the face's ROI
    frames_u8 = smooth_frames_u8(1, (64, 96), seed=293)
    landmarks_u8 = synthetic_landmarks(1, (64, 96), seed=297)
    rng = np.random.default_rng(307)
    faces = rng.integers(0, 256, size=(1, 512, 512, 3), dtype=np.uint8)
    affines = gpu_ops.estimate_affines(landmarks_u8, FACE_TEMPLATE)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)
    gray_flags = np.zeros(1, dtype=bool)
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _ParseFaceHelper(parse_net=_StubParseNet(seed=311))))

    monkeypatch.setattr(gpu_paste, "device_output_mode", lambda *a, **k: "full")
    boxed = processor._batched_paste(
        frames_u8, faces, list(affines), landmarks_u8, 1.0, True, gray_flags)
    canvas = boxed.pop(gpu_paste.DEVICE_CANVAS_KEY)
    x0, y0, x1, y1 = gpu_paste.face_roi_box(inverses[0], (64, 96))
    assert x1 > x0 + 2 and y1 > y0 + 2, (x0, y0, x1, y1)
    green = torch.tensor([0, 255, 0], dtype=torch.uint8)
    assert torch.equal(canvas[0, y0 : y0 + 2, x0:x1], green.expand(2, x1 - x0, 3)), (
        "the top border of the box is missing from the canvas"
    )
    assert torch.equal(canvas[0, y1 - 2 : y1, x0:x1], green.expand(2, x1 - x0, 3)), (
        "the bottom border of the box is missing from the canvas"
    )
    assert torch.equal(canvas[0, y0:y1, x0 : x0 + 2], green.expand(y1 - y0, 2, 3)), (
        "the left border of the box is missing from the canvas"
    )
    assert torch.equal(canvas[0, y0:y1, x1 - 2 : x1], green.expand(y1 - y0, 2, 3)), (
        "the right border of the box is missing from the canvas"
    )


@pytest.mark.unit
def test_the_grayscale_frames_ride_the_batched_core(processor_module):
    """A grayscale frame rides the batched core (fix 1.2, 2026-10-30).

    The gray skip of the eligibility is gone: the device blend pastes a
    grayscale frame with the same tensor logic as a color frame, so the
    gray corpus of the old
    ``test_the_vendored_paste_stays_reachable_for_the_grayscale_frames``
    takes the batched paste - the vendored paste is not called and the
    output shape holds.
    """
    landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _ParseFaceHelper(parse_net=_StubParseNet(), landmarks=landmarks)
    gray = torch.full((1, 64, 96, 3), 0.4, dtype=torch.float32)
    gray[..., 1] = 0.5

    output, _ = _run_sequence(processor_module, face_helper, gray)

    assert "paste_faces_to_input_image" not in face_helper.calls, face_helper.calls
    assert tuple(output.shape) == (1, 64, 96, 3), output.shape


@pytest.mark.unit
def test_the_vendored_paste_stays_reachable_for_a_non_parse_face(processor_module, monkeypatch):
    """The vendored paste stays reachable for a restored face off the ParseNet size.

    The gray routing this file used to pin is gone (fix 1.2, 2026-10-30):
    the per-frame ineligibility that keeps a frame on the vendored paste
    (with a parse net, no box and no face upsampler) is the restored face
    of a size other than ``PARSE_FACE_SIZE``.  The pin patches the
    processor's ``PARSE_FACE_SIZE`` to 256, so the restored 512 faces are
    ineligible while the landmarks and the affines stay valid (no
    short-circuit) - the frame takes the vendored paste.
    """
    landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
    face_helper = _ParseFaceHelper(parse_net=_StubParseNet(), landmarks=landmarks)
    frames = torch.rand((1, 64, 96, 3), dtype=torch.float32)
    monkeypatch.setattr(processor_module, "PARSE_FACE_SIZE", 256)

    output, _ = _run_sequence(processor_module, face_helper, frames)

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


@pytest.mark.unit
def test_the_batched_paste_returns_the_device_output_sentinels(
    processor_module, gpu_paste, gpu_ops, monkeypatch
):
    """The wiring: the selector's verdict picks the sentinel of the mapping.

    ``_batched_paste`` runs the real device-mode code on the CPU device
    (the F1 pins prove it) - only the selector is patched, so the
    ``"full"`` verdict returns :data:`~.keep_gpu_paste.DEVICE_CANVAS_KEY`
    with one uint8 canvas, the ``"chunked"`` verdict
    :data:`~.keep_gpu_paste.DEVICE_OUTPUT_CHUNKS_KEY` with the float32
    IMAGE rows, and the unpatched CPU run keeps the host mapping
    (``{index: frame}``, no sentinel).  Since round 18 the chunk sentinel
    carries the **whole-clip** float32 buffer, so a mapped index names the
    buffer's own final row (in the all-eligible case of this pin that is
    also the order of the eligible rows).
    """
    frames = smooth_frames_u8(2, (64, 96), seed=251)
    landmarks = synthetic_landmarks(2, (64, 96), seed=257)
    rng = np.random.default_rng(263)
    faces = rng.integers(0, 256, size=(2, 512, 512, 3), dtype=np.uint8)
    affines = gpu_ops.estimate_affines(landmarks, FACE_TEMPLATE)
    gray_flags = np.zeros(2, dtype=bool)
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _ParseFaceHelper(parse_net=_StubParseNet(seed=269))))

    def call():
        return processor._batched_paste(
            frames, faces, affines, landmarks, 1.0, False, gray_flags)

    host = call()
    assert gpu_paste.DEVICE_CANVAS_KEY not in host
    assert gpu_paste.DEVICE_OUTPUT_CHUNKS_KEY not in host
    assert all(isinstance(host[index], np.ndarray) for index in host)

    monkeypatch.setattr(gpu_paste, "device_output_mode", lambda *a, **k: "full")
    full = call()
    canvas = full.pop(gpu_paste.DEVICE_CANVAS_KEY)
    assert isinstance(canvas, torch.Tensor), type(canvas)
    assert canvas.dtype == torch.uint8 and canvas.device.type == "cpu"
    assert tuple(canvas.shape) == (2, 64, 96, 3), canvas.shape
    assert gpu_paste.DEVICE_OUTPUT_CHUNKS_KEY not in full
    assert list(full) == [0, 1]
    assert torch.equal(torch.stack([full[0], full[1]]), canvas)

    monkeypatch.setattr(gpu_paste, "device_output_mode", lambda *a, **k: "chunked")
    chunked = call()
    chunks = chunked.pop(gpu_paste.DEVICE_OUTPUT_CHUNKS_KEY)
    assert isinstance(chunks, torch.Tensor), type(chunks)
    assert chunks.dtype == torch.float32 and chunks.device.type == "cpu"
    assert tuple(chunks.shape) == (2, 64, 96, 3), chunks.shape
    assert gpu_paste.DEVICE_CANVAS_KEY not in chunked
    assert list(chunked) == [0, 1]
    assert torch.equal(torch.stack([chunked[0], chunked[1]]), chunks)


@pytest.mark.unit
def test_the_device_modes_keep_the_sequence_output_end_to_end(
    processor_module, gpu_paste, monkeypatch
):
    """AC 3c: the chunked and the full device modes ship the same IMAGE batch.

    A four-frame sequence runs through ``process_image_sequence`` four
    times: the unpatched host run at the factor of one, the full mode
    forced at one, then both device modes forced at four.  Within one
    factor the final IMAGE tensors are identical, byte for byte.

    Fix 1.2 (2026-10-30): the corpus's fourth frame used to be gray
    (``frames[3] = 0.4``) and pinned the vendored-fallback row across the
    modes; with the gray skip gone it is batch-eligible, so all four rows
    ride the batched core and the whole-tensor comparisons cover the gray
    row too (all assemblies share the reworked ``_paste_chunk_roi``).
    Every frame batched, the device modes take the reuse fast path (fix
    1.1, the buffer alias) - the canvas / chunk buffer ships as the final
    IMAGE with no fresh whole-clip allocation.  The mixed corpus with a
    real fallback row stays pinned by the NaN-landmark pin below.  The
    factor-4 host run is **not** compared: its background is the cv2
    Lanczos while the device modes use the owner-accepted device Lanczos
    - a documented tolerance route, not a drift.
    """

    def run(final_upscale_factor, canvas_hw, mode):
        landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
        face_helper = _ParseFaceHelper(
            parse_net=_StubParseNet(seed=277), landmarks=landmarks)
        frames = torch.rand(
            (4, 64, 96, 3), generator=torch.Generator().manual_seed(281))
        frames[3] = 0.4  # the gray frame: batch-eligible since fix 1.2
        if mode is not None:
            monkeypatch.setattr(
                gpu_paste, "device_output_mode", lambda *a, **k: mode)
        output, _ = _run_sequence(
            processor_module, face_helper, frames,
            final_upscale_factor=final_upscale_factor)
        return output

    host_f1 = run(1.0, (64, 96), None)
    full_f1 = run(1.0, (64, 96), "full")
    full_f4 = run(4.0, (256, 384), "full")
    chunked_f4 = run(4.0, (256, 384), "chunked")

    assert tuple(host_f1.shape) == (4, 64, 96, 3), host_f1.shape
    assert torch.equal(full_f1, host_f1), (
        "the forced-full run drifted from the host run at the factor of one"
    )

    assert tuple(full_f4.shape) == (4, 256, 384, 3), full_f4.shape
    assert full_f4.dtype == torch.float32
    assert torch.equal(chunked_f4, full_f4), (
        "the chunked mode drifted from the full mode at the factor of four"
    )


@pytest.mark.unit
def test_the_paste_bar_ticks_every_frame_exactly_once_across_the_modes(
    processor_module, gpu_paste, monkeypatch
):
    """The paste bar totals ``num_input_frames`` in every mode (fix 1.1, 2026-10-30).

    The paste bar's discipline (the owner's fix 1.1): the fallback rows
    tick in the processor's loop, the batched rows tick inside the paste
    core per completed chunk/group.  A five-frame all-eligible corpus runs
    host, forced full and forced chunked: the core ticks the five rows per
    chunk - ``[5]`` in every mode (one chunk at the 64x96 canvas, the
    chunk cap is 16).  A five-frame corpus with one None-affine frame
    (``estimate_affines`` patched to return ``None`` for the first frame -
    the LMEDS failure of the production route) runs host, full and
    chunked: the core ticks ``[4]``, the loop ticks ``[1]`` for the
    background-only row - the sum is the frame count, nobody ticks twice,
    nobody is missed.  Before the fix the host mode's batched rows were
    never ticked (the cores had no bar) and an all-eligible CPU clip
    finished its bar under its total.
    """
    bars: list = []

    class _RecordingStageBar:
        """The StageBar double: it records every tick the processor makes."""

        def __init__(self, label, total, *, unit="frames",
                     interval_s=None, clock=None, stream=None):
            self.label = label
            self.total = total
            self.ticks: list[int] = []
            self.finishes = 0
            bars.append(self)

        def tick(self, n=1):
            self.ticks.append(n)

        def finish(self):
            self.finishes += 1

    monkeypatch.setattr(processor_module, "StageBar", _RecordingStageBar)

    def paste_bar():
        return next(
            bar for bar in bars
            if bar.label == "Pasting faces and finalizing frames")

    def run(mode, drop_first_affine):
        landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
        face_helper = _ParseFaceHelper(
            parse_net=_StubParseNet(seed=353), landmarks=landmarks)
        if drop_first_affine:
            real_estimate = processor_module.estimate_affines

            def one_none(landmark_list, template):
                result = list(real_estimate(landmark_list, template))
                result[0] = None
                return result

            monkeypatch.setattr(processor_module, "estimate_affines", one_none)
        frames = torch.rand(
            (5, 64, 96, 3), generator=torch.Generator().manual_seed(359))
        if mode is not None:
            monkeypatch.setattr(
                gpu_paste, "device_output_mode", lambda *a, **k: mode)
        output, _ = _run_sequence(
            processor_module, face_helper, frames, final_upscale_factor=1.0)
        return output

    output = run(None, False)
    assert tuple(output.shape) == (5, 64, 96, 3), output.shape
    assert paste_bar().total == 5
    assert paste_bar().ticks == [5], paste_bar().ticks
    assert paste_bar().finishes == 1

    for mode in ("full", "chunked"):
        bars.clear()
        run(mode, False)
        assert paste_bar().ticks == [5], (mode, paste_bar().ticks)
        assert paste_bar().finishes == 1

    for mode in (None, "full", "chunked"):
        bars.clear()
        output = run(mode, True)
        assert tuple(output.shape) == (5, 64, 96, 3), output.shape
        # the core ticks the four batched rows, the loop the background-only
        # row - the bar's sum is the frame count in every mode
        assert paste_bar().ticks == [4, 1], (mode, paste_bar().ticks)
        assert paste_bar().finishes == 1


@pytest.mark.unit
def test_the_nan_landmark_short_circuit_keeps_the_device_tails_aligned(
    processor_module, gpu_paste, monkeypatch
):
    """The NaN-landmark short-circuit records its fallback row (the device tails).

    The short-circuit of the paste loop (a frame whose smoothed landmarks
    are NaN) appends its output but never recorded the frame in
    ``fallback_indices`` - in a device-output mode the tail's indexed
    assignment then length-mismatches and raises.  The production route
    cannot reach the branch today: ``interpolate_sequence`` fills the NaN
    landmarks (``np.interp`` flat-fills the edges too), so the branch is
    defensive.  The pin forces a NaN landmark through - the interpolation
    patched to the identity, the first detection emitting NaN - so the
    branch runs and the three output modes stay byte-for-byte equal.
    """
    monkeypatch.setattr(processor_module, "interpolate_sequence", lambda seq: seq)

    def run(mode):
        landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
        face_helper = _ParseFaceHelper(
            parse_net=_StubParseNet(seed=311), landmarks=landmarks)
        detections = {"n": 0}
        original = face_helper.get_face_landmarks_5

        def landmarks_with_nan(**kwargs):
            result = original(**kwargs)
            detections["n"] += 1
            if detections["n"] == 1:
                face_helper.all_landmarks_5 = [np.full_like(landmarks, np.nan)]
            return result

        face_helper.get_face_landmarks_5 = landmarks_with_nan
        if mode is not None:
            monkeypatch.setattr(
                gpu_paste, "device_output_mode", lambda *a, **k: mode)
        frames = torch.rand(
            (24, 64, 96, 3), generator=torch.Generator().manual_seed(313))
        output, _ = _run_sequence(
            processor_module, face_helper, frames, final_upscale_factor=1.0)
        return output

    host = run(None)
    full = run("full")
    chunked = run("chunked")
    assert tuple(host.shape) == (24, 64, 96, 3), host.shape
    assert torch.equal(full, host), (
        "the forced-full run drifted from the host run with a NaN landmark frame"
    )
    assert torch.equal(chunked, full), (
        "the chunked mode drifted from the full mode with a NaN landmark frame"
    )


@pytest.mark.unit
def test_the_none_affine_fallback_stays_background_only(
    processor_module, gpu_paste, monkeypatch
):
    """A None-affine frame stops taking the vendored paste (fix 1.1, 2026-10-30).

    The owner's critical fix: a frame whose affine matrix is ``None`` (the
    LMEDS failure of ``estimate_affines``) with valid landmarks used to
    reach the vendored paste - which handles a ``None`` inverse gracefully
    - and now short-circuits to the background-only row, aligning the
    fallback condition with the batched eligibility.  The pin patches
    ``estimate_affines`` to return ``None`` for the second frame (the
    production route's failure shape), so the frame hits the new first
    disjunct of the short-circuit: the frame never touches the face
    helper (no ``add_restored_face``, no vendored paste) and its output
    row is the background (the converted raw frame, non-zero - the stub
    paste returns zeros), recorded in ``fallback_indices`` - the three
    output modes stay byte-for-byte equal (the device tails aligned).
    """

    def run(mode):
        landmarks = (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])
        face_helper = _ParseFaceHelper(
            parse_net=_StubParseNet(seed=367), landmarks=landmarks)
        real_estimate = processor_module.estimate_affines

        def one_none(landmark_list, template):
            result = list(real_estimate(landmark_list, template))
            result[1] = None
            return result

        monkeypatch.setattr(processor_module, "estimate_affines", one_none)
        if mode is not None:
            monkeypatch.setattr(
                gpu_paste, "device_output_mode", lambda *a, **k: mode)
        frames = torch.rand(
            (2, 64, 96, 3), generator=torch.Generator().manual_seed(373))
        output, _ = _run_sequence(
            processor_module, face_helper, frames, final_upscale_factor=1.0)
        return output, face_helper

    host, host_helper = run(None)
    full, _ = run("full")
    chunked, _ = run("chunked")
    assert tuple(host.shape) == (2, 64, 96, 3), host.shape
    assert torch.equal(full, host), (
        "the forced-full run drifted from the host run with a None-affine frame"
    )
    assert torch.equal(chunked, full), (
        "the chunked mode drifted from the full mode with a None-affine frame"
    )
    # the None-affine frame never reaches the helper: no helper setup, no
    # vendored paste - the row is the background, not the stub's zeros
    assert "paste_faces_to_input_image" not in host_helper.calls, host_helper.calls
    assert "add_restored_face" not in host_helper.calls, host_helper.calls
    assert host[1].abs().sum() > 0, "the background-only row must not be zeros"


# ---------------------------------------------------------------------------
# The spandrel-descriptor helpers (keep-fix5-implement2, 2026-11-03)
# ---------------------------------------------------------------------------


class _ReportedParam:
    """The object the lifecycle manager's device probe reads ``.device`` from."""

    def __init__(self, device: torch.device) -> None:
        self.device = device


class _InnerDescriptorModel:
    """The callable, probe-able inner ``.model`` of the descriptor stubs.

    keep-fix5-implement2 (2026-11-03): the re-entrant
    ``_spandrel_on_device`` probes ``next(m.model.parameters()).device`` -
    this model reports the device the owner's ``.to`` last recorded (a real
    descriptor's weights follow its ``.to``), so a nested manager sees the
    outer one's move exactly like a real descriptor would.
    """

    def __init__(self, owner) -> None:
        self.owner = owner

    def parameters(self):
        return iter((_ReportedParam(self.owner.reported),))

    def __call__(self, samples):
        self.owner.calls.append(tuple(samples.shape))
        return self.owner.run(samples)


class _DescriptorStub:
    """A spandrel ``ImageModelDescriptor`` stand-in for the implement2 pins.

    The inner ``.model`` is callable and probe-able (the shape
    ``_spandrel_on_device`` and ``_call_spandrel`` expect of a real
    descriptor); the stub itself is **not** callable, so
    ``_call_spandrel`` takes its ``model.model(x)`` fallback - the same
    route the plain stubs of the pixel pins ride.  The default weights
    device ``"meta"`` differs from the CPU tests' device, so a manager
    wrap shows its moves in ``moves``.
    """

    scale = 2.0

    def __init__(self, device: str = "meta") -> None:
        self.calls: list[tuple] = []
        self.moves: list[torch.device] = []
        self.reported = torch.device(device)
        self.model = _InnerDescriptorModel(self)

    def to(self, device, *args, **kwargs):
        device = torch.device(device)
        self.moves.append(device)
        self.reported = device
        return self

    def run(self, samples):
        return F.interpolate(samples, scale_factor=self.scale, mode="nearest")


class _OomOnceDescriptorStub(_DescriptorStub):
    """The OOM script of the tile-shrink pin: the first model call OOMs."""

    def __init__(self) -> None:
        super().__init__()
        self.ooms = 0

    def run(self, samples):
        if not self.ooms:
            self.ooms += 1
            raise torch.cuda.OutOfMemoryError("the scripted first tile")
        return F.interpolate(samples, scale_factor=self.scale, mode="nearest")


class _AlwaysOomDescriptorStub(_DescriptorStub):
    """The OOM script of the re-raise pin: every model call OOMs."""

    def run(self, samples):
        raise torch.cuda.OutOfMemoryError("the scripted every-tile OOM")


class _BrokenRestoreStub(_DescriptorStub):
    """The restore script: the offload ``.to`` raises."""

    def to(self, device, *args, **kwargs):
        device = torch.device(device)
        if device == torch.device("meta"):
            raise RuntimeError("the scripted offload failure")
        return super().to(device, *args, **kwargs)


@pytest.mark.unit
def test_the_spandrel_lifecycle_moves_once_and_restores_the_prior_device(gpu_paste):
    """The re-entrant manager (keep-fix5-implement2, 2026-11-03): one move per stage.

    The stage-level wrap moves the descriptor to the device **once**,
    restores its **prior** device on exit, a nested manager sees the
    weights already on the device and is a no-op (the inner manager of
    ``_tiled_scale_and_resize_batch`` /
    ``upscale_faces_batch_gpu`` under the stage wrap), and a model already
    on the device is never touched.  ``"meta"`` stands in for the offload
    device the production weights start on.
    """
    device = torch.device("cpu")
    model = _DescriptorStub("meta")
    with gpu_paste._spandrel_on_device(device, model):
        assert model.moves == [torch.device("cpu")], model.moves
        # the nested manager (the per-helper wrap): no second move
        with gpu_paste._spandrel_on_device(device, model):
            assert model.moves == [torch.device("cpu")], model.moves
        assert model.moves == [torch.device("cpu")], model.moves
    # the exit restored the PRIOR device (the offload device), once
    assert model.moves == [torch.device("cpu"), torch.device("meta")], model.moves

    already_there = _DescriptorStub("cpu")
    with gpu_paste._spandrel_on_device(device, already_there):
        assert already_there.moves == [], already_there.moves
    assert already_there.moves == [], already_there.moves


@pytest.mark.unit
def test_the_spandrel_lifecycle_filters_none_models(gpu_paste):
    """The ``None`` models of an optional stage (no bg model, no face model)
    are filtered before the device probe - a ``None`` never reaches
    ``getattr(m, 'model', m)`` and the present model still moves."""
    model = _DescriptorStub("meta")
    with gpu_paste._spandrel_on_device(torch.device("cpu"), None, model, None):
        assert model.moves == [torch.device("cpu")], model.moves
    assert model.moves == [torch.device("cpu"), torch.device("meta")], model.moves


@pytest.mark.unit
def test_the_spandrel_lifecycle_swallows_a_restore_failure(gpu_paste):
    """A failed offload must not mask the body's exception (the plan's
    ``except Exception: pass``) - the body's ``RuntimeError`` propagates,
    the restore's ``RuntimeError`` is swallowed."""
    model = _BrokenRestoreStub("meta")
    with pytest.raises(RuntimeError, match="the scripted body failure"):
        with gpu_paste._spandrel_on_device(torch.device("cpu"), model):
            assert model.moves == [torch.device("cpu")], model.moves
            raise RuntimeError("the scripted body failure")
    # the restore ran (and failed) - the move is the only recorded one
    assert model.moves == [torch.device("cpu")], model.moves


@pytest.mark.unit
def test_the_spandrel_lifecycle_keeps_the_body_exception_when_the_restore_holds(gpu_paste):
    """The complement of the swallow pin: a clean restore still lets the
    body's exception out of the ``with``."""
    model = _DescriptorStub("meta")
    with pytest.raises(RuntimeError, match="the scripted body failure"):
        with gpu_paste._spandrel_on_device(torch.device("cpu"), model):
            raise RuntimeError("the scripted body failure")
    assert model.moves == [torch.device("cpu"), torch.device("meta")], model.moves


@pytest.mark.unit
def test_call_spandrel_calls_the_descriptor_and_falls_back_to_the_raw_module(gpu_paste):
    """``_call_spandrel`` (keep-fix5-implement2, 2026-11-03): a callable model
    (a spandrel descriptor - its ``__call__`` applies the
    ``size_requirements`` padding) is called directly; a non-callable
    ``.model`` wrapper (the plain stubs) keeps the raw ``model.model(x)``
    route - the fallback the existing pixel pins ride unchanged."""
    class _CallableDescriptor:
        def __call__(self, samples):
            return samples + 1.0

    class _RawModuleWrapper:
        def model(self, samples):
            return samples * 2.0

    sample = torch.zeros(1, 3, 4, 4)
    assert torch.equal(
        gpu_paste._call_spandrel(_CallableDescriptor(), sample), sample + 1.0
    )
    assert torch.equal(
        gpu_paste._call_spandrel(_RawModuleWrapper(), sample), sample * 2.0
    )


@pytest.mark.unit
def test_the_fp16_gate_follows_supports_half_and_the_device(gpu_paste):
    """``_spandrel_supports_fp16`` (keep-fix5-implement2, 2026-11-03):
    ``False`` on a non-CUDA device whatever the flag says (the CPU path
    runs FP32), the descriptor's ``supports_half`` on CUDA, and the
    duck-typed ``getattr`` default ``True`` for a descriptor without the
    flag."""
    assert gpu_paste._spandrel_supports_fp16(object(), torch.device("cpu")) is False
    assert (
        gpu_paste._spandrel_supports_fp16(
            types.SimpleNamespace(supports_half=False), torch.device("cpu")
        )
        is False
    )
    assert (
        gpu_paste._spandrel_supports_fp16(
            types.SimpleNamespace(supports_half=True), torch.device("cuda:0")
        )
        is True
    )
    assert (
        gpu_paste._spandrel_supports_fp16(
            types.SimpleNamespace(supports_half=False), torch.device("cuda:0")
        )
        is False
    )
    # a descriptor without the flag: the duck-typed default is True
    assert gpu_paste._spandrel_supports_fp16(object(), torch.device("cuda:0")) is True


@pytest.mark.unit
def test_the_tile_shrink_survives_one_oom_and_stays_sticky_across_frames(
    gpu_paste, monkeypatch
):
    """The OOM-resilient tile shrink (keep-fix5-implement2, 2026-11-03).

    A descriptor whose first model call raises
    ``torch.cuda.OutOfMemoryError``: the 512 tile halves to 256 and the
    frame completes; the next frame **starts at 256** (the tile geometry
    is sticky across frames - a card that OOMed at 512 does not retry
    512); the shrink couples the overlap to the tile
    (``min(overlap, tile_size // 4)``).  The recorded
    ``(tile_x, overlap)`` script is the pin.
    """
    seen: list[tuple[int, int]] = []

    def recording_tiled_scale(sample, function, tile_x=512, tile_y=512, overlap=64, **kwargs):
        seen.append((int(tile_x), int(overlap)))
        return function(sample)

    monkeypatch.setattr("comfy.utils.tiled_scale", recording_tiled_scale)

    frames = smooth_frames_u8(2, (64, 96), seed=61)
    model = _OomOnceDescriptorStub()

    out = gpu_paste._tiled_scale_and_resize_batch(
        model, frames, (128, 192), torch.device("cpu"), False
    )

    assert out.shape == (2, 128, 192, 3), out.shape
    # frame 0: 512 OOMs -> 256 completes; frame 1: sticky 256, no retry at 512
    assert seen == [(512, 64), (256, 64), (256, 64)], seen
    assert model.ooms == 1, model.ooms
    # the re-entrant manager moved the descriptor (weights started on
    # 'meta') and restored it - the standalone lifecycle of the full mode
    assert model.moves == [torch.device("cpu"), torch.device("meta")], model.moves
    # three model calls: the failed 512 attempt + one completed call per frame
    assert len(model.calls) == 3, model.calls


@pytest.mark.unit
def test_the_tile_shrink_re_raises_below_the_128_floor(gpu_paste, monkeypatch):
    """A tile that OOMs at every size re-raises the
    ``torch.cuda.OutOfMemoryError`` once the halving passes the 128 floor
    - the pipeline fails loudly instead of shrinking forever (the
    ``(tile_x, overlap)`` script also pins the 128 stop: the last tried
    tile is 128 with ``min(64, 128 // 4) = 32`` overlap)."""
    seen: list[tuple[int, int]] = []

    def recording_tiled_scale(sample, function, tile_x=512, tile_y=512, overlap=64, **kwargs):
        seen.append((int(tile_x), int(overlap)))
        return function(sample)

    monkeypatch.setattr("comfy.utils.tiled_scale", recording_tiled_scale)

    frames = smooth_frames_u8(1, (64, 96), seed=62)
    model = _AlwaysOomDescriptorStub()

    with pytest.raises(torch.cuda.OutOfMemoryError):
        gpu_paste._tiled_scale_and_resize_batch(
            model, frames, (128, 192), torch.device("cpu"), False
        )
    assert seen == [(512, 64), (256, 64), (128, 32)], seen
    # the failed frame still restored the descriptor
    assert model.moves == [torch.device("cpu"), torch.device("meta")], model.moves


@pytest.mark.unit
def test_the_paste_wrap_moves_the_face_upscaler_to_the_device_and_back(
    gpu_paste, gpu_ops, face_helper
):
    """The stage-level wrap of ``paste_faces_batch`` (keep-fix5-implement2,
    2026-11-03): a host-output paste with a face upscaler whose weights
    start on the offload device ('meta' here) records the ``.to`` move to
    the compute device and the restore back - one move per paste, not per
    face - and the upscaled ROI paste still runs (the model saw the
    512 face)."""
    frames = smooth_frames_u8(1, (360, 640), seed=63)
    landmarks = synthetic_landmarks(1, (360, 640), seed=64)
    rng = np.random.default_rng(65)
    faces = rng.integers(0, 256, size=(1, 512, 512, 3), dtype=np.uint8)
    parse_net = _StubParseNet(seed=66)
    affines = gpu_ops.estimate_affines(landmarks, face_helper.face_template)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)
    model = _DescriptorStub("meta")

    pasted = gpu_paste.paste_faces_batch(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=frames,
        device=torch.device("cpu"),
        parse_net=parse_net,
        face_upscale_model=model,
    )

    assert len(pasted) == 1
    # the stage wrap: one move to the device, one restore to the prior
    # device - the inner manager of ``upscale_faces_batch_gpu`` saw the
    # weights already on it and moved nothing
    assert model.moves == [torch.device("cpu"), torch.device("meta")], model.moves
    # the face upscaler ran on the 512 face (the ROI warp sampled the x2)
    assert model.calls == [(1, 3, 512, 512)], model.calls
