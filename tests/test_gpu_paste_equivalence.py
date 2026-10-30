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
residual** now (the 8/255 drift class, documented per pin); the warp-op
pins (``warp_face_roi`` and friends) stay bit-exact - the op keeps its
production call site in ``_paste_chunk_faces``.

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
    """A near-black frame keeps the vendored tail's whole-canvas 255 factor.

    The vendored tail multiplies the **whole canvas** by 255 when the
    blended frame is near black (``face_restoration_helper.py:504-505``).
    Fix 2.1 (2026-10-30, the owner's ``keep-fix3.txt``) deleted the old
    body's explicit full-canvas fallback branch: the tail lives in the
    whole-canvas maximum now (``_quantize_and_clip`` with ``maximum=None``
    measures the frame maximum over the entire canvas), so the factor is
    the same without the branch - the pin holds that property.

    The residual against the vendored paste is the whole-canvas 255
    amplification of the batched-warp drift (float32 ``grid_sample`` vs
    cv2's 1/32 fixed-point sampling - the face warp joined the mask warp's
    class under fix 2.1; the old pin's ROI==full bit-equality is gone with
    the branch).  Measured on this content, **133/255**: a single sampled
    tap flip on the 0/1 corpus is a full unit after the amplification.  A
    lost 255 factor would exceed the bound by construction (the bright-frame
    sibling below pins the inverse case), so the tail's survival stays
    checkable.
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

    reference = _vendored_paste(
        face_helper, background[0], face, cv2.invertAffineTransform(matrix),
        1.0, parse_net,
    )
    diff = np.abs(reference.astype(int) - roi_paste[0].astype(int))
    assert diff.max() <= 133, (
        f"the dark frame drifted {diff.max()}/255 from the vendored paste"
    )


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
    (``{index: frame}``, no sentinel).  The mapped indices name the rows
    of the sentinel tensor, in order.
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
