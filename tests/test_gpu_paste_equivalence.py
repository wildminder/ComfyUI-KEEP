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
pins are **bit-exact** - the ROI path has to reproduce the crop of the
full-canvas result byte for byte (a tolerance would be a QA failure, not
an acceptance).
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
def test_the_roi_paste_is_bit_identical_to_the_full_canvas_paste(
    gpu_paste, corpus, position
):
    """The whole ROI paste == the whole full-canvas paste, byte for byte.

    The ROI path is the shipped path (``use_roi=True``); the full-canvas
    path (``use_roi=False``) is the reference the ROI path replaces.  The
    pin is bit-exact: the face warp, the mask warp, the blend and the host
    assembly reproduce the full-canvas result on every pixel.

    The two frames of the chunk have **different** ROIs (a different
    scale), so the per-frame ROI of the chunk - and the padded batched
    mask warp that serves it - is exercised.
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
        assert np.array_equal(roi_frame, full_frame), (
            f"frame {index}: the ROI paste drifted from the full-canvas paste "
            f"on the {corpus} corpus at the {position} ROI"
        )


@pytest.mark.unit
def test_the_roi_transfers_are_roi_sized(gpu_paste):
    """The transfer counters are ROI-sized, not canvas-sized.

    The counters are the product's own accounting
    (``PASTE_TRANSFER_STATS``) - the probe reads them for the before/after
    table, so the claim "no full-canvas H2D / D2H of the background and of
    the blended canvas" is checkable here: the background patch and the
    blended patch that travel are the ROI ones.
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
    roi_bytes = 0
    for matrix in matrices:
        x0, y0, x1, y1 = gpu_paste.face_roi_box(matrix, ROI_CANVAS_HW)
        roi_bytes += (x1 - x0) * (y1 - y0) * 3

    assert stats["frames"] == 2
    assert stats["roi_frames"] == 2
    assert stats["fallback_frames"] == 0
    assert stats["canvas_bytes"] == height * width * 3
    assert stats["h2d_bytes"] == roi_bytes, stats
    assert stats["d2h_bytes"] == roi_bytes, stats
    assert stats["face_h2d_bytes"] == 2 * 512 * 512 * 3, stats
    # the ROI transfers are a small part of the canvas transfers
    assert stats["h2d_bytes"] < 2 * stats["canvas_bytes"], stats
    assert stats["d2h_bytes"] < 2 * stats["canvas_bytes"], stats


@pytest.mark.unit
def test_a_dark_frame_takes_the_full_canvas_fallback(gpu_paste, face_helper):
    """A near-black frame falls back to the full-canvas path.

    The vendored tail multiplies the **whole canvas** by 255 when the
    blended frame is near black (``face_restoration_helper.py:504-505``) -
    a frame-global operation the ROI cannot reproduce.  Such a frame falls
    back to the full-canvas path.

    The pin: the fallback counter is on, the ROI paste is bit-identical to
    the full-canvas paste and the output matches the vendored paste.  The
    residual against the vendored paste is the mask-warp residual of the
    other pins (the vendored mask is cv2's 1/32-quantised warp, the
    product's is the float32 grid sample) - **not** a residual of the ROI
    path, which is bit-identical to the full-canvas paste.  The near-black
    branch multiplies the blend by 255, so that residual is amplified:
    measured on this content, **5/255** (the un-amplified residual is
    5/255/255, well inside the 8/255 the noise pin of the batched paste
    measures).
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
    assert stats["fallback_frames"] == 1, stats
    assert stats["roi_frames"] == 0, stats

    full_paste = gpu_paste.paste_faces_batch(use_roi=False, **kwargs)
    assert np.array_equal(roi_paste[0], full_paste[0]), (
        "the dark frame's ROI paste drifted from the full-canvas paste"
    )

    reference = _vendored_paste(
        face_helper, background[0], face, cv2.invertAffineTransform(matrix),
        1.0, parse_net,
    )
    diff = np.abs(reference.astype(int) - roi_paste[0].astype(int))
    assert diff.max() <= 5, (
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
    assert np.array_equal(roi_paste[0], full_paste[0]), (
        "the dark ROI's paste drifted from the full-canvas paste"
    )
    # the pixels outside the ROI keep their background value
    outside = np.ones((height, width), dtype=bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(roi_paste[0][outside], background[0][outside]), (
        "the pixels outside the ROI did not keep their background value"
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
