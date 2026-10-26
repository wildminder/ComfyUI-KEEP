"""The measured old-vs-new equivalence of the batched GPU ops.

The video pipeline redesign (design section 7 of
``.dev/docs/2026-10-25-keep-gpu-video-pipeline.md``) replaces the per-frame
cv2 / numpy loops of ``process_image_sequence`` with one batched torch op
each.  Every op is pinned here against the vendored reference it replaces,
with the tolerance the runs measured - measured, not assumed (the pattern of
``tests/test_image_op_equivalence.py``).

Bound provenance:

- the converters, ``tensor2img_batch`` and the parse colormap are the same
  elementwise chains as their per-frame references - pinned **bit-exact**;
- the warps replace ``cv2.warpAffine``, which samples in 1/32 px fixed point
  (``INTER_TAB_SIZE``) and weights, while torch samples in float32: the
  residual per pixel is bounded by ``local gradient / 64`` (the coordinate
  quantization) plus the fixed-point weight rounding - the tests measure it
  on smooth content and on adversarial noise and pin the measured bound;
- the parse blur keeps the exact ``cv2.getGaussianKernel`` kernel and the
  ``BORDER_REFLECT101`` extension, so its residual against the two cv2 calls
  is float summation order only;
- the ParseNet forward is pinned batch-invariant (a batched forward must
  return the per-frame pixels).
"""

from __future__ import annotations

import importlib

import cv2
import numpy as np
import pytest
import torch

#: The vendored 5-point 512 template (``face_restoration_helper.py:52-53``).
FACE_TEMPLATE = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]]
)


@pytest.fixture(scope="module")
def gpu_ops():
    return importlib.import_module("ComfyUI_KEEP.src.keep_gpu_ops")


@pytest.fixture(scope="module")
def image_utils():
    return importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")


@pytest.fixture(scope="module")
def img_util():
    return importlib.import_module("ComfyUI_KEEP.modules.deps.basicsr.utils.img_util")


@pytest.fixture(scope="module")
def helper_module():
    return importlib.import_module("ComfyUI_KEEP.modules.deps.facelib.utils.face_restoration_helper")


@pytest.fixture()
def face_helper(helper_module, monkeypatch):
    """A vendored ``FaceRestoreHelper`` without its weight-backed models.

    The detection / parsing initializations are stubbed (the weights are not
    on this machine and the affine / paste math under test does not touch
    them), so the reference methods run the real vendored code.
    """
    monkeypatch.setattr(helper_module, "init_detection_model", lambda *a, **k: object())
    monkeypatch.setattr(helper_module, "init_parsing_model", lambda *a, **k: object())
    return helper_module.FaceRestoreHelper(
        upscale_factor=1, face_size=512, crop_ratio=(1, 1),
        det_model="retinaface_resnet50", save_ext="png", use_parse=False,
        device="cpu", model_rootpath="unused",
    )


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
    """Low-frequency content: the realistic worst case for the warp residual."""
    rng = np.random.default_rng(seed)
    height, width = frame_hw
    small = rng.uniform(0.0, 255.0, size=(num_frames, 12, 22, 3)).astype(np.float32)
    frames = np.stack(
        [cv2.resize(small[i], (width, height), interpolation=cv2.INTER_CUBIC) for i in range(num_frames)]
    )
    return np.clip(frames, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# The batched converters (design sections 7.1, 7.4, 7.8)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_comfy_batch_to_bgr_u8_is_bit_exact_against_the_per_frame_loop(image_utils):
    """The batch conversion equals the per-frame ``comfy_image_to_cv2`` loop."""
    tensor = torch.rand((4, 16, 24, 3), generator=torch.Generator().manual_seed(7))

    batched = image_utils.comfy_batch_to_bgr_u8(tensor)
    per_frame = np.stack(
        [image_utils.comfy_image_to_cv2(tensor[i : i + 1]) for i in range(tensor.shape[0])]
    )

    assert batched.shape == (4, 16, 24, 3)
    assert batched.dtype == np.uint8
    assert np.array_equal(batched, per_frame)


@pytest.mark.unit
def test_comfy_batch_to_bgr_u8_matches_the_former_numpy_implementation(image_utils):
    """The scale / cast / flip reproduce the former numpy + cv2 path exactly."""
    tensor = torch.rand((3, 9, 11, 3), generator=torch.Generator().manual_seed(13))

    batched = image_utils.comfy_batch_to_bgr_u8(tensor)

    reference = np.stack(
        [
            cv2.cvtColor((tensor[i].numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            for i in range(tensor.shape[0])
        ]
    )
    assert np.array_equal(batched, reference)


@pytest.mark.unit
def test_bgr_u8_batch_to_comfy_is_bit_exact_against_the_per_frame_loop(image_utils):
    """The batch conversion equals the per-frame ``cv2_to_comfy_image`` loop.

    Both accepted inputs are covered: the numpy stack of the per-frame paste
    outputs and the torch uint8 tensor the batched blend produces on the
    device.
    """
    rng = np.random.default_rng(11)
    frames = rng.integers(0, 256, size=(5, 12, 18, 3), dtype=np.uint8)

    from_numpy = image_utils.bgr_u8_batch_to_comfy(frames)
    from_torch = image_utils.bgr_u8_batch_to_comfy(torch.from_numpy(frames.copy()))
    per_frame = torch.cat([image_utils.cv2_to_comfy_image(frames[i]) for i in range(5)], dim=0)

    assert from_numpy.shape == (5, 12, 18, 3)
    assert from_numpy.dtype == torch.float32
    assert torch.equal(from_numpy, from_torch)
    assert torch.equal(from_numpy, per_frame)


@pytest.mark.unit
def test_bgr_u8_batch_to_normalized_is_bit_exact_against_img2tensor(image_utils, img_util):
    """The batched crop tail equals ``img2tensor`` + the normalize replacement.

    The reference is the per-frame chain of ``keep_processor.py:311-313``:
    ``img2tensor(face / 255., bgr2rgb=True, float32=True)`` (the numpy
    float64 division, then the float32 cast) and the in-place
    ``(t - 0.5) / 0.5``.  The float64 division of the batch form reproduces
    the double rounding bit for bit.
    """
    rng = np.random.default_rng(23)
    faces = rng.integers(0, 256, size=(3, 3, 48, 64), dtype=np.uint8)

    batched = image_utils.bgr_u8_batch_to_normalized(torch.from_numpy(faces))
    reference = torch.stack(
        [
            img_util.img2tensor(
                faces[i].transpose(1, 2, 0) / 255.0, bgr2rgb=True, float32=True
            ).sub_(0.5).div_(0.5)
            for i in range(3)
        ]
    )

    assert batched.dtype == torch.float32
    assert torch.equal(batched, reference)


# ---------------------------------------------------------------------------
# Affine capture (design section 7.3) - identity by construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_estimate_affines_match_the_vendored_align_warp_face(gpu_ops, face_helper):
    """The captured matrices are the ones ``align_warp_face`` appends."""
    landmarks = synthetic_landmarks(3, (720, 1280), seed=5)
    # the vendored method reads ``input_img`` for the crop warp (any image
    # works - only the matrices are under test here)
    face_helper.input_img = np.zeros((128, 128, 3), dtype=np.uint8)
    face_helper.all_landmarks_5 = [landmarks[i] for i in range(3)]
    face_helper.align_warp_face()
    captured = gpu_ops.estimate_affines(landmarks, face_helper.face_template)

    assert len(captured) == 3
    for reference, matrix in zip(face_helper.affine_matrices, captured):
        assert matrix is not None
        assert matrix.dtype == np.float64
        assert np.array_equal(reference, matrix)


@pytest.mark.unit
def test_estimate_affines_preserves_the_lmeds_none(gpu_ops, face_helper):
    """Degenerate landmarks keep the ``None`` of the vendored ``:315-319`` branch."""
    degenerate = np.zeros((5, 2), dtype=np.float64)

    face_helper.input_img = np.zeros((128, 128, 3), dtype=np.uint8)
    face_helper.all_landmarks_5 = [degenerate]
    face_helper.align_warp_face()
    captured = gpu_ops.estimate_affines([degenerate], face_helper.face_template)

    assert captured == [None]
    assert face_helper.affine_matrices == [None]


@pytest.mark.unit
def test_inverse_affines_match_the_vendored_get_inverse_affine(gpu_ops, face_helper):
    """The inverse warps equal ``get_inverse_affine`` for upscale 1 and 2."""
    landmarks = synthetic_landmarks(3, (360, 640), seed=9)
    affines = gpu_ops.estimate_affines(landmarks, face_helper.face_template)

    for upscale in (1, 2):
        face_helper.upscale_factor = upscale
        face_helper.affine_matrices = list(affines)
        face_helper.get_inverse_affine(None)
        captured = gpu_ops.inverse_affines(affines, upscale_factor=upscale)
        for reference, matrix in zip(face_helper.inverse_affine_matrices, captured):
            assert np.array_equal(reference, matrix)


# ---------------------------------------------------------------------------
# The batched warps (design sections 7.4 and 7.7)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_warp_affine_batch_is_exact_for_identity_and_integer_shift(gpu_ops):
    """The normalized theta math lands cv2-exactly on aligned geometries.

    An identity matrix and a one-pixel integer translation sample exact pixel
    centers, where cv2's 1/32 px fixed point and the torch float grid agree
    to the last bit - so the quantized outputs must be pixel-identical.
    """
    rng = np.random.default_rng(31)
    frames = rng.integers(0, 256, size=(2, 3, 48, 64), dtype=np.uint8)
    source = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    shift = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    warped = gpu_ops.warp_affine_batch(source, [identity, shift], (64, 48))
    warped_u8 = warped.round().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    reference = np.stack(
        [cv2.warpAffine(frames[i], matrix, (64, 48)) for i, matrix in enumerate((identity, shift))]
    )

    assert np.array_equal(warped_u8, reference)


@pytest.mark.unit
def test_warp_affine_batch_rejects_mismatched_and_none_matrices(gpu_ops):
    """A None matrix or a wrong count fails loudly (the no-face frames are
    filtered by the caller, like today's zero-tensor branch)."""
    source = torch.rand((2, 3, 16, 16))
    with pytest.raises(ValueError):
        gpu_ops.warp_affine_batch(source, [np.eye(2, 3)], (16, 16))
    with pytest.raises(ValueError):
        gpu_ops.warp_affine_batch(source, [np.eye(2, 3), None], (16, 16))


@pytest.mark.unit
def test_crop_faces_batch_matches_cv2_on_smooth_content(gpu_ops):
    """The batched crop vs the vendored warp: smooth content.

    Measured over 4 frames of 1280x720 low-frequency content with jittered
    landmark geometries: the max residual is **1/255** (2.1% of the pixels
    differ at all) - cv2 samples in 1/32 px fixed point, torch in float32.
    """
    frames = smooth_frames_u8(4, (720, 1280), seed=3)
    landmarks = synthetic_landmarks(4, (720, 1280), seed=4)
    affines = gpu_ops.estimate_affines(landmarks, FACE_TEMPLATE)
    source = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

    cropped = gpu_ops.crop_faces_batch(source, affines).permute(0, 2, 3, 1).numpy()
    reference = np.stack(
        [
            cv2.warpAffine(
                frames[i], affines[i], (512, 512),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132),
            )
            for i in range(4)
        ]
    )

    assert cropped.dtype == np.uint8
    diff = np.abs(reference.astype(int) - cropped.astype(int))
    assert diff.max() <= 1, f"the batched crop drifted {diff.max()}/255 from cv2"


@pytest.mark.unit
def test_crop_faces_batch_matches_cv2_on_noise_content(gpu_ops):
    """The batched crop vs the vendored warp: adversarial content.

    Pure per-pixel noise is the worst case for cv2's fixed-point sampling
    (gradients up to 255 per pixel).  Measured: max **6/255** (p99.9 = 4,
    62% of the pixels differ by at least 1).  The pinned bound is the
    quantization analysis ceiling of ~2 x 255 / 64.
    """
    rng = np.random.default_rng(6)
    frames = rng.integers(0, 256, size=(4, 720, 1280, 3), dtype=np.uint8)
    landmarks = synthetic_landmarks(4, (720, 1280), seed=7)
    affines = gpu_ops.estimate_affines(landmarks, FACE_TEMPLATE)
    source = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

    cropped = gpu_ops.crop_faces_batch(source, affines).permute(0, 2, 3, 1).numpy()
    reference = np.stack(
        [
            cv2.warpAffine(
                frames[i], affines[i], (512, 512),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132),
            )
            for i in range(4)
        ]
    )

    diff = np.abs(reference.astype(int) - cropped.astype(int))
    assert diff.max() <= 8, f"the batched crop drifted {diff.max()}/255 from cv2"


@pytest.mark.unit
def test_crop_faces_batch_composites_the_vendored_gray_border(gpu_ops):
    """A fully out-of-frame face returns the exact gray constant everywhere.

    The border composite must reproduce cv2's ``BORDER_CONSTANT`` value
    (135, 133, 132) - not the black a zero-padded ``grid_sample`` gives.
    """
    outside = FACE_TEMPLATE + np.array([2000.0, 2000.0])  # the face sits far outside
    affines = gpu_ops.estimate_affines([outside], FACE_TEMPLATE)
    frame = torch.full((1, 3, 64, 64), 200.0)

    cropped = gpu_ops.crop_faces_batch(frame, affines).permute(0, 2, 3, 1).numpy()
    reference = cv2.warpAffine(
        np.full((64, 64, 3), 200, dtype=np.uint8), affines[0], (512, 512),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132),
    )

    assert np.array_equal(cropped[0], reference)
    assert np.all(cropped[0][:, :, 0] == 135)
    assert np.all(cropped[0][:, :, 1] == 133)
    assert np.all(cropped[0][:, :, 2] == 132)


@pytest.mark.unit
def test_warp_affine_batch_matches_cv2_for_the_inverse_paste_warp(gpu_ops, face_helper):
    """The batched inverse warp of the faces vs the vendored paste warp.

    The paste warps the uint8 ``tensor2img`` faces back onto the (upscaled)
    frame with ``cv2.warpAffine(face, inverse_affine, (w_up, h_up))``
    (``face_restoration_helper.py:406``, the default black border).  Both
    upscale factors are measured at the production 720p canvas: the max
    residual is **7/255** for upscale 1 and 2 (18% of the pixels differ).
    """
    rng = np.random.default_rng(8)
    faces = rng.integers(0, 256, size=(3, 3, 512, 512), dtype=np.uint8)
    landmarks = synthetic_landmarks(3, (720, 1280), seed=10)
    affines = gpu_ops.estimate_affines(landmarks, face_helper.face_template)
    source = torch.from_numpy(faces).float()

    for upscale in (1, 2):
        inverses = gpu_ops.inverse_affines(affines, upscale_factor=upscale)
        out_size = (1280 * upscale, 720 * upscale)  # the cv2 (w_up, h_up) order
        warped = (
            gpu_ops.warp_affine_batch(source, inverses, out_size)
            .round().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        )
        faces_hwc = faces.transpose(0, 2, 3, 1)  # cv2 wants (H, W, C)
        reference = np.stack(
            [cv2.warpAffine(faces_hwc[i], inverses[i], out_size) for i in range(3)]
        )
        diff = np.abs(reference.astype(int) - warped.astype(int))
        assert diff.max() <= 8, (
            f"upscale {upscale}: the batched inverse warp drifted {diff.max()}/255 from cv2"
        )


@pytest.mark.unit
def test_warp_affine_batch_matches_cv2_for_the_float_mask_warp(gpu_ops):
    """The float warp (the parse mask path) matches the cv2 float warp.

    The parse mask is warped as float32 with ``INTER_LINEAR`` semantics
    (``face_restoration_helper.py:477``).  cv2's fixed-point path quantizes
    the affine coefficients to 1/1024, so the position error grows with the
    output coordinate - measured at the production 720p canvas:
    **0.0257** mask units on per-pixel-independent content (the worst case)
    and **0.0075** on the smoothed binary mask the pipeline actually warps.
    """
    rng = np.random.default_rng(12)
    uniform = rng.uniform(0.0, 1.0, size=(3, 1, 512, 512)).astype(np.float32)
    binary = (rng.uniform(size=(3, 1, 512, 512)) > 0.5).astype(np.float32) * 255.0
    for i in range(3):  # the production shape: the colormap output, blurred
        blurred = cv2.GaussianBlur(binary[i, 0], (101, 101), 11)
        binary[i, 0] = cv2.GaussianBlur(blurred, (101, 101), 11)
    landmarks = synthetic_landmarks(3, (720, 1280), seed=13)
    affines = gpu_ops.estimate_affines(landmarks, FACE_TEMPLATE)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)

    reference = np.stack(
        [cv2.warpAffine(uniform[i, 0], inverses[i], (1280, 720)) for i in range(3)]
    )[:, None]  # (N, 1, H, W) like the batched form
    warped = gpu_ops.warp_affine_batch(torch.from_numpy(uniform), inverses, (1280, 720)).numpy()
    diff = np.abs(reference - warped).max()
    assert diff <= 0.03, f"the float mask warp drifted {diff:.2e} mask units from cv2"

    reference = np.stack(
        [cv2.warpAffine(binary[i, 0] / 255.0, inverses[i], (1280, 720)) for i in range(3)]
    )[:, None]
    warped = gpu_ops.warp_affine_batch(torch.from_numpy(binary / 255.0), inverses, (1280, 720)).numpy()
    diff = np.abs(reference - warped).max()
    assert diff <= 0.01, f"the smoothed mask warp drifted {diff:.2e} mask units from cv2"


# ---------------------------------------------------------------------------
# tensor2img batch (design section 7.6) - bit-exact
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tensor2img_batch_is_bit_exact_against_the_vendored_loop(gpu_ops, img_util):
    """The batched ``tensor2img`` equals the per-frame vendored call.

    The values overshoot (-1, 1) on purpose so the clamp branch is covered;
    the torch uint8 cast of the rounded values is exact, like the numpy
    ``astype`` of the vendored code.
    """
    tensor = torch.randn((5, 3, 32, 32), generator=torch.Generator().manual_seed(17)) * 1.5

    batched = gpu_ops.tensor2img_batch(tensor, min_max=(-1, 1))
    reference = np.stack(
        [img_util.tensor2img(tensor[i], rgb2bgr=True, min_max=(-1, 1)) for i in range(5)]
    )

    assert batched.shape == (5, 32, 32, 3)
    assert batched.dtype == np.uint8
    assert np.array_equal(batched, reference)


@pytest.mark.unit
def test_tensor2img_u8_batch_stays_on_the_device_and_matches_the_numpy_form(gpu_ops):
    """The paste-stage face stack is the same quantization, without the transfer."""
    tensor = torch.tanh(torch.randn((3, 3, 16, 16), generator=torch.Generator().manual_seed(19)))

    u8 = gpu_ops.tensor2img_u8_batch(tensor, min_max=(-1, 1))
    as_numpy = gpu_ops.tensor2img_batch(tensor, min_max=(-1, 1))

    assert u8.dtype == torch.uint8
    assert torch.equal(u8, torch.from_numpy(as_numpy))


# ---------------------------------------------------------------------------
# The parse postprocess (design section 7.7, step 4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_colormap_batch_matches_the_vendored_assignment_loop(gpu_ops, helper_module):
    """The argmax + gather writes the values of the vendored per-class loop."""
    logits = torch.randn((2, 19, 32, 32), generator=torch.Generator().manual_seed(21))

    batched = gpu_ops.parse_colormap_batch(logits).squeeze(1).numpy()
    reference = []
    for i in range(2):
        out_parse = logits[i].argmax(dim=0).cpu().numpy()
        raw = np.zeros(out_parse.shape, dtype=np.float32)
        for map_idx, color_val in enumerate(gpu_ops.PARSE_MASK_COLORMAP):
            raw[out_parse == map_idx] = color_val
        reference.append(raw)
    reference = np.stack(reference)

    assert np.array_equal(batched, reference)


@pytest.mark.unit
def test_gaussian_blur_matches_two_cv2_gaussian_blurs(gpu_ops):
    """The batched two-pass blur vs the vendored ``GaussianBlur`` pair.

    The binary 0/255 masks are the realistic parse output, the uniform
    random masks the adversarial one; both keep the exact kernel and the
    reflect101 extension, so the residual is float summation order only -
    measured **1.7e-4** mask units (0..255 scale) at worst.  The pinned
    bound still trips on any structural change (a wrong border mode would
    show up in the tens).
    """
    rng = np.random.default_rng(3)
    binary = (rng.uniform(size=(2, 1, 512, 512)) > 0.5).astype(np.float32) * 255.0
    uniform = rng.uniform(0.0, 255.0, size=(2, 1, 512, 512)).astype(np.float32)

    for masks_np in (binary, uniform):
        masks = torch.from_numpy(masks_np)
        blurred = gpu_ops.gaussian_blur_2d_reflect101(masks, 101, 11.0)
        blurred = gpu_ops.gaussian_blur_2d_reflect101(blurred, 101, 11.0)
        reference = np.stack(
            [
                cv2.GaussianBlur(
                    cv2.GaussianBlur(masks_np[i, 0], (101, 101), 11), (101, 101), 11
                )
                for i in range(2)
            ]
        )
        diff = np.abs(reference - blurred[:, 0].numpy()).max()
        assert diff <= 0.01, f"the batched blur drifted {diff:.4f} mask units from cv2"


@pytest.mark.unit
def test_parse_soft_masks_match_the_vendored_postprocess(gpu_ops):
    """The full batched postprocess vs the vendored ``:463-472`` chain.

    Measured drift: **8.9e-7** in [0, 1] mask units (~2e-4 on the 0..255
    scale, the float summation order of the two blur passes).
    """
    logits = torch.randn((2, 19, 512, 512), generator=torch.Generator().manual_seed(27))

    soft = gpu_ops.parse_soft_masks(logits).numpy()[:, 0]  # (N, H, W) like the reference
    reference = []
    for i in range(2):
        out_parse = logits[i].argmax(dim=0).cpu().numpy()
        raw = np.zeros(out_parse.shape, dtype=np.float32)
        for map_idx, color_val in enumerate(gpu_ops.PARSE_MASK_COLORMAP):
            raw[out_parse == map_idx] = color_val
        blurred = cv2.GaussianBlur(raw, (101, 101), 11)
        blurred = cv2.GaussianBlur(blurred, (101, 101), 11)
        thres = 10
        blurred[:thres, :] = 0
        blurred[-thres:, :] = 0
        blurred[:, :thres] = 0
        blurred[:, -thres:] = 0
        reference.append(blurred / 255.0)
    reference = np.stack(reference)

    diff = np.abs(reference - soft).max()
    assert diff <= 1.0 / 255.0, f"the soft masks drifted {diff:.2e} from the vendored chain"
    # the border zeroing is exact in both
    assert np.all(soft[0, :10, :] == 0.0) and np.all(reference[0, :10, :] == 0.0)
    assert np.all(soft[0, :, -10:] == 0.0) and np.all(reference[0, :, -10:] == 0.0)


@pytest.mark.unit
def test_parse_forward_chunks_are_batch_invariant():
    """A chunked ParseNet forward returns the per-frame pixels.

    The batched paste forwards the parse net in chunks (the VRAM-aware cap
    of the design); batch independence in eval mode is the property that
    makes the chunk size a pure performance knob.  ``in_size`` 256 keeps the
    probe cheap - the claim is resolution-independent (BatchNorm in eval
    mode uses running stats, so no cross-sample path exists).
    """
    parsing = importlib.import_module("ComfyUI_KEEP.modules.deps.facelib.parsing.parsenet")
    net = parsing.ParseNet(in_size=256, out_size=256).eval()
    torch.manual_seed(29)
    inputs = torch.randn(8, 3, 256, 256)

    with torch.no_grad():
        per_frame = torch.cat([net(inputs[i : i + 1])[0] for i in range(8)], dim=0)
        chunked = torch.cat([net(inputs[j : j + 4])[0] for j in range(0, 8, 4)], dim=0)
        whole = net(inputs)[0]

    assert torch.equal(per_frame, chunked)
    assert torch.equal(per_frame, whole)
