"""The multi-face paste core: several faces per canvas row (the t3 round).

The batched paste of ``src/keep_gpu_paste.py`` gains the ``face_frame``
map: the faces, the affines and the ROIs are **face-indexed**, the
background, the canvas and the output stay **frame-indexed**.  The owner's
fix 4 (2026-10-31) reworked the chunk: every face runs the per-face ROI
tail of the single-face chunk (``_compute_roi_theta`` + ``F.affine_grid``
/ ``F.grid_sample``, the plain ``round_().clamp_(0, 255)`` quantize) into
its row - the t3 sequential per-row accumulator and the near-black
full-canvas fallback are gone, so overlapping ROIs no longer see an
earlier face's blend and a fully near-black row stays dark.

The pins:

1. the multi-face rows match the **vendored all-faces composite** - the
   real ``FaceRestoreHelper`` with all of the row's faces and affines in
   one ``paste_faces_to_input_image`` call (the reference of design
   section 4) - on the smooth and the noise corpus, with overlapping and
   with disjoint ROIs, at the upscale factors 1 and 2;
2. the 1-face degeneration holds the **measured residual** against the
   shipped identity path (``face_frame=None``), in the host, device-output
   and chunked assemblies;
3. the device-output canvas and the chunked assembly are **bit-exact**
   against the host stack for a ragged multi-face layout, across group
   boundaries (a group never splits a frame's faces);
4. the near-black tail is plain on this path: a fully near-black row
   stays dark (no fallback, no 255 factor) while the vendored composite
   amplifies, and a near-black face inside a bright row does not disturb
   the row;
5. the cancel poll runs once per group (the C1 contract of the new
   loops);
6. the map is validated loudly (wrong length, out-of-range row, the
   full-canvas chunk rejection, a ``num_frames`` mismatch).
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

#: The canvas of the multi-face pins - big enough for two overlapping
#: faces, small enough for a fast test.
MULTI_CANVAS_HW = (360, 640)


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
def image_utils():
    return importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")


@pytest.fixture()
def face_helper(helper_module, monkeypatch):
    """A vendored ``FaceRestoreHelper`` without its weight-backed models.

    The detection / parsing initializations are stubbed (the weights are
    not on this machine), so the reference paste runs the real vendored
    code - the same fixture the e2e pins of
    ``tests/test_gpu_paste_equivalence.py`` run.
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
    of the pixels alone, so the forward is batch invariant by construction
    - the vendored reference's per-face forwards and the batched paste's
    chunked forwards see the same masks.
    """

    def __init__(self, seed: int = 5) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(19, 3, 1, 1, generator=generator) * 0.35

    def __call__(self, samples):
        return (F.conv2d(samples, self.weight), None)


def _face_u8(corpus: str, seed: int) -> np.ndarray:
    """One 512x512 BGR face of a corpus (the smooth and the noise one)."""
    rng = np.random.default_rng(seed)
    if corpus == "noise":
        return rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    small = rng.uniform(0.0, 255.0, size=(32, 32, 3)).astype(np.float32)
    smooth = cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)
    return np.clip(smooth, 0, 255).astype(np.uint8)


def _multiface_corpus(
    gpu_ops,
    spec: list[list[tuple[float, float, float, str]]],
    canvas_hw: tuple[int, int] = MULTI_CANVAS_HW,
    face_seed: int = 311,
    bg_seed: int = 313,
):
    """The faces, the crop affines, the map and the background of a layout.

    ``spec`` names, per canvas row, the faces of the row as
    ``(x-fraction, y-fraction, scale, corpus)``.  The paste maps are
    derived from landmarks through the vendored
    ``estimateAffinePartial2D`` (the e2e pattern of
    ``tests/test_gpu_paste_equivalence.py``), so the reference paste and
    the batched paste ride the same affines: the crop affine goes to the
    vendored helper (which inverts it itself) and its
    ``cv2.invertAffineTransform * upscale`` product to the batched paste.
    """
    rng = np.random.default_rng(bg_seed)
    height, width = canvas_hw
    faces: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    face_frame: list[int] = []
    for row, descriptors in enumerate(spec):
        for x_fraction, y_fraction, scale, corpus in descriptors:
            index = len(faces)
            faces.append(_face_u8(corpus, face_seed + 2 * index))
            landmarks.append(
                (FACE_TEMPLATE - 256.0) * scale
                + np.array([width * x_fraction, height * y_fraction])
            )
            face_frame.append(row)
    affines = gpu_ops.estimate_affines(landmarks, FACE_TEMPLATE)
    background = rng.integers(
        0, 256, size=(len(spec), height, width, 3), dtype=np.uint8
    )
    return np.stack(faces), affines, face_frame, background


def _row_faces(face_frame: list[int]) -> dict[int, list[int]]:
    """The face indices of every canvas row, in face (detection) order."""
    rows: dict[int, list[int]] = {}
    for index, row in enumerate(face_frame):
        rows.setdefault(row, []).append(index)
    return rows


def _vendored_multi_paste(
    helper,
    frame_bgr: np.ndarray,
    faces: list[np.ndarray],
    crop_affines: list[np.ndarray],
    upscale_factor: float,
    parse_net,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """The vendored composite of one row's faces (``:355-528``).

    The helper's own multi-face flow: all of the row's faces and affines
    in **one** ``paste_faces_to_input_image`` call - the forward ``:495``
    loop blends every face into the accumulated image and the ``:497-507``
    tail runs once, over the final composite.
    """
    helper.clean_all()
    helper.use_parse = True
    helper.face_parse = parse_net
    helper.input_img = frame_bgr
    helper.is_gray = False
    helper.restored_faces = [face.copy() for face in faces]
    helper.affine_matrices = [matrix.copy() for matrix in crop_affines]
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
# The multi-face rows against the vendored all-faces composite
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_multi_face_rows_match_the_vendored_all_faces_composite(
    gpu_paste, gpu_ops, face_helper
):
    """The multi-face rows vs the vendored all-faces composite: smooth content.

    The reference is the real vendored ``paste_faces_to_input_image`` with
    all of the row's faces and affines in one call.  Row 0 carries two
    **overlapping** faces - the case the sequential per-row accumulator
    exists for: the second face has to blend against the first one's
    result, like the vendored ``:495`` loop does - and row 2 two disjoint
    ones; row 1 is the 1-face degeneration.

    Measured on this content: **3/255** at worst (the disjoint pair of row
    2), ~0.2% of the pixels differing at all - the two-warp residual of
    the single-face pins, carried through the sequential blend - and the
    1-face row **0/255** on this content.
    """
    faces, affines, face_frame, background = _multiface_corpus(
        gpu_ops,
        [
            [(0.38, 0.45, 0.28, "smooth"), (0.52, 0.45, 0.28, "smooth")],
            [(0.5, 0.5, 0.25, "smooth")],
            [(0.25, 0.35, 0.22, "smooth"), (0.75, 0.65, 0.22, "smooth")],
        ],
    )
    parse_net = _StubParseNet(seed=23)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)

    pasted = gpu_paste.paste_faces_batch(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        face_frame=face_frame,
    )

    assert len(pasted) == 3
    assert gpu_paste.PASTE_TRANSFER_STATS["frames"] == 3
    for row, indices in _row_faces(face_frame).items():
        reference = _vendored_multi_paste(
            face_helper, background[row],
            [faces[i] for i in indices], [affines[i] for i in indices],
            1.0, parse_net,
        )
        diff = np.abs(reference.astype(int) - pasted[row].astype(int))
        assert diff.max() <= 3, (
            f"row {row}: the multi-face paste drifted {diff.max()}/255 from "
            "the vendored all-faces composite"
        )


@pytest.mark.unit
def test_the_multi_face_rows_match_the_vendored_composite_on_noise_content(
    gpu_paste, gpu_ops, face_helper
):
    """The multi-face rows vs the vendored composite: adversarial content.

    Per-pixel noise is the worst case for the fixed-point sampling (the
    gradients reach 255 per pixel); the overlapping pair rides the same
    bound the single-face noise pin measures.

    Round 10 (the owner's fix 4, 2026-10-31): the chunk's per-face ROI
    ``grid_sample`` warp (``_compute_roi_theta``) replaced the cv2-exact
    fixed-point ROI warp, and a face blends against the background patch,
    not against the earlier faces' composite - so the drift class moved.
    Measured on this content: **4/255** at worst (the overlapping pair of
    row 0), pinned at the 8/255 class bound of the batched warps.
    """
    faces, affines, face_frame, background = _multiface_corpus(
        gpu_ops,
        [
            [(0.4, 0.42, 0.3, "noise"), (0.55, 0.55, 0.3, "noise")],
            [(0.3, 0.6, 0.26, "noise"), (0.75, 0.3, 0.26, "noise")],
        ],
    )
    parse_net = _StubParseNet(seed=37)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)

    pasted = gpu_paste.paste_faces_batch(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        face_frame=face_frame,
    )

    for row, indices in _row_faces(face_frame).items():
        reference = _vendored_multi_paste(
            face_helper, background[row],
            [faces[i] for i in indices], [affines[i] for i in indices],
            1.0, parse_net,
        )
        diff = np.abs(reference.astype(int) - pasted[row].astype(int))
        assert diff.max() <= 8, (
            f"row {row}: the multi-face paste drifted {diff.max()}/255 on noise"
        )


@pytest.mark.unit
def test_the_multi_face_composite_matches_the_vendored_at_upscale_two(
    gpu_paste, gpu_ops, face_helper
):
    """The multi-face paste at an upscale factor of two.

    The inverse warps of the paste are scaled by the upscale factor (the
    vendored ``get_inverse_affine``), so the canvas is the double one and
    the background is the LANCZOS4 resize of the frames - the same
    background the vendored reference receives.
    """
    canvas_hw = (240, 320)
    faces, affines, face_frame, frames = _multiface_corpus(
        gpu_ops,
        [
            [(0.4, 0.45, 0.3, "smooth"), (0.62, 0.5, 0.3, "smooth")],
            [(0.5, 0.5, 0.26, "smooth")],
        ],
        canvas_hw=canvas_hw,
    )
    parse_net = _StubParseNet(seed=47)
    background = gpu_paste.batched_bg_frames(frames, 2.0, None, torch.device("cpu"))
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=2.0)

    pasted = gpu_paste.paste_faces_batch(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        face_frame=face_frame,
    )

    assert background.shape == (2, 480, 640, 3), background.shape
    for row, indices in _row_faces(face_frame).items():
        reference = _vendored_multi_paste(
            face_helper, frames[row],
            [faces[i] for i in indices], [affines[i] for i in indices],
            2.0, parse_net, background=background[row],
        )
        diff = np.abs(reference.astype(int) - pasted[row].astype(int))
        assert diff.max() <= 2, (
            f"row {row}: the upscaled multi-face paste drifted {diff.max()}/255"
        )


# ---------------------------------------------------------------------------
# The bit-exact bookkeeping pins (CPU: every op is batch-invariant)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_one_face_per_row_map_matches_the_identity_path_within_the_measured_residual(
    gpu_paste, gpu_ops
):
    """``face_frame=[0, 1, 2]`` on one-face rows vs the shipped identity path.

    Fix 2.1 (2026-10-30, the owner's ``keep-fix3.txt``) vectorized the
    single-face chunk (the identity path), which left the mapped path a
    measured residual off it (3/255).  Round 10 (the owner's fix 4,
    2026-10-31) gives both chunks the same per-face ROI tail - a 1-face
    row degenerates to the identity path **byte for byte** again: the
    mapped rows are bit-exact against the identity rows in every assembly
    (host stack, device canvas, chunked IMAGE rows; measured 0/255, the
    bound stays at the 8/255 class limit of the batched warps).
    """
    faces, affines, face_frame, background = _multiface_corpus(
        gpu_ops,
        [
            [(0.4, 0.45, 0.26, "smooth")],
            [(0.55, 0.5, 0.3, "noise")],
            [(0.35, 0.55, 0.24, "smooth")],
        ],
    )
    assert face_frame == [0, 1, 2]
    parse_net = _StubParseNet(seed=23)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)
    batch_kwargs = dict(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        frame_chunk=2,
    )
    chunked_kwargs = dict(
        upscale_factor=1.0,
        bg_upscale_model=None,
        canvas_hw=None,
    )

    identity = gpu_paste.paste_faces_batch(**batch_kwargs)
    mapped = gpu_paste.paste_faces_batch(face_frame=face_frame, **batch_kwargs)
    for row, (identity_frame, mapped_frame) in enumerate(zip(identity, mapped)):
        diff = np.abs(identity_frame.astype(int) - mapped_frame.astype(int)).max()
        assert diff <= 8, (
            f"row {row}: the mapped host stack drifted {diff}/255 from the identity path"
        )

    identity_canvas = gpu_paste.paste_faces_batch(device_output=True, **batch_kwargs)
    mapped_canvas = gpu_paste.paste_faces_batch(
        device_output=True, face_frame=face_frame, **batch_kwargs
    )
    canvas_diff = int((identity_canvas.float() - mapped_canvas.float()).abs().max())
    assert canvas_diff <= 8, (
        f"the mapped device canvas drifted {canvas_diff}/255 from the identity path"
    )

    chunked_identity = gpu_paste.paste_faces_batch_chunked(
        **chunked_kwargs, **batch_kwargs
    )
    chunked_mapped = gpu_paste.paste_faces_batch_chunked(
        **chunked_kwargs, face_frame=face_frame, **batch_kwargs
    )
    chunked_diff = (chunked_identity - chunked_mapped).abs().max().item() * 255
    assert chunked_diff <= 8 + 1e-6, (
        f"the mapped chunked assembly drifted {chunked_diff:.2f}/255 from the identity path"
    )


@pytest.mark.unit
def test_the_device_output_canvas_is_bit_identical_to_the_host_stack_multi_face(
    gpu_paste, gpu_ops
):
    """The device-output canvas is the multi-face host stack, byte for byte.

    The device-output mode moves the assembly of the output stack onto the
    device; with the map the scatter targets are the mapped rows.  The
    canvas has to equal the list of the host rows exactly - on every
    pixel, for every row of the ragged layout - and the counters keep
    their row semantics.  Round 10 (the owner's fix 4, 2026-10-31): the
    ROI patch staging retired with the accumulator - ``roi_staged_bytes``
    stays 0 on this path and only ``face_h2d_bytes`` counts the upload.
    """
    faces, affines, face_frame, background = _multiface_corpus(
        gpu_ops,
        [
            [(0.38, 0.45, 0.28, "smooth"), (0.52, 0.45, 0.28, "noise")],
            [(0.5, 0.5, 0.25, "smooth")],
            [(0.25, 0.35, 0.22, "noise"), (0.75, 0.65, 0.22, "smooth")],
        ],
    )
    parse_net = _StubParseNet(seed=131)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)
    kwargs = dict(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        frame_chunk=2,
        face_frame=face_frame,
    )

    host_rows = gpu_paste.paste_faces_batch(**kwargs)
    device_canvas = gpu_paste.paste_faces_batch(device_output=True, **kwargs)

    assert isinstance(device_canvas, torch.Tensor), type(device_canvas)
    assert tuple(device_canvas.shape) == (3, *MULTI_CANVAS_HW, 3), device_canvas.shape
    for row, host_frame in enumerate(host_rows):
        assert np.array_equal(device_canvas[row].numpy(), host_frame), (
            f"row {row}: the device-output canvas drifted from the host stack"
        )
    stats = gpu_paste.PASTE_TRANSFER_STATS
    assert stats["h2d_bytes"] == 0 and stats["d2h_bytes"] == 0, stats
    assert stats["roi_staged_bytes"] == 0, stats
    assert stats["face_h2d_bytes"] > 0, stats
    assert stats["frames"] == 3 and stats["roi_frames"] == 3, stats
    assert stats["fallback_frames"] == 0, stats


@pytest.mark.unit
def test_the_chunked_multi_face_assembly_matches_the_host_stack_and_the_full_mode(
    gpu_paste, gpu_ops, image_utils
):
    """The chunked IMAGE rows == the host stack == the full device canvas.

    The helper pastes in ``frame_chunk=1`` groups - one row per group,
    with all of the row's faces - while the full mode runs the whole batch
    and the host stack is the mapped list return.  On the CPU device
    every op of the paste is batch-invariant (measured), so all three
    assemblies come out as the same float32 IMAGE batch, byte for byte.
    """
    faces, affines, face_frame, background = _multiface_corpus(
        gpu_ops,
        [
            [(0.38, 0.45, 0.28, "smooth"), (0.52, 0.45, 0.28, "smooth")],
            [(0.5, 0.5, 0.25, "noise")],
            [(0.25, 0.35, 0.22, "smooth"), (0.75, 0.65, 0.22, "noise")],
        ],
    )
    parse_net = _StubParseNet(seed=233)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)
    kwargs = dict(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        face_frame=face_frame,
    )

    host_rows = gpu_paste.paste_faces_batch(frame_chunk=3, **kwargs)
    full_canvas = gpu_paste.paste_faces_batch(
        frame_chunk=3, device_output=True, **kwargs)
    chunked = gpu_paste.paste_faces_batch_chunked(
        upscale_factor=1.0,
        bg_upscale_model=None,
        canvas_hw=None,
        frame_chunk=1,
        **kwargs,
    )

    assert chunked.dtype == torch.float32 and chunked.device.type == "cpu"
    assert tuple(chunked.shape) == (3, *MULTI_CANVAS_HW, 3), chunked.shape
    assert torch.equal(
        chunked, image_utils.bgr_u8_batch_to_comfy(np.stack(host_rows))
    ), "the chunked helper drifted from the host stack"
    assert torch.equal(chunked, gpu_paste.comfy_convert_out(full_canvas)), (
        "the chunked helper drifted from the full device mode"
    )
    stats = gpu_paste.PASTE_TRANSFER_STATS
    assert stats["frames"] == 3, stats


@pytest.mark.unit
def test_the_chunked_multi_face_assembly_is_independent_of_the_group_boundaries(
    gpu_paste, gpu_ops
):
    """Ragged face counts across every group boundary: the same IMAGE rows.

    Four rows of 1, 3, 2 and 1 faces in groups of one and two rows - the
    group boundaries never split a frame's faces (a group is a set of
    whole rows), and the group size must not change a pixel: on the CPU
    device the parse forward, the warps, the blend and the convert chain
    are batch-invariant (measured), so every grouping comes out identical.
    """
    faces, affines, face_frame, background = _multiface_corpus(
        gpu_ops,
        [
            [(0.4, 0.45, 0.26, "smooth")],
            [(0.35, 0.4, 0.26, "noise"), (0.55, 0.5, 0.26, "smooth"),
             (0.72, 0.62, 0.26, "noise")],
            [(0.3, 0.55, 0.24, "smooth"), (0.68, 0.4, 0.24, "noise")],
            [(0.5, 0.5, 0.25, "smooth")],
        ],
    )
    assert [len(v) for v in _row_faces(face_frame).values()] == [1, 3, 2, 1]
    parse_net = _StubParseNet(seed=241)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)
    kwargs = dict(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        face_frame=face_frame,
    )

    full_canvas = gpu_paste.paste_faces_batch(
        frame_chunk=4, device_output=True, **kwargs)
    chunked_two = gpu_paste.paste_faces_batch_chunked(
        upscale_factor=1.0, bg_upscale_model=None, canvas_hw=None,
        frame_chunk=2, **kwargs)
    chunked_one = gpu_paste.paste_faces_batch_chunked(
        upscale_factor=1.0, bg_upscale_model=None, canvas_hw=None,
        frame_chunk=1, **kwargs)

    assert torch.equal(chunked_two, gpu_paste.comfy_convert_out(full_canvas)), (
        "the two-row groups changed the pasted pixels"
    )
    assert torch.equal(chunked_one, chunked_two), (
        "the one-row groups changed the pasted pixels"
    )


@pytest.mark.unit
def test_the_multi_face_core_writes_the_indexed_destination_byte_for_byte(
    gpu_paste, gpu_ops
):
    """The multi-face branch of the indexed destination: the same bytes.

    The multi-face path is the second half of round 18: the faces, the
    affines and the ROIs are **face-indexed** while the canvas rows stay
    **frame-indexed**, so the destination of a canvas row is a row of the
    caller's buffer - the same indexed write as the single-face branch,
    on the multi-face group's rows.

    The pin is **bit-exact**: the destination rows carry the core's
    own-buffer bytes (the same run without ``out=``), byte for byte, and
    the rows the map does **not** name keep the sentinel fill.  The map
    ``[0, 2, 5]`` is scattered on purpose and the group size of two splits
    it across the groups (the first group owns the rows 0 and 2, the
    second one the row 5) - a slice write or a read-side ``out[rows]``
    copy (advanced indexing - a temporary) would fail this pin loudly.
    """
    faces, affines, face_frame, background = _multiface_corpus(
        gpu_ops,
        [
            [(0.38, 0.45, 0.28, "smooth"), (0.52, 0.45, 0.28, "smooth")],
            [(0.5, 0.5, 0.25, "noise")],
            [(0.25, 0.35, 0.22, "noise"), (0.75, 0.65, 0.22, "smooth")],
        ],
        face_seed=401,
        bg_seed=403,
    )
    assert face_frame == [0, 0, 1, 2, 2], face_frame
    parse_net = _StubParseNet(seed=405)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)
    kwargs = dict(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        face_frame=face_frame,
        num_frames=3,
    )
    core_kwargs = dict(
        upscale_factor=1.0,
        bg_upscale_model=None,
        canvas_hw=None,
        frame_chunk=2,
    )
    destination_rows = [0, 2, 5]

    own = gpu_paste.paste_faces_batch_chunked(**core_kwargs, **kwargs)
    assert tuple(own.shape) == (3, *MULTI_CANVAS_HW, 3), own.shape

    sentinel = -7.5
    given = torch.full((6, *MULTI_CANVAS_HW, 3), sentinel, dtype=torch.float32)
    returned = gpu_paste.paste_faces_batch_chunked(
        **core_kwargs, **kwargs, out=given, out_rows=destination_rows)

    assert returned is given, "the core did not return the destination it was given"
    for canvas_row, destination_row in enumerate(destination_rows):
        assert torch.equal(given[destination_row], own[canvas_row]), (
            f"the destination row {destination_row} does not carry the "
            f"canvas row {canvas_row}'s bytes"
        )
    untouched = torch.full((3, *MULTI_CANVAS_HW, 3), sentinel, dtype=torch.float32)
    for offset, row in enumerate((1, 3, 4)):
        assert torch.equal(given[row], untouched[offset]), (
            f"the row {row} was written although the map does not name it - "
            "the write was a slice, not an indexed assignment"
        )


# ---------------------------------------------------------------------------
# The per-row near-black tail
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_near_black_row_stays_dark_without_the_full_canvas_fallback(
    gpu_paste, gpu_ops, face_helper
):
    """A row that is near black **everywhere** stays dark - no fallback (round 10).

    The vendored tail scales the **whole canvas** by 255 when the blended
    frame is near black (``:497-507``); the t3 chunk re-blended such a row
    at the full canvas to reproduce it.  The owner's fix 4 (2026-10-31)
    replaces the chunk's quantizing tail with the plain
    ``round_().clamp_(0, 255)`` - the near-black branch (and its fallback)
    is gone from this path (owner-authorized pixel change): the 0/1 blend
    quantizes back to 0/1 and the row stays dark, while the vendored
    composite still amplifies to 255.  The pin holds exactly that: no
    fallback counter, the row bit-identical between the host and the
    device-output modes, dark on the batched path and amplified on the
    vendored one (the measured behavior replacing the old
    ``test_a_near_black_row_takes_the_full_canvas_fallback_with_all_its_faces``
    pin, which bound the amplified row at 1/255).
    """
    canvas_hw = MULTI_CANVAS_HW
    rng = np.random.default_rng(277)
    faces = np.stack([
        rng.integers(0, 2, size=(512, 512, 3), dtype=np.uint8),
        rng.integers(0, 2, size=(512, 512, 3), dtype=np.uint8),
    ])
    landmarks = [
        (FACE_TEMPLATE - 256.0) * 0.26 + np.array([230.0, 160.0]),
        (FACE_TEMPLATE - 256.0) * 0.26 + np.array([400.0, 200.0]),
    ]
    affines = gpu_ops.estimate_affines(landmarks, FACE_TEMPLATE)
    background = rng.integers(0, 2, size=(1, *canvas_hw, 3), dtype=np.uint8)
    parse_net = _StubParseNet(seed=281)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)
    kwargs = dict(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        frame_chunk=1,
        face_frame=[0, 0],
    )

    host_rows = gpu_paste.paste_faces_batch(**kwargs)
    stats = gpu_paste.PASTE_TRANSFER_STATS
    assert stats["fallback_frames"] == 0, stats

    device_canvas = gpu_paste.paste_faces_batch(device_output=True, **kwargs)
    stats = gpu_paste.PASTE_TRANSFER_STATS
    assert stats["fallback_frames"] == 0 and stats["roi_frames"] == 1, stats
    assert np.array_equal(
        device_canvas[0].numpy(), host_rows[0]
    ), "the near-black row drifted between the host and the device modes"

    # the plain tail keeps the near-black row dark - no 255 factor here
    assert int(host_rows[0].max()) <= 1, int(host_rows[0].max())

    reference = _vendored_multi_paste(
        face_helper, background[0], list(faces), list(affines), 1.0, parse_net,
    )
    # the vendored composite still takes its near-black branch
    assert int(reference.max()) == 255, int(reference.max())


@pytest.mark.unit
def test_a_near_black_face_in_a_bright_row_keeps_the_row_global_branch(
    gpu_paste, gpu_ops, face_helper
):
    """Near-black faces do not drag a bright row into the fallback.

    Row 0: two near-black faces whose union is near black while a bright
    pixel far outside stays bright - the frame-global branch of the tail
    (``maximum = max(union, outside)``), which must quantize with clip,
    not scale by 255.  Row 1: one smooth face next to a near-black one -
    the tail is per **row**, so the bright face keeps the row in the
    normal branch.  Neither row may take the fallback, and both match the
    vendored composite.
    """
    canvas_hw = MULTI_CANVAS_HW
    rng = np.random.default_rng(283)
    height, width = canvas_hw
    faces = np.stack([
        rng.integers(0, 2, size=(512, 512, 3), dtype=np.uint8),
        rng.integers(0, 2, size=(512, 512, 3), dtype=np.uint8),
        _face_u8("smooth", 289),
        rng.integers(0, 2, size=(512, 512, 3), dtype=np.uint8),
    ])
    landmarks = [
        (FACE_TEMPLATE - 256.0) * 0.24 + np.array([width * 0.3, height * 0.4]),
        (FACE_TEMPLATE - 256.0) * 0.24 + np.array([width * 0.62, height * 0.45]),
        (FACE_TEMPLATE - 256.0) * 0.26 + np.array([width * 0.35, height * 0.5]),
        (FACE_TEMPLATE - 256.0) * 0.26 + np.array([width * 0.7, height * 0.55]),
    ]
    affines = gpu_ops.estimate_affines(landmarks, FACE_TEMPLATE)
    background = rng.integers(64, 256, size=(2, *canvas_hw, 3), dtype=np.uint8)
    background[0, height - 1, width - 1] = 200  # the bright pixel far outside
    parse_net = _StubParseNet(seed=293)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)

    pasted = gpu_paste.paste_faces_batch(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        frame_chunk=2,
        face_frame=[0, 0, 1, 1],
    )

    stats = gpu_paste.PASTE_TRANSFER_STATS
    assert stats["fallback_frames"] == 0, stats
    assert stats["roi_frames"] == 2, stats

    rows = _row_faces([0, 0, 1, 1])
    # measured: 3/255 on the near-black pair's row, 1/255 on the mixed row
    bounds = {0: 3, 1: 1}
    for row, indices in rows.items():
        reference = _vendored_multi_paste(
            face_helper, background[row],
            [faces[i] for i in indices], [affines[i] for i in indices],
            1.0, parse_net,
        )
        diff = np.abs(reference.astype(int) - pasted[row].astype(int))
        assert diff.max() <= bounds[row], (
            f"row {row}: the frame-global branch drifted {diff.max()}/255"
        )


# ---------------------------------------------------------------------------
# The cancel polls of the new loops (the C1 contract)
# ---------------------------------------------------------------------------


class _Cancel(Exception):
    """The stub cancel of the poll pins."""


@pytest.mark.unit
def test_the_multi_face_groups_poll_the_cancel_flag(gpu_paste, gpu_ops, monkeypatch):
    """One cancel poll per group - the C1 contract of the new loops.

    The poll is the module-qualified patch point (``src/keep_interrupt.py``):
    a stub that raises on the second call must abort both the batched and
    the chunked multi-face loops at the second group, and a counting no-op
    must see exactly one call per group (3 rows in ``frame_chunk=1``
    groups -> 3 polls).
    """
    faces, affines, face_frame, background = _multiface_corpus(
        gpu_ops,
        [
            [(0.4, 0.45, 0.26, "smooth")],
            [(0.35, 0.4, 0.26, "smooth"), (0.6, 0.5, 0.26, "smooth")],
            [(0.5, 0.55, 0.24, "smooth")],
        ],
    )
    parse_net = _StubParseNet(seed=307)
    inverses = gpu_ops.inverse_affines(affines, upscale_factor=1.0)
    batch_kwargs = dict(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
        frame_chunk=1,
    )

    calls: list[int] = []

    def cancel_on_second():
        calls.append(len(calls))
        if len(calls) == 2:
            raise _Cancel()

    monkeypatch.setattr(gpu_paste.keep_interrupt, "check_interrupted", cancel_on_second)
    with pytest.raises(_Cancel):
        gpu_paste.paste_faces_batch(face_frame=face_frame, **batch_kwargs)
    assert len(calls) == 2, calls

    calls.clear()
    with pytest.raises(_Cancel):
        gpu_paste.paste_faces_batch_chunked(
            upscale_factor=1.0,
            bg_upscale_model=None,
            canvas_hw=None,
            face_frame=face_frame,
            **batch_kwargs,
        )
    assert len(calls) == 2, calls

    monkeypatch.setattr(
        gpu_paste.keep_interrupt, "check_interrupted", lambda: calls.append(1)
    )
    calls.clear()
    gpu_paste.paste_faces_batch(face_frame=face_frame, **batch_kwargs)
    assert len(calls) == 3, calls


# ---------------------------------------------------------------------------
# The validation of the map
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_face_frame_map_is_validated_loudly(gpu_paste):
    """A broken map fails loudly instead of mis-scattering a face."""
    faces = np.zeros((2, 512, 512, 3), dtype=np.uint8)
    inverses = [np.eye(2, 3)] * 2
    background = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    parse_net = _StubParseNet()
    batch_kwargs = dict(
        faces_bgr=faces,
        inverse_affines=inverses,
        background_bgr=background,
        device=torch.device("cpu"),
        parse_net=parse_net,
    )
    chunked_kwargs = dict(
        upscale_factor=1.0,
        bg_upscale_model=None,
        device=torch.device("cpu"),
        canvas_hw=None,
        parse_net=parse_net,
    )

    with pytest.raises(ValueError):  # the wrong length
        gpu_paste.paste_faces_batch(face_frame=[0], **batch_kwargs)
    with pytest.raises(ValueError):  # a row outside the background stack
        gpu_paste.paste_faces_batch(face_frame=[0, 5], **batch_kwargs)
    with pytest.raises(ValueError):  # the full-canvas chunk has no map
        gpu_paste.paste_faces_batch(face_frame=[0, 1], use_roi=False, **batch_kwargs)
    with pytest.raises(ValueError):  # the row count does not match the stack
        gpu_paste.paste_faces_batch(
            face_frame=[0, 1], num_frames=3, **batch_kwargs)
    with pytest.raises(ValueError):  # the chunked helper: the wrong length
        gpu_paste.paste_faces_batch_chunked(
            faces_bgr=faces, inverse_affines=inverses, background_bgr=background,
            face_frame=[0, 1, 0], **chunked_kwargs)
    with pytest.raises(ValueError):  # the chunked helper: a row outside
        gpu_paste.paste_faces_batch_chunked(
            faces_bgr=faces, inverse_affines=inverses, background_bgr=background,
            face_frame=[0, 7], **chunked_kwargs)
