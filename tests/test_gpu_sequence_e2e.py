"""The end-to-end regression of the sequence path: the shipped
``process_image_sequence`` against the **base** implementation (``2e8d9b9``).

The acceptance list of the GPU video pipeline design asks for "an end-to-end
synthetic regression test new-vs-old ``process_image_sequence`` within
documented tolerance".  This is that test: the base implementation is
extracted from the repository history (``git show 2e8d9b9:src/keep_processor.py``)
and run against the shipped one **with the real vendored
``FaceRestoreHelper``** (its ``read_image`` upscale, its
``paste_faces_to_input_image`` canvas sizing and its parse chain included) and
a deterministic detector, parse net and KEEP net.

What it pins:

- the **shape** of the output - the base and the shipped pipeline must agree
  (the round-1 defect: a frame with ``min(h, w) < 512`` left the shipped
  pipeline at the raw resolution and clipped the pasted face);
- the **pixels** - every branch stays inside the documented tolerances: the
  batched device crop contributes its measured envelope
  (:data:`CROP_TOLERANCE`, Opt 1) to both paste branches, and the parse
  mask blur of the device adds :data:`PASTE_TOLERANCE` on the batched-paste
  branch;
- the **ROI path** - the shipped paste runs the vectorized chunk (fix 2.1,
  2026-10-30: the full-canvas ``grid_sample`` face warp replaced the cv2
  fixed-point warp + ROI slicing), and the suite asserts the ROI marker
  (the product's own transfer accounting) over both paste branches (with
  and without a parse net) and the small-frame case: the pixels alone
  cannot prove that the vectorized chunk ran, because it produces a
  within-tolerance image of the full-canvas paste.

The test is skipped when the base implementation cannot be reached (a
distribution without the git history) - it never fails silently.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import pytest
import torch

#: The commit the shipped pipeline is compared against (the base of the
#: GPU video pipeline work).
BASE_COMMIT = "2e8d9b9"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The vendored 5-point 512 template (``face_restoration_helper.py:52-53``).
FACE_TEMPLATE = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]]
)

#: The tolerance of the batched paste against the vendored paste - the
#: parse mask blur of the device (the only replaced paste op with a
#: deviation).
PASTE_TOLERANCE = 1

#: The measured envelope of the batched device crop against the vendored
#: ``cv2.warpAffine`` (Opt 1): the float32 ``grid_sample`` vs cv2's 1/32 px
#: fixed-point grid - measured 6/255, pinned bound 8/255 on the per-pixel
#: noise content of this file (``tests/test_gpu_batch_equivalence.py``;
#: 1/255 on smooth content).
CROP_TOLERANCE = 8


@pytest.fixture(scope="module")
def helper_module():
    return importlib.import_module(
        "ComfyUI_KEEP.modules.deps.facelib.utils.face_restoration_helper")


@pytest.fixture(scope="module")
def base_module(helper_module):
    """The base implementation of ``process_image_sequence`` (``2e8d9b9``)."""
    try:
        source = subprocess.run(
            ["git", "-C", REPO_ROOT, "show", f"{BASE_COMMIT}:src/keep_processor.py"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"the base implementation is not reachable ({exc})")
    name = "ComfyUI_KEEP.src._base_keep_processor"
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix="base_keep_processor_", delete=False,
        encoding="utf-8")
    with handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location(name, handle.name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    os.remove(handle.name)


@pytest.fixture(scope="module")
def shipped_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_processor")


class _StubParseNet(torch.nn.Module):
    """A deterministic parse net with a **spatial** mask.

    Class 1 (colormap 255) inside an ellipse, class 14 (colormap 0) outside
    - so the soft mask of the paste is non-trivial (an all-zero mask would
    make the paste output the background and hide every difference).
    """

    def forward(self, x):
        n, _, h, w = x.shape
        ys = torch.arange(h, dtype=x.dtype, device=x.device).view(1, 1, h, 1)
        xs = torch.arange(w, dtype=x.dtype, device=x.device).view(1, 1, 1, w)
        ellipse = (((xs - w * 0.5) / (w * 0.28)) ** 2
                   + ((ys - h * 0.5) / (h * 0.38)) ** 2) < 1.0
        logits = torch.zeros((n, 19, h, w), dtype=x.dtype, device=x.device)
        logits[:, 1] = torch.where(ellipse[0, 0].expand(n, h, w), 5.0, 0.0)
        logits[:, 14] = torch.where(ellipse[0, 0].expand(n, h, w), 0.0, 5.0)
        return (logits,)


class _DetectingHelper:
    """The vendored helper with a deterministic detector.

    The landmarks are derived from the image the helper read - so the small
    frames (the ``read_image`` upscale) get their landmarks in the upscaled
    space, exactly like the real detector.
    """

    def __init__(self, helper_module, use_parse: bool):
        helper_module.init_detection_model = lambda *a, **k: object()
        helper_module.init_parsing_model = lambda *a, **k: None
        self.use_parse = use_parse
        self.helper = helper_module.FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model="retinaface_resnet50", save_ext="png", use_parse=False,
            device="cpu", model_rootpath="unused")
        if use_parse:
            self.helper.face_parse = _StubParseNet()
            self.helper.use_parse = True

    def get_face_landmarks_5(self, only_center_face=False, resize=640,
                             eye_dist_threshold=5, only_keep_largest=False):
        height, width = self.helper.input_img.shape[:2]
        scale = min(height, width) / 512.0 * 0.72
        center = np.array([width * 0.5, height * 0.46])
        self.helper.all_landmarks_5 = [(FACE_TEMPLATE - 256.0) * scale + center]
        return 1

    def __getattr__(self, name):
        return getattr(self.helper, name)

    def __setattr__(self, name, value):
        if name in ("use_parse", "helper"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.helper, name, value)


class _DegenerateHelper(_DetectingHelper):
    """A detector whose landmarks are **degenerate**.

    All five points of every frame sit on the same pixel, so the vendored
    ``cv2.estimateAffinePartial2D(..., LMEDS)`` fails and returns ``None``
    - the error path of ``face_restoration_helper.py:314-319``.
    """

    def get_face_landmarks_5(self, only_center_face=False, resize=640,
                             eye_dist_threshold=5, only_keep_largest=False):
        height, width = self.helper.input_img.shape[:2]
        point = np.array([width * 0.5, height * 0.5], dtype=np.float64)
        self.helper.all_landmarks_5 = [np.repeat(point[None, :], 5, axis=0)]
        return 1


class _StubKeepNet:
    """A deterministic KEEP net (the restored face is a function of the crop)."""

    def __call__(self, frames, need_upscale=False):
        return frames * 0.5 + 0.25


def _make_pack(helper):
    return types.SimpleNamespace(
        keep_net=_StubKeepNet(), face_helper=helper, bg_upscale_model=None,
        face_upscale_model=None, device=torch.device("cpu"), model_type_str="KEEP")


def _run(module, helper, frames, upscale=1.0, aligned=False):
    processor = module.KEEPFaceProcessor(_make_pack(helper))
    with torch.no_grad():
        return processor.process_image_sequence(
            image_sequence_tensor=frames, final_upscale_factor=upscale,
            has_aligned_frames=aligned, only_center_face=True, draw_box=False,
            max_clip_length=20)


def _synthetic_frames(num_frames, height, width, seed=7):
    rng = np.random.default_rng(seed)
    raw = (rng.random((num_frames, height, width, 3)) * 200).astype(np.uint8)
    return torch.from_numpy(raw).float() / 255.0


@pytest.mark.unit
@pytest.mark.parametrize("use_parse", [False, True])
@pytest.mark.parametrize("frame_hw", [(720, 1280), (300, 400)])
def test_the_sequence_matches_the_base_implementation(
    helper_module, base_module, shipped_module, use_parse, frame_hw
):
    """The shipped sequence path vs the base implementation of ``2e8d9b9``.

    The frames of 720x1280 (the normal case) and of 300x400 (the
    ``min(h, w) < 512`` case of the round-1 defect) are both covered.

    The crop warps on the device (Opt 1): both paste branches carry its
    measured envelope against cv2's fixed-point sampling
    (:data:`CROP_TOLERANCE` - this corpus is per-pixel noise, the warp's
    adversarial case).  With the parse net the batched paste adds the
    parse mask blur of the device (:data:`PASTE_TOLERANCE`).
    """
    height, width = frame_hw
    frames = _synthetic_frames(2, height, width)

    base_out = _run(base_module, _DetectingHelper(helper_module, use_parse), frames)
    shipped_out = _run(shipped_module, _DetectingHelper(helper_module, use_parse), frames)

    base = base_out.numpy()
    shipped = shipped_out.numpy()
    assert shipped.shape == base.shape, (
        f"the shipped pipeline changed the output shape for {height}x{width}: "
        f"{shipped.shape} != {base.shape}"
    )
    diff = np.abs(base.astype(np.float32) - shipped.astype(np.float32))
    tolerance = CROP_TOLERANCE / 255.0 + (
        PASTE_TOLERANCE / 255.0 if use_parse else 0.0
    )
    assert diff.max() <= tolerance + 1e-6, (
        f"the shipped pipeline drifted {diff.max() * 255:.2f}/255 from the base "
        f"for {height}x{width} (use_parse={use_parse})"
    )


@pytest.mark.unit
def test_the_small_frames_keep_the_upscaled_canvas_of_the_base(
    helper_module, base_module, shipped_module
):
    """The ``min(h, w) < 512`` frames stay on the base's canvas.

    The vendored ``read_image`` upscales such a frame to 512 px on its
    short side (``face_restoration_helper.py:182-184``) and the vendored
    paste sizes its canvas from that image (``:362-370``) - the base called
    ``read_image`` in the paste loop.  The shipped pipeline has to produce
    the same resolution.
    """
    frames = _synthetic_frames(2, 300, 400)
    base_out = _run(base_module, _DetectingHelper(helper_module, True), frames)
    shipped_out = _run(shipped_module, _DetectingHelper(helper_module, True), frames)
    assert tuple(base_out.shape) == (2, 512, 683, 3), base_out.shape
    assert tuple(shipped_out.shape) == tuple(base_out.shape), shipped_out.shape


@pytest.mark.unit
def test_the_aligned_frames_match_the_base_implementation(
    helper_module, base_module, shipped_module
):
    """The ``has_aligned_frames`` branch vs the base implementation.

    The frames are the faces themselves: the base and the shipped pipeline
    resize them with the same torch helper (``resize_bgr_u8``) and paste the
    restored face - the pin carries the pipeline's documented tolerance of
    1/255 (the uint8 rounding of the paste), not a bit-exactness claim.
    """
    frames = _synthetic_frames(2, 512, 512)
    base_out = _run(base_module, _DetectingHelper(helper_module, True), frames,
                    aligned=True)
    shipped_out = _run(shipped_module, _DetectingHelper(helper_module, True), frames,
                       aligned=True)
    base = base_out.numpy()
    shipped = shipped_out.numpy()
    assert shipped.shape == base.shape, (
        f"the shipped pipeline changed the output shape: {shipped.shape} != {base.shape}"
    )
    diff = np.abs(base.astype(np.float32) - shipped.astype(np.float32))
    assert diff.max() <= 1.0 / 255.0 + 1e-6, (
        f"the shipped pipeline drifted {diff.max() * 255:.2f}/255 from the base "
        f"on the aligned branch"
    )


@pytest.mark.unit
def test_the_failed_affine_degrades_instead_of_crashing(
    helper_module, base_module, shipped_module
):
    """The LMEDS failure of the vendored crop (design section 15.3).

    When the vendored ``cv2.estimateAffinePartial2D(..., LMEDS)`` fails
    (``face_restoration_helper.py:314-319``), the **base implementation
    crashes**: ``align_warp_face`` appends ``None`` to ``cropped_faces``, so
    the ``if not self.face_helper.cropped_faces`` guard is False and the
    base evaluates ``img2tensor(None / 255.)`` -> ``TypeError``.

    The **shipped pipeline degrades gracefully**: the frame gets the base's
    neutral 0.5 crop, the paste skips the frame (its inverse affine is
    ``None``) and the frame leaves the pipeline as its background.  This is
    a crash -> graceful divergence, **not** a pixel divergence - it is
    pinned here so that it can never change silently, and the divergence is
    documented in the design.
    """
    frames = _synthetic_frames(2, 720, 1280)

    # the base crashes on this path
    with pytest.raises(TypeError):
        _run(base_module, _DegenerateHelper(helper_module, False), frames)

    # the shipped pipeline does not
    shipped_out = _run(
        shipped_module, _DegenerateHelper(helper_module, False), frames)
    assert tuple(shipped_out.shape) == (2, 720, 1280, 3), shipped_out.shape

    # the frames leave the pipeline as their background (no face pasted)
    expected = (frames.numpy() * 255.0).round().astype(np.uint8)
    diff = np.abs(expected.astype(np.float32) - shipped_out.numpy() * 255.0)
    assert diff.max() <= 1.0, (
        f"the failed-affine frame is not its own background: "
        f"{diff.max():.2f}/255 off"
    )


# ---------------------------------------------------------------------------
# The ROI path (the paste-stage CPU optimization round)
#
# The shipped paste runs in the **ROI** of the transformed face: the face
# warp, the parse-mask warp and the blend run there and only the ROI
# patch travels between the host and the device.  The ROI path produces
# the same pixels as the full-canvas paste it replaces - so the pixels
# alone cannot prove that it ran.  The **ROI marker** is what proves it:
# the product's own transfer accounting (``PASTE_TRANSFER_STATS``), which
# the suite resets before a run and reads after it.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_paste():
    """The paste module of the shipped pipeline (the ROI marker lives there)."""
    return importlib.import_module("ComfyUI_KEEP.src.keep_gpu_paste")


def _run_marked(gpu_paste, module, helper, frames, **kwargs):
    """Run the pipeline and return its output and the ROI marker.

    ``PASTE_TRANSFER_STATS`` describes the **last** paste call, so the
    counters are reset before the run - a stale counter of an earlier
    call must never be read as the evidence of this one.
    """
    gpu_paste.reset_paste_transfer_stats()
    output = _run(module, helper, frames, **kwargs)
    return output, dict(gpu_paste.PASTE_TRANSFER_STATS)


@pytest.mark.unit
@pytest.mark.parametrize("frame_hw", [(720, 1280), (300, 400)])
def test_the_batched_paste_takes_the_roi_path(
    helper_module, base_module, shipped_module, gpu_paste, frame_hw
):
    """The shipped paste runs the vectorized chunk (the fix-2.1 marker).

    The paste-stage CPU optimization round made the ROI path the shipped
    path of the batched paste; fix 2.1 (2026-10-30, the owner's
    ``keep-fix3.txt``) vectorized that chunk - the full-canvas
    ``grid_sample`` face warp replaced the cv2 fixed-point warp + ROI
    slicing, and the ROI patch transfers are gone with it.

    The **ROI marker** - the product's own transfer accounting - is what
    proves that the path was taken.  The pixels alone would not: the
    full-canvas paste produces a within-tolerance image (the next test
    pins that), so a suite that only compared the output would stay green
    with the vectorized chunk switched off.

    Both frame sizes are covered: the normal 720x1280 and the
    ``min(h, w) < 512`` frame, whose canvas is the 512 px upscaled frame
    of the vendored ``read_image``.  The tolerance against the base
    implementation is re-measured (2026-10-30): the vectorized paste's
    full-canvas ``grid_sample`` face warp joins the documented drift class
    of the batched warps (up to 8/255 on high-frequency content) - the
    whole-pipeline residual measured **4/255** (720x1280) and **2/255**
    (300x400), pinned inside the crop's own class bound
    (:data:`CROP_TOLERANCE`); the parse blur (:data:`PASTE_TOLERANCE`)
    rides inside the same envelope.
    """
    height, width = frame_hw
    frames = _synthetic_frames(2, height, width)

    base_out = _run(base_module, _DetectingHelper(helper_module, True), frames)
    shipped_out, marker = _run_marked(
        gpu_paste, shipped_module, _DetectingHelper(helper_module, True), frames
    )

    base = base_out.numpy()
    shipped = shipped_out.numpy()
    assert shipped.shape == base.shape, (
        f"the ROI paste changed the output shape for {height}x{width}: "
        f"{shipped.shape} != {base.shape}"
    )
    diff = np.abs(base.astype(np.float32) - shipped.astype(np.float32))
    assert diff.max() <= CROP_TOLERANCE / 255.0 + 1e-6, (
        f"the vectorized paste drifted {diff.max() * 255:.2f}/255 from the base "
        f"for {height}x{width}"
    )

    # the ROI marker: both frames took the vectorized chunk, none of them
    # fell back (the near-black branch of the old body is deleted)
    assert marker["frames"] == 2, marker
    assert marker["roi_frames"] == 2, marker
    assert marker["fallback_frames"] == 0, marker
    # retired by fix 2.1 (2026-10-30): no ROI patch travels any more - the
    # vectorized chunk warps and blends at the full canvas on the device
    assert marker["h2d_bytes"] == 0, marker
    assert marker["d2h_bytes"] == 0, marker
    assert marker["canvas_bytes"] == 0, marker
    # the restored faces are uploaded whole - they are not ROI-sized by nature
    assert marker["face_h2d_bytes"] == 2 * 512 * 512 * 3, marker


@pytest.mark.unit
def test_the_vendored_paste_branch_does_not_take_the_roi_path(
    helper_module, base_module, shipped_module, gpu_paste
):
    """Without a parse net the paste stays with the vendored helper.

    The ROI path is the **default branch** of the paste stage; the
    vendored ``paste_faces_to_input_image`` stays the fallback of the
    processor (a face helper without a parse net).  The ROI marker has
    to stay empty there - a filled marker would mean that the ROI path
    ran on a branch that has no parse mask to warp - and the pixels stay
    inside the device crop's documented envelope (:data:`CROP_TOLERANCE`,
    Opt 1) against the base: the vendored fallback is the reference of
    the suite and the crop is the one replaced op that reaches it.
    """
    frames = _synthetic_frames(2, 720, 1280)

    base_out = _run(base_module, _DetectingHelper(helper_module, False), frames)
    shipped_out, marker = _run_marked(
        gpu_paste, shipped_module, _DetectingHelper(helper_module, False), frames
    )

    assert marker["frames"] == 0, marker
    assert marker["roi_frames"] == 0, marker
    assert marker["fallback_frames"] == 0, marker
    assert marker["h2d_bytes"] == 0, marker
    assert marker["d2h_bytes"] == 0, marker
    assert marker["canvas_bytes"] == 0, marker

    diff = np.abs(
        base_out.numpy().astype(np.float32) - shipped_out.numpy().astype(np.float32)
    )
    assert diff.max() <= CROP_TOLERANCE / 255.0 + 1e-6, (
        f"the vendored paste branch drifted {diff.max() * 255:.2f}/255 "
        f"from the base"
    )


@pytest.mark.unit
@pytest.mark.parametrize("frame_hw", [(720, 1280), (300, 400)])
def test_the_roi_path_matches_the_full_canvas_path_within_the_measured_residual(
    helper_module, shipped_module, gpu_paste, monkeypatch, frame_hw
):
    """The end-to-end vectorized ROI path vs the full-canvas reference.

    The full-canvas paste (``use_roi=False``) is the reference (the
    unchanged ``_paste_chunk``).  Forcing it **inside the same pipeline**
    isolates the paste change - the crop, the restore and the background
    are the very same operations.  Fix 2.1 (2026-10-30, the owner's
    ``keep-fix3.txt``) vectorized the ROI chunk - the full-canvas
    ``grid_sample`` face warp replaced the cv2 fixed-point warp + ROI
    slicing, so the two paths are no longer byte-identical; the pin holds
    the measured residual instead: **2/255** (720x1280) and **1/255**
    (300x400), pinned at the 8/255 class bound of the batched warps.

    The small-frame case (``min(h, w) < 512``) is covered as well: its
    canvas is the 512 px upscaled frame of the vendored ``read_image``
    (512x683 here), so the warp runs at a second canvas geometry.
    """
    height, width = frame_hw
    frames = _synthetic_frames(2, height, width)

    roi_out, roi_marker = _run_marked(
        gpu_paste, shipped_module, _DetectingHelper(helper_module, True), frames
    )
    assert roi_marker["roi_frames"] == 2, roi_marker

    original = shipped_module.paste_faces_batch

    def _full_canvas_paste(*args, **kwargs):
        kwargs["use_roi"] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(shipped_module, "paste_faces_batch", _full_canvas_paste)
    full_out, full_marker = _run_marked(
        gpu_paste, shipped_module, _DetectingHelper(helper_module, True), frames
    )

    # the forced full-canvas paste is not the ROI path - the marker says so
    assert full_marker["frames"] == 2, full_marker
    assert full_marker["roi_frames"] == 0, full_marker
    assert full_marker["h2d_bytes"] == 0, full_marker
    assert full_marker["d2h_bytes"] == 0, full_marker

    diff = np.abs(
        roi_out.numpy().astype(np.float32) - full_out.numpy().astype(np.float32)
    ).max()
    assert diff <= CROP_TOLERANCE / 255.0 + 1e-6, (
        f"the vectorized ROI path drifted {diff * 255:.2f}/255 from the "
        f"full-canvas reference for the {height}x{width} pipeline"
    )
