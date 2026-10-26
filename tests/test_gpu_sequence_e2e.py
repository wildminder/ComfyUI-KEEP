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
- the **pixels** - the vendored-fallback branch (no parse net) is bit-exact
  against the base; the batched paste branch stays inside the documented
  tolerance of the parse mask blur (the only replaced op with a non-zero
  deviation).

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
#: parse mask blur of the device (the only replaced op with a deviation).
PASTE_TOLERANCE = 1


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

    Without a parse net the shipped pipeline falls back to the vendored
    paste and has to be **bit-exact** against the base.  With the parse net
    the batched paste replaces the vendored one; the only replaced op with a
    deviation is the parse mask blur of the device, bounded by
    :data:`PASTE_TOLERANCE` (1/255).
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
    tolerance = 0.0 if not use_parse else PASTE_TOLERANCE / 255.0
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
