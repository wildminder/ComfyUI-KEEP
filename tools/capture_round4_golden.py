"""Round 4 golden capture - the pre-restructure outputs of the processor (D-R2.0).

This tool is a **one-shot historical record**: it runs the four golden cases
of the round-4 design (``D-R2.0``) through the *current*
``KEEPFaceProcessor`` and writes the outputs to
``tests/fixtures/round4_golden_<case>.npz``.  The memory-bounds restructure
(D-R2.1 - D-R2.4) must reproduce these bytes exactly; the P-M1 pins load the
fixtures and compare.  Re-running the tool after the restructure would
overwrite the baseline with the new behavior - do not do that (the fixtures
are tracked; restore them from git instead).

The cases (design section D-R2.0):

* ``single_seq``      - 6 uniform 256x256 frames (sub-512: the vendored
  ``read_image`` upscale - and therefore the ``needs_upscale`` flag - fires
  on **every** frame; a mixed-size IMAGE batch is unconstructible), one face
  per frame, ``max_clip_length=4``, frame 1 grayscale (the vendored-fallback
  paste path);
* ``multiface_seq``   - 5 frames with 3/1/0/2/1 faces (7 faces -> restore
  chunks 3/3/1, the 1-face duplicate-to-2 trick), ``max_clip_length=3``,
  frame 2 grayscale **and** face-less (the background gather of the paste);
* ``image_single``    - one 256x256 image, one face, the single-face path;
* ``image_multiface`` - one 256x256 image, two faces, the multi-face path.

The stubs are the exact seams of ``tests/test_multiface_sequence.py`` (the
real vendored helper with stubbed model inits, the ``_SequenceDetector``
per-frame face sets, the identity ``_KeepNet``, the deterministic
``_StubParseNet``), loaded from the test module file - the tests are not a
package.  The ComfyUI mocks come from the root ``conftest`` (importing it
installs them).

Run with the embedded interpreter:

    COMFYUI_ROOT=<ComfyUI source> python tools/capture_round4_golden.py

Exit code 0 when all four fixtures were written.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing the root conftest installs the ComfyUI mocks and the package
# alias - the vendored helper imports ``comfy.model_management`` at module
# import time, so this must happen before any pack import below.
import conftest  # noqa: E402 - the mock install is the point of the import

#: The uniform frame size of the sequence cases: sub-512 on the short side,
#: so every frame goes through the vendored upscale (the ``needs_upscale``
#: crop-source path the restructure must re-derive byte-identically).
GOLDEN_HW = (256, 256)

#: The gray level of the gray frames (the value of the gray-frame pins).
GRAY_LEVEL = 102

#: Parse stub seed per case - any fixed values; the same seeds are used by
#: the P-M1 pins that load the fixtures.
PARSE_SEEDS = {
    "single_seq": 211,
    "multiface_seq": 223,
    "image_single": 227,
    "image_multiface": 229,
}

#: The face positions of the golden frames, by name (256-frame coordinates).
FACE_OFFSETS = {
    "a": [128.0, 112.0],
    "b": [136.0, 124.0],
    "c": [96.0, 160.0],
}

#: The sequence cases - the single source of truth for the capture tool and
#: for the P-M1 pins that re-run them (the faces name :data:`FACE_OFFSETS`).
SEQUENCE_CASES = {
    "single_seq": {
        "faces": [["a"], ["b"], ["a"], ["c"], ["a"], ["b"]],
        "gray_index": 1,
        "max_clip_length": 4,
        "only_center_face": True,
        "frame_seed": 61,
    },
    "multiface_seq": {
        "faces": [["a", "b", "c"], ["a"], [], ["b", "c"], ["a"]],
        "gray_index": 2,
        "max_clip_length": 3,
        "only_center_face": False,
        "frame_seed": 67,
    },
}

#: The single-image cases (the same contract, one image each).
IMAGE_CASES = {
    "image_single": {"faces": ["a"], "only_center_face": True, "frame_seed": 73},
    "image_multiface": {"faces": ["a", "b"], "only_center_face": False, "frame_seed": 79},
}


def golden_face(stubs, name: str) -> np.ndarray:
    """A 5-point face by :data:`FACE_OFFSETS` name in a 256x256 frame."""
    return (stubs.FACE_TEMPLATE - 256.0) * 0.25 + np.asarray(
        FACE_OFFSETS[name], dtype=np.float64
    )


def golden_frames(count: int, seed: int) -> np.ndarray:
    """Smooth seeded frames of :data:`GOLDEN_HW` (the ``_smooth_frames`` recipe)."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(count):
        small = rng.uniform(0.0, 255.0, size=(32, 32, 3)).astype(np.float32)
        smooth = cv2.resize(small, (GOLDEN_HW[1], GOLDEN_HW[0]), interpolation=cv2.INTER_CUBIC)
        frames.append(np.clip(smooth, 0, 255).astype(np.uint8))
    return np.stack(frames)


def sequence_frames(case: dict) -> np.ndarray:
    """The input frames of a sequence case (the gray frame substituted)."""
    frames = golden_frames(len(case["faces"]), case["frame_seed"])
    if case["gray_index"] is not None:
        frames[case["gray_index"]] = np.full((*GOLDEN_HW, 3), GRAY_LEVEL, dtype=np.uint8)
    return frames


def _load_test_stubs():
    """The stubs of ``tests/test_multiface_sequence.py`` (not a package)."""
    source = REPO_ROOT / "tests" / "test_multiface_sequence.py"
    spec = importlib.util.spec_from_file_location("_round4_golden_stubs", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CountingParseNet:
    """The parse stub plus the call count the fixtures record."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0

    def __call__(self, samples):
        self.calls += 1
        return self._inner(samples)


def _make_helper(stubs, helper_module, per_frame_faces, seed):
    """A real vendored helper with the stubbed detector / parse net."""
    face_helper = helper_module.FaceRestoreHelper(
        upscale_factor=1, face_size=512, crop_ratio=(1, 1),
        det_model="retinaface_resnet50", save_ext="png", use_parse=True,
        device="cpu", model_rootpath="unused",
    )
    face_helper.face_parse = _CountingParseNet(stubs._StubParseNet(seed=seed))
    face_helper.face_detector = stubs._SequenceDetector(per_frame_faces)
    return face_helper


def _keep_records(keep_net) -> tuple[np.ndarray, np.ndarray]:
    """The ``_KeepNet`` call log as fixture arrays (shapes + flags)."""
    shapes = np.array([shape for shape, _ in keep_net.calls], dtype=np.int64)
    flags = np.array([flag for _, flag in keep_net.calls], dtype=bool)
    return shapes, flags


def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    size_kb = path.stat().st_size / 1024
    print(f"  wrote {path.name} ({size_kb:.0f} KiB)")


def capture_sequence_case(name, stubs, helper_module, image_utils, out_dir: Path) -> None:
    """One sequence case: run, record, write the npz."""
    case = SEQUENCE_CASES[name]
    faces = {key: golden_face(stubs, key) for key in FACE_OFFSETS}
    per_frame = [[faces[key] for key in frame] for frame in case["faces"]]
    frames_bgr = sequence_frames(case)
    tensor = image_utils.bgr_u8_batch_to_comfy(frames_bgr)

    face_helper = _make_helper(stubs, helper_module, per_frame, PARSE_SEEDS[name])
    keep_net = stubs._KeepNet()
    pack = stubs._make_pack(keep_net, face_helper)
    processor = importlib.import_module("ComfyUI_KEEP.src.keep_processor")
    output = processor.KEEPFaceProcessor(pack).process_image_sequence(
        image_sequence_tensor=tensor,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=case["only_center_face"],
        draw_box=False,
        max_clip_length=case["max_clip_length"],
    )

    shapes, flags = _keep_records(keep_net)
    print(
        f"  {name}: input {tuple(tensor.shape)} -> output {tuple(output.shape)}, "
        f"{len(keep_net.calls)} KEEP calls, {face_helper.face_parse.calls} parse calls"
    )
    _save(
        out_dir / f"round4_golden_{name}.npz",
        input_frames=tensor.detach().cpu().numpy(),
        output_frames=output.detach().cpu().numpy(),
        keep_net_shapes=shapes,
        keep_net_need_upscale=flags,
        parse_calls=np.array(face_helper.face_parse.calls, dtype=np.int64),
        max_clip_length=np.array(case["max_clip_length"], dtype=np.int64),
        only_center_face=np.array(case["only_center_face"], dtype=bool),
    )


def capture_image_case(name, stubs, helper_module, out_dir: Path) -> None:
    """One single-image case: run, record, write the npz."""
    case = IMAGE_CASES[name]
    image_bgr = golden_frames(1, case["frame_seed"])[0]
    faces = {key: golden_face(stubs, key) for key in FACE_OFFSETS}
    face_helper = _make_helper(
        stubs, helper_module, [[faces[key] for key in case["faces"]]], PARSE_SEEDS[name])
    keep_net = stubs._KeepNet()
    pack = stubs._make_pack(keep_net, face_helper)
    processor = importlib.import_module("ComfyUI_KEEP.src.keep_processor")
    output = processor.KEEPFaceProcessor(pack).process_image(
        cv2_image_orig=image_bgr,
        final_upscale_factor=1.0,
        has_aligned=False,
        only_center_face=case["only_center_face"],
        draw_box=False,
    )

    shapes, flags = _keep_records(keep_net)
    print(
        f"  {name}: input {image_bgr.shape} -> output {output.shape}, "
        f"{len(keep_net.calls)} KEEP calls, {face_helper.face_parse.calls} parse calls"
    )
    _save(
        out_dir / f"round4_golden_{name}.npz",
        input_image=image_bgr,
        output_image=np.asarray(output),
        keep_net_shapes=shapes,
        keep_net_need_upscale=flags,
        parse_calls=np.array(face_helper.face_parse.calls, dtype=np.int64),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "tests" / "fixtures"),
        help="the fixture directory (default: tests/fixtures)")
    args = parser.parse_args()
    out_dir = Path(args.out)

    stubs = _load_test_stubs()
    helper_module = importlib.import_module(
        "ComfyUI_KEEP.modules.deps.facelib.utils.face_restoration_helper")
    # The weights are not on this machine - the exact seam of ``make_helper``.
    helper_module.init_detection_model = lambda *a, **k: object()
    helper_module.init_parsing_model = lambda *a, **k: object()
    image_utils = importlib.import_module("ComfyUI_KEEP.src.keep_image_utils")

    print("capturing the round-4 golden fixtures from the current processor:")
    for name in SEQUENCE_CASES:
        capture_sequence_case(name, stubs, helper_module, image_utils, out_dir)
    for name in IMAGE_CASES:
        capture_image_case(name, stubs, helper_module, out_dir)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
