"""P5.4 - the opt-in GPU acceptance test (real weights, real GPU).

The test is **opt-in**: it needs ``RUN_KEEP_E2E=1`` **and**
``KEEP_MODEL_DIR=<absolute path>`` (the **models root** with the real
weights - ``keep_models/KEEP/``, ``facedetection/``, ``gmflow/``) and
``COMFYUI_ROOT=<path to the ComfyUI checkout>`` (the real ``comfy`` modules -
the headless bootstrap of P1.4 drives the pack **without a server**) and
``KEEP_NO_COMFY_MOCKS=1`` (the harness must not replace the real ComfyUI
modules with its mocks - an acceptance run against the mocks would prove
nothing).

Without those variables the test **skips cleanly**, naming them - the
acceptance gate runs without environment variables and must never pull a
gigabyte from upstream.

What it proves (one factor varied per case, a fixed seed):

* the single image node restores a synthetic face-like frame - the output has
  the expected shape and dtype, carries no NaN and differs from the input by a
  measurable margin (the regression class of the random-weight flownet);
* the sequence node restores five synthetic frames - the same assertions on
  the frames;
* the VRAM returns to its baseline after the release (``unload`` +
  ``soft_empty_cache``);
* the optical flow net carries the **trained** weights: the tensors of the
  flownet are compared with the tensors of their source (the KEEP checkpoint
  or the GMFlow file) - random weights cannot pass that comparison.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

#: The environment variable that arms the test.
RUN_E2E_ENV = "RUN_KEEP_E2E"

#: The environment variable with the directory of the real weights.
MODEL_DIR_ENV = "KEEP_MODEL_DIR"

#: The environment variable with the ComfyUI checkout.
COMFYUI_ROOT_ENV = "COMFYUI_ROOT"

#: The environment variable that keeps the harness mocks out of the way.
NO_MOCKS_ENV = "KEEP_NO_COMFY_MOCKS"

#: The values of :data:`RUN_E2E_ENV` that count as "on".
TRUTHY = frozenset({"1", "true", "yes", "on"})

#: The seed of the synthetic frames (the run must be reproducible).
SEED = 20261023

#: The size of the synthetic (aligned) face frames.
FRAME_SIZE = 512

pytestmark = pytest.mark.gpu_acceptance


def _load_helper():
    """The headless bootstrap helper (loaded by path, like P1.4 does)."""
    path = Path(__file__).resolve().parent / "_headless.py"
    spec = importlib.util.spec_from_file_location("keep_headless_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _missing_environment() -> list[str]:
    """The environment variables that are missing (empty when it may run)."""
    missing: list[str] = []
    if (os.environ.get(RUN_E2E_ENV) or "").strip().lower() not in TRUTHY:
        missing.append(f"{RUN_E2E_ENV}=1")
    if not (os.environ.get(MODEL_DIR_ENV) or "").strip():
        missing.append(f"{MODEL_DIR_ENV}=<absolute path to the KEEP weights>")
    if not (os.environ.get(COMFYUI_ROOT_ENV) or "").strip():
        missing.append(f"{COMFYUI_ROOT_ENV}=<path to the ComfyUI checkout>")
    if (os.environ.get(NO_MOCKS_ENV) or "").strip().lower() not in TRUTHY:
        missing.append(
            f"{NO_MOCKS_ENV}=1 (the acceptance run needs the **real** ComfyUI "
            "modules, not the harness mocks)"
        )
    return missing


@pytest.fixture(scope="module")
def e2e():
    """The headless environment of the acceptance run (or a clean skip)."""
    missing = _missing_environment()
    if missing:
        pytest.skip(
            "the GPU acceptance test is opt-in - set " + " and ".join(missing)
        )

    helper = _load_helper()
    root = helper.resolve_comfyui_root()
    # the real ComfyUI modules have to be importable before the package is
    # executed (its entry point imports ``comfy_api.latest``)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    helper.load_extra_model_paths(str(root))
    package_dir = Path(__file__).resolve().parent.parent
    alias = helper.register_package(package_dir, execute=True)
    helper.stub_ui()

    # P5.4 (review fix): ``KEEP_MODEL_DIR`` must really decide where the
    # weights come from.  The pack resolves every weight through ComfyUI's
    # ``folder_paths`` (the primary root is ``folder_paths.models_dir``), so
    # the models root of the run **is** the models root of the pack.
    import folder_paths  # noqa: PLC0415 - the real ComfyUI module

    model_dir = helper.resolve_model_dir()
    folder_paths.models_dir = str(model_dir)
    paths_module = importlib.import_module(
        f"{alias.__name__}.src.keep_paths"
    )
    paths_module.register_model_folders()

    return {
        "root": root,
        "model_dir": model_dir,
        "package": alias,
    }


@pytest.fixture(scope="module")
def keep_pack(e2e):
    """The real model pack (real weights, real GPU)."""
    import torch  # noqa: PLC0415 - the opt-in environment

    loader_module = importlib.import_module(
        f"{e2e['package'].__name__}.src.keep_model_loader"
    )
    loader = loader_module.KEEPModelLoader()
    pack = loader.load_keep_model_pack("KEEP", "retinaface_resnet50")
    return pack


def _face_like_frame(index: int = 0):
    """A deterministic synthetic face-like frame ``(1, 512, 512, 3)``.

    The frames are *aligned* faces (the ``has_aligned_face`` path), so the
    test does not depend on a detector finding a face in a synthetic image.
    """
    import torch  # noqa: PLC0415 - the opt-in environment

    generator = torch.Generator().manual_seed(SEED + index)
    size = FRAME_SIZE
    frame = torch.rand((1, size, size, 3), generator=generator, dtype=torch.float32)
    # a simple face-like structure: an oval, two eyes and a mouth
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    centre_x, centre_y = size / 2, size / 2
    oval = (((xx - centre_x) / (size * 0.30)) ** 2 + ((yy - centre_y) / (size * 0.40)) ** 2) < 1.0
    for eye_x in (centre_x - size * 0.11, centre_x + size * 0.11):
        eye = ((xx - eye_x) ** 2 + (yy - (centre_y - size * 0.08)) ** 2) < (size * 0.035) ** 2
        frame[0, ..., 0][eye] = 0.05
    mouth = (((xx - centre_x) / (size * 0.10)) ** 2 + ((yy - (centre_y + size * 0.18)) / (size * 0.02)) ** 2) < 1.0
    frame[0, ..., 2][mouth] = 0.05
    frame[0, ..., 1][oval] = frame[0, ..., 1][oval] * 0.9 + 0.1
    return frame


def _assert_sane(output, expected_shape):
    """The shape / dtype / NaN assertions of the restored frames."""
    import torch  # noqa: PLC0415 - the opt-in environment

    assert tuple(output.shape) == tuple(expected_shape), output.shape
    assert output.dtype == torch.float32, output.dtype
    assert not torch.isnan(output).any(), "the output carries NaN"
    assert float(output.min()) >= -1e-3 and float(output.max()) <= 1.0 + 1e-3


def _margin(input_frame, output_frame) -> float:
    """The mean absolute difference of the two frames."""
    import torch  # noqa: PLC0415 - the opt-in environment

    return float((input_frame - output_frame).abs().mean())


@pytest.mark.gpu_acceptance
def test_the_weights_come_from_the_model_dir(e2e, keep_pack):
    """``KEEP_MODEL_DIR`` really decides where the weights come from.

    The pack resolves every weight through ComfyUI's ``folder_paths`` (the
    primary root is ``folder_paths.models_dir``), and the models root of the
    run **is** ``KEEP_MODEL_DIR``.  A weight from anywhere else (the ComfyUI
    checkout, a stale extra model root) would mean that the environment
    variable has no effect - the defect the P5.4 review found.
    """
    configs = importlib.import_module(
        f"{e2e['package'].__name__}.src.keep_model_configs"
    )
    download = importlib.import_module(
        f"{e2e['package'].__name__}.src.keep_download"
    )
    entry = configs.weight_entry_for("KEEP")
    path = download.find_weight(entry)
    assert path is not None, "the KEEP checkpoint is not in the model roots"

    model_dir = Path(e2e["model_dir"]).resolve()
    assert Path(path).resolve().is_relative_to(model_dir), (path, model_dir)


@pytest.mark.gpu_acceptance
def test_the_single_image_node_restores_a_face(e2e, keep_pack):
    """The single image path restores the frame measurably."""
    import torch  # noqa: PLC0415 - the opt-in environment

    node_module = importlib.import_module(
        f"{e2e['package'].__name__}.nodes.keep_face_upscale_image"
    )
    frame = _face_like_frame()
    try:
        output = node_module.KEEP_FaceUpscaleImageNode.execute(
            image=frame,
            keep_model=keep_pack,
            final_upscale_factor=1.0,
            has_aligned_face=True,
            only_center_face=True,
            draw_bounding_box=False,
        ).result[0]
    finally:
        keep_pack.offload()

    _assert_sane(output, frame.shape)
    # the restoration must change the frame (the random-weight regression
    # class would leave it untouched)
    assert _margin(frame, output) > 1e-3, "the restored frame is identical"


@pytest.mark.gpu_acceptance
def test_the_sequence_node_restores_five_frames(e2e, keep_pack):
    """The sequence path restores five frames measurably."""
    import torch  # noqa: PLC0415 - the opt-in environment

    node_module = importlib.import_module(
        f"{e2e['package'].__name__}.nodes.keep_process_image_sequence"
    )
    frames = torch.cat([_face_like_frame(index) for index in range(5)], dim=0)
    try:
        output = node_module.KEEP_ProcessImageSequenceNode.execute(
            images=frames,
            keep_model=keep_pack,
            final_upscale_factor=1.0,
            has_aligned_frames=True,
            only_center_face=True,
            draw_bounding_box=False,
            max_clip_length=20,
        ).result[0]
    finally:
        keep_pack.offload()

    _assert_sane(output, frames.shape)
    margins = [_margin(frames[i], output[i]) for i in range(frames.shape[0])]
    assert min(margins) > 1e-3, margins


@pytest.mark.gpu_acceptance
def test_the_vram_returns_to_the_baseline(e2e, keep_pack):
    """The release gives the VRAM back (unload + ``soft_empty_cache``)."""
    import torch  # noqa: PLC0415 - the opt-in environment

    from comfy import model_management  # noqa: PLC0415 - the real ComfyUI

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device - the VRAM assertion needs a GPU")

    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    keep_pack.load_device()
    peak = torch.cuda.memory_allocated()
    keep_pack.offload()
    model_management.soft_empty_cache()
    after = torch.cuda.memory_allocated()

    assert peak > baseline, "the models did not move to the GPU"
    assert after <= baseline + 1024 * 1024, (baseline, after)


@pytest.mark.gpu_acceptance
def test_the_flow_weights_load_strictly(e2e, keep_pack):
    """The optical flow net carries the **trained** weights (P5.5).

    The flow weights come from the KEEP checkpoint (the shipped one carries
    them) or from the GMFlow file.  The guarantee is the **strict** load of
    the whole checkpoint (the loader raises otherwise) - and this test proves
    the second half of it: the tensors of the flownet are the tensors of
    their source, bit for bit.  Randomly initialised weights (the
    regression class) cannot pass that comparison - a "the weights are not
    all zero" assertion would, because a random net is not all zero either.
    """
    import torch  # noqa: PLC0415 - the opt-in environment

    flownet = keep_pack.flownet
    assert flownet is not None, "the KEEP net has no flownet"

    # the tensors of the flownet, with the prefix they have in the net
    actual = {
        "flownet." + key: tensor for key, tensor in flownet.state_dict().items()
    }
    assert actual, "the flownet has no weights"

    if keep_pack.flownet_from_checkpoint:
        source = _checkpoint_flownet_tensors(e2e)
    else:
        path = keep_pack.flownet_path
        assert path, (
            "neither the checkpoint nor the GMFlow file provided the flow weights"
        )
        source = _gmflow_tensors(path)

    assert set(source) == set(actual), sorted(set(source) ^ set(actual))[:10]
    for key, tensor in source.items():
        assert torch.equal(actual[key].cpu(), tensor.cpu()), key


def _checkpoint_flownet_tensors(e2e) -> dict:
    """The ``flownet.*`` tensors of the KEEP checkpoint (the source of truth).

    The checkpoint is resolved through the **production** path (the manifest
    and the downloader), so the test compares against the file the pack has
    really loaded.
    """
    import torch  # noqa: PLC0415 - the opt-in environment

    configs = importlib.import_module(
        f"{e2e['package'].__name__}.src.keep_model_configs"
    )
    download = importlib.import_module(
        f"{e2e['package'].__name__}.src.keep_download"
    )
    entry = configs.weight_entry_for("KEEP")
    path = download.find_weight(entry)
    assert path, "the KEEP checkpoint is not in the model roots"

    try:  # the memory-mapped load keeps the peak small
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except (RuntimeError, ValueError):  # pragma: no cover - old save format
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    return {
        key: value for key, value in state.items() if key.startswith("flownet.")
    }


def _gmflow_tensors(path: str) -> dict:
    """The tensors of the GMFlow file (the source of truth)."""
    import torch  # noqa: PLC0415 - the opt-in environment

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model", checkpoint)
    return {"flownet.model." + key: value for key, value in weights.items()}
