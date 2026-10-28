"""The cancel pins (the cancel round, C1 - design section 2.4).

ComfyUI's Cancel button sets a global flag; the long loops of the pack
poll ``keep_interrupt.check_interrupted()`` (``src/keep_interrupt.py``),
which raises ``InterruptProcessingException`` - a **BaseException**
subclass - when the flag is set.  The pins here encode four facts:

1. the helper raises the mock's exception and resets the flag (the real
   ``throw_exception_if_processing_interrupted`` semantics,
   ``comfy/model_management.py:2214``), and is silent when the flag is
   clear;
2. the k-th check of ``process_image_sequence`` aborts the run - no KEEP
   work happens after the raise;
3. the cancel escapes **both** node ``execute``s - the exception flies
   through the ``try``/``finally`` (no ``except`` anywhere) and the
   ``finally`` offload still runs;
4. without a cancel the wired checks are no-ops: a patched no-op run is
   bit-identical to the unpatched run (the existing pins stay untouched
   and green).

The stubs follow the established patterns: ``_StubKeepNet`` /
``_ParseFaceHelper`` / ``_make_pack`` (``tests/test_gpu_paste_equivalence.py:1323-1405``)
and the real ``KEEPModelPack`` with stub residency hooks
(``tests/test_nodes_execute.py:48-63`` - the node checks the pack type).
Every test patches the **module attribute** ``keep_interrupt.check_interrupted``
- the single patch point the callers are contracted to use (the
module-qualified call).
"""

from __future__ import annotations

import importlib

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
def keep_interrupt():
    return importlib.import_module("ComfyUI_KEEP.src.keep_interrupt")


@pytest.fixture(scope="module")
def processor_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_processor")


@pytest.fixture(scope="module")
def loader_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_model_loader")


@pytest.fixture(scope="module")
def image_node_module():
    return importlib.import_module("ComfyUI_KEEP.nodes.keep_face_upscale_image")


@pytest.fixture(scope="module")
def sequence_node_module():
    return importlib.import_module("ComfyUI_KEEP.nodes.keep_process_image_sequence")


@pytest.fixture
def model_management():
    """The interrupt module (the mock of the gate env, the real one headless).

    The flag is session-global (one module): every test starts and ends
    with it cleared, so a cancelled run cannot leak into the next test.
    """
    module = importlib.import_module("comfy.model_management")
    module.interrupt_current_processing(False)
    yield module
    module.interrupt_current_processing(False)


# ---------------------------------------------------------------------------
# The stubs (the patterns of tests/test_gpu_paste_equivalence.py:1323-1405)
# ---------------------------------------------------------------------------


class _StubParseNet:
    """A parse net with the vendored contract (``[0]`` are the logits)."""

    def __init__(self, seed: int = 5) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(19, 3, 1, 1, generator=generator) * 0.35

    def __call__(self, samples):
        return (F.conv2d(samples, self.weight), None)


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

    def align_warp_face(self):
        # the crop of the single-image path - not reached by these pins
        # (the cancel raises before it), recorded for the completeness of
        # the call log
        self.calls.append("align_warp_face")

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


def _node_pack(loader_module, keep_net, face_helper, monkeypatch):
    """A real ``KEEPModelPack`` (the node checks the type) with stub hooks."""
    pack = loader_module.KEEPModelPack(
        keep_net=keep_net,
        face_helper=face_helper,
        bg_upscale_model=None,
        face_upscale_model=None,
        model_type_str="KEEP",
    )
    calls: list[str] = []
    monkeypatch.setattr(pack, "load_device", lambda: calls.append("load_device"))
    monkeypatch.setattr(pack, "offload", lambda: calls.append("offload"))
    monkeypatch.setattr(pack, "ensure_flow", lambda: calls.append("ensure_flow"))
    pack.calls = calls
    return pack


def _face_landmarks() -> np.ndarray:
    """One face: the 512 template at the in-frame position of the paste pins."""
    return (FACE_TEMPLATE - 256.0) * 0.1 + np.array([48.0, 32.0])


def _make_processor(processor_module, keep_net, face_helper):
    return processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))


def _run_sequence(processor, frames, **kwargs):
    return processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=kwargs.pop("final_upscale_factor", 1.0),
        has_aligned_frames=False,
        only_center_face=True,
        draw_box=kwargs.pop("draw_box", False),
        max_clip_length=kwargs.pop("max_clip_length", 20),
    )


def _cancel_on_call(model_management, cancel_at: int):
    """A ``check_interrupted`` stand-in that raises on the ``cancel_at``-th call.

    Returns ``(check, counter)`` - the callable to patch in and the dict
    that counts the calls it saw (the raise is the ``cancel_at``-th one).
    """
    counter = {"calls": 0}

    def check_interrupted():
        counter["calls"] += 1
        if counter["calls"] >= cancel_at:
            raise model_management.InterruptProcessingException()

    return check_interrupted, counter


# ---------------------------------------------------------------------------
# Pin 1 - the helper against the interrupt module (mock or real)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_helper_raises_and_resets_the_flag(model_management, keep_interrupt):
    """A set flag makes the helper raise - and clears it (the real semantics).

    ``InterruptProcessingException`` derives from **BaseException** (the
    real ``comfy/model_management.py:2196``): the broad ``except Exception``
    handlers of the pipeline must not be able to swallow a cancel.
    """
    assert issubclass(
        model_management.InterruptProcessingException, BaseException
    ), "the exception must be a BaseException subclass, or an except Exception could swallow the cancel"

    model_management.interrupt_current_processing(True)
    with pytest.raises(model_management.InterruptProcessingException):
        keep_interrupt.check_interrupted()
    assert not model_management.processing_interrupted(), (
        "the thrower resets the flag after raising "
        "(the real semantics, comfy/model_management.py:2214-2221)"
    )


@pytest.mark.unit
def test_the_helper_is_silent_when_the_flag_is_clear(model_management, keep_interrupt):
    """A clear flag (or no ComfyUI at all) makes the check a no-op."""
    model_management.interrupt_current_processing(False)
    assert keep_interrupt.check_interrupted() is None
    assert not model_management.processing_interrupted()


# ---------------------------------------------------------------------------
# Pin 2 - the k-th check aborts the sequence with no KEEP work after it
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_seventh_check_aborts_the_sequence_without_more_keep_work(
    processor_module, model_management, keep_interrupt, monkeypatch
):
    """The k-th cancel check escapes the processor mid-restore.

    5 frames with ``max_clip_length=2``: the checks fire once per detection
    frame (calls 1-5) and once per restore chunk (6-8).  Raising on the 7th
    call lets the first chunk restore (exactly one KEEP call) and kills
    everything after it - the second chunk's KEEP forward never happens.
    """
    check, counter = _cancel_on_call(model_management, 7)
    monkeypatch.setattr(keep_interrupt, "check_interrupted", check)
    keep_net = _StubKeepNet()
    face_helper = _ParseFaceHelper(parse_net=_StubParseNet(), landmarks=_face_landmarks())
    processor = _make_processor(processor_module, keep_net, face_helper)
    frames = torch.rand((5, 64, 96, 3), dtype=torch.float32)

    with pytest.raises(model_management.InterruptProcessingException):
        _run_sequence(processor, frames, max_clip_length=2)

    assert counter["calls"] == 7, (
        f"the checks must fire 5x in the detection loop and 2x in the "
        f"restore chunk loop, saw {counter['calls']}"
    )
    assert keep_net.calls == [((1, 2, 3, 512, 512), False)], (
        f"exactly the first restore chunk may reach the KEEP net, "
        f"got {keep_net.calls}"
    )


# ---------------------------------------------------------------------------
# Pin 3 - the cancel escapes both nodes; the finally offload still ran
# ---------------------------------------------------------------------------
@pytest.mark.mock_integration
def test_the_cancel_escapes_the_sequence_node_and_the_pack_is_offloaded(
    sequence_node_module, loader_module, model_management, keep_interrupt, monkeypatch
):
    """The exception flies through the node's ``try``/``finally`` unharmed."""
    check, _ = _cancel_on_call(model_management, 1)
    monkeypatch.setattr(keep_interrupt, "check_interrupted", check)
    keep_net = _StubKeepNet()
    face_helper = _ParseFaceHelper(parse_net=_StubParseNet(), landmarks=_face_landmarks())
    pack = _node_pack(loader_module, keep_net, face_helper, monkeypatch)

    with pytest.raises(model_management.InterruptProcessingException):
        sequence_node_module.KEEP_ProcessImageSequenceNode.execute(
            torch.rand((2, 64, 96, 3), dtype=torch.float32),
            pack,
            final_upscale_factor=1.0,
            has_aligned_frames=False,
            only_center_face=True,
            draw_bounding_box=False,
            max_clip_length=20,
        )

    assert pack.calls == ["load_device", "ensure_flow", "offload"], (
        f"the finally offload must run after the cancel, got {pack.calls}"
    )
    assert keep_net.calls == [], f"no KEEP work after the raise, got {keep_net.calls}"


@pytest.mark.mock_integration
def test_the_cancel_escapes_the_image_node_and_the_pack_is_offloaded(
    image_node_module, loader_module, model_management, keep_interrupt, monkeypatch
):
    """The same escape through the single-image node (the check after detection)."""
    check, _ = _cancel_on_call(model_management, 1)
    monkeypatch.setattr(keep_interrupt, "check_interrupted", check)
    keep_net = _StubKeepNet()
    face_helper = _ParseFaceHelper(parse_net=_StubParseNet(), landmarks=_face_landmarks())
    pack = _node_pack(loader_module, keep_net, face_helper, monkeypatch)

    with pytest.raises(model_management.InterruptProcessingException):
        image_node_module.KEEP_FaceUpscaleImageNode.execute(
            torch.rand((1, 64, 96, 3), dtype=torch.float32),
            pack,
            final_upscale_factor=1.0,
            has_aligned_face=False,
            only_center_face=True,
            draw_bounding_box=False,
        )

    assert pack.calls == ["load_device", "offload"], (
        f"the finally offload must run after the cancel, got {pack.calls}"
    )
    assert keep_net.calls == [], f"no KEEP work after the raise, got {keep_net.calls}"


# ---------------------------------------------------------------------------
# Pin 4 - without a cancel the checks are no-ops and nothing changes
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_no_cancel_run_is_bit_identical_to_a_run_without_the_checks(
    processor_module, keep_interrupt, monkeypatch
):
    """A patched no-op run equals the unpatched run (the wiring does no harm).

    The unpatched run already exercises the real helper end to end (a
    no-op: the flag is clear in the gate env, and the helper returns
    without ComfyUI).  The patched run replaces it with a counting no-op -
    same pixels, same KEEP calls, and the counter proves the loops really
    poll.
    """
    frames = torch.rand(
        (5, 64, 96, 3), generator=torch.Generator().manual_seed(11), dtype=torch.float32
    )

    unpatched_net = _StubKeepNet()
    unpatched = _run_sequence(
        _make_processor(
            processor_module, unpatched_net,
            _ParseFaceHelper(parse_net=_StubParseNet(), landmarks=_face_landmarks()),
        ),
        frames,
    )

    counter = {"calls": 0}

    def noop_check():
        counter["calls"] += 1

    monkeypatch.setattr(keep_interrupt, "check_interrupted", noop_check)

    patched_net = _StubKeepNet()
    patched = _run_sequence(
        _make_processor(
            processor_module, patched_net,
            _ParseFaceHelper(parse_net=_StubParseNet(), landmarks=_face_landmarks()),
        ),
        frames,
    )

    # the deterministic minimum: 5 detection frames + 1 restore chunk + 5
    # paste-loop iterations (the paste chunk loop adds environment-sized
    # chunks on top)
    assert counter["calls"] >= 11, (
        f"the checks must be wired into the loops, saw {counter['calls']}"
    )
    assert torch.equal(unpatched, patched), "a no-op poll must not change the pixels"
    assert unpatched_net.calls == patched_net.calls, (
        "a no-op poll must not change the KEEP calls"
    )
