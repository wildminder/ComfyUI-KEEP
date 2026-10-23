"""P4.5 - the model residency of the pack (ModelPatcher) and ``ensure_flow``.

The pack wraps its three model components (the KEEP net with its flownet, the
face detector, the face parser) in ``ModelPatcher`` instances and lets ComfyUI
arbitrate the VRAM.  The tests run against the **mocked**
``comfy.model_management`` of the conftest: every residency call is recorded,
so the tests can assert what the pack asks ComfyUI to do - and that it asks
exactly once per execute.
"""

from __future__ import annotations

import importlib

import pytest
import torch


@pytest.fixture(scope="module")
def loader_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_model_loader")


@pytest.fixture(scope="module")
def node_module():
    return importlib.import_module("ComfyUI_KEEP.nodes.keep_face_upscale_image")


@pytest.fixture
def model_management():
    """The mocked ``comfy.model_management`` (the calls are reset per test)."""
    module = importlib.import_module("comfy.model_management")
    module.calls.clear()
    return module


class _StubModule:
    """A minimal stand-in for a torch module (it only has to move)."""

    def __init__(self, name: str = "module") -> None:
        self.name = name
        self.moved_to: list = []

    def to(self, device):
        self.moved_to.append(device)
        return self


class _StubKeepNet(_StubModule):
    """The KEEP net - it owns the flownet as a submodule."""

    def __init__(self) -> None:
        super().__init__("keep_net")
        self.flownet = _StubModule("flownet")


class _StubFaceHelper:
    """The facelib helper - it owns the detector and the parser."""

    def __init__(self) -> None:
        self.face_detector = _StubModule("detector")
        self.face_parse = _StubModule("parser")
        self.device = None


@pytest.fixture
def stub_models():
    return _StubKeepNet(), _StubFaceHelper()


@pytest.fixture
def pack(loader_module, stub_models):
    """A real ``KEEPModelPack`` around the stub models."""
    keep_net, face_helper = stub_models
    return loader_module.KEEPModelPack(
        keep_net=keep_net,
        face_helper=face_helper,
        bg_upscale_model=None,
        face_upscale_model=None,
        model_type_str="KEEP",
    )


# ---------------------------------------------------------------------------
# the patchers
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_pack_builds_three_patchers(pack):
    """Three patchers: the KEEP net, the face detector and the face parser."""
    assert len(pack.patchers) == 3, pack.patchers
    assert pack.keep_patcher.model is pack.keep_net
    assert pack.detector_patcher.model is pack.face_helper.face_detector
    assert pack.parser_patcher.model is pack.face_helper.face_parse


@pytest.mark.unit
def test_exactly_one_patcher_contains_the_flownet(pack):
    """The flownet belongs to the KEEP net - it must not be patched twice.

    A second patcher for the flownet would count its VRAM twice (the plan
    P4.5).
    """
    assert pack.flownet is pack.keep_net.flownet
    assert pack.keep_patcher.model.flownet is pack.flownet
    # the flownet is not the model of a patcher of its own
    assert all(patcher.model is not pack.flownet for patcher in pack.patchers)
    assert pack.detector_patcher.model is not pack.flownet
    assert pack.parser_patcher.model is not pack.flownet


@pytest.mark.unit
def test_the_patchers_are_shared_by_the_packs_of_the_same_models(
    loader_module, stub_models
):
    """The patchers belong to the models - they are built once per build."""
    keep_net, face_helper = stub_models
    first = loader_module.KEEPModelPack(keep_net, face_helper, None, None, "KEEP")
    second = loader_module.KEEPModelPack(
        keep_net, face_helper, None, None, "KEEP", patchers=first.patchers_by_role
    )
    assert second.keep_patcher is first.keep_patcher
    assert second.patchers == first.patchers


# ---------------------------------------------------------------------------
# the residency
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_load_device_lets_comfyui_arbitrate_the_vram(pack, model_management):
    """The pack hands all its patchers to ``load_models_gpu`` - once."""
    pack.load_device()

    load_calls = [
        call for call in model_management.calls if call[0] == "load_models_gpu"
    ]
    assert len(load_calls) == 1, model_management.calls
    _, models, memory_required = load_calls[0]
    assert list(models) == list(pack.patchers)
    assert memory_required == 0


@pytest.mark.unit
def test_offload_moves_the_models_back_and_frees_the_vram(pack, model_management):
    """The release path: the models go back to the offload device and ComfyUI
    frees the VRAM."""
    pack.offload()

    names = [call[0] for call in model_management.calls]
    assert "unload_all_models" in names, model_management.calls
    assert "soft_empty_cache" in names, model_management.calls
    # the models were moved back to the offload device
    assert pack.keep_net.moved_to[-1] == pack.offload_device
    assert pack.face_helper.face_detector.moved_to[-1] == pack.offload_device
    assert pack.face_helper.face_parse.moved_to[-1] == pack.offload_device


@pytest.mark.unit
def test_the_cpu_mode_makes_the_same_calls(
    loader_module, model_management, monkeypatch
):
    """``get_torch_device()`` -> cpu: the same residency calls are made.

    The conftest mock answers ``cpu`` by default (``conftest.py``: the
    ``get_torch_device`` / ``unet_offload_device`` of the mocked
    ``comfy.model_management``), so patching the device to ``cpu`` alone
    would exercise the default path and prove nothing.  The CPU leg is
    therefore compared with a **GPU leg**: both devices must produce the
    same residency calls - the pack must not branch on the device.  A
    device dependent path is exactly the B7 CPU crash (the pack called
    ``torch.cuda.empty_cache()`` and crashed on a CPU-only host).
    """

    def residency_calls(device):
        monkeypatch.setattr(model_management, "get_torch_device", lambda: device)
        monkeypatch.setattr(model_management, "unet_offload_device", lambda: device)
        model_management.calls.clear()

        pack = loader_module.KEEPModelPack(
            _StubKeepNet(), _StubFaceHelper(), None, None, "KEEP"
        )
        # the patched device really reaches the pack - the leg is not a no-op
        assert pack.device == device
        assert pack.offload_device == device

        pack.load_device()
        pack.offload()
        return [call[0] for call in model_management.calls]

    gpu_calls = residency_calls(torch.device("cuda"))
    cpu_calls = residency_calls(torch.device("cpu"))

    assert cpu_calls == gpu_calls, (cpu_calls, gpu_calls)
    assert cpu_calls.count("load_models_gpu") == 1, cpu_calls
    assert cpu_calls.count("unload_all_models") == 1, cpu_calls
    assert cpu_calls.count("soft_empty_cache") == 1, cpu_calls


# ---------------------------------------------------------------------------
# the node around the pack (mock-integration)
# ---------------------------------------------------------------------------
def _patch_the_node(monkeypatch, node_module, processor_cls):
    """Stub the heavy parts of the single image node."""
    monkeypatch.setattr(node_module, "KEEPFaceProcessor", processor_cls)
    monkeypatch.setattr(node_module, "comfy_image_to_cv2", lambda image: image)
    monkeypatch.setattr(node_module, "cv2_to_comfy_image", lambda image: image)
    monkeypatch.setattr(node_module, "report_progress", lambda *args: True)


class _StubProcessor:
    """A processor that returns a stub image."""

    def __init__(self, pack) -> None:
        self.pack = pack

    def process_image(self, **kwargs):
        return _StubModule("processed")


class _FailingProcessor:
    """A processor that fails (the node must give the VRAM back)."""

    def __init__(self, pack) -> None:
        self.pack = pack

    def process_image(self, **kwargs):
        raise RuntimeError("the processing failed")


@pytest.mark.mock_integration
def test_one_execute_asks_for_the_vram_once_and_gives_it_back(
    node_module, loader_module, model_management, monkeypatch
):
    """One execute -> one ``load_models_gpu`` with all the pack patchers and
    the release afterwards."""
    pack = loader_module.KEEPModelPack(
        _StubKeepNet(), _StubFaceHelper(), None, None, "KEEP"
    )
    _patch_the_node(monkeypatch, node_module, _StubProcessor)

    node_module.KEEP_FaceUpscaleImageNode.execute(
        image=torch.zeros((1, 8, 8, 3)),
        keep_model=pack,
        final_upscale_factor=1.0,
        has_aligned_face=False,
        only_center_face=True,
        draw_bounding_box=False,
    )

    names = [call[0] for call in model_management.calls]
    assert names.count("load_models_gpu") == 1, model_management.calls
    assert names.count("unload_all_models") == 1, model_management.calls
    assert names.count("soft_empty_cache") == 1, model_management.calls
    load_call = [c for c in model_management.calls if c[0] == "load_models_gpu"][0]
    assert list(load_call[1]) == list(pack.patchers)


@pytest.mark.mock_integration
def test_the_node_gives_the_vram_back_when_the_processing_fails(
    node_module, loader_module, model_management, monkeypatch
):
    """The release happens in the ``finally`` path - also on a failure."""
    pack = loader_module.KEEPModelPack(
        _StubKeepNet(), _StubFaceHelper(), None, None, "KEEP"
    )
    _patch_the_node(monkeypatch, node_module, _FailingProcessor)

    with pytest.raises(RuntimeError):
        node_module.KEEP_FaceUpscaleImageNode.execute(
            image=torch.zeros((1, 8, 8, 3)),
            keep_model=pack,
            final_upscale_factor=1.0,
            has_aligned_face=False,
            only_center_face=True,
            draw_bounding_box=False,
        )

    names = [call[0] for call in model_management.calls]
    assert names.count("load_models_gpu") == 1, model_management.calls
    assert names.count("unload_all_models") == 1, model_management.calls
    assert names.count("soft_empty_cache") == 1, model_management.calls


# ---------------------------------------------------------------------------
# ensure_flow
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_ensure_flow_is_idempotent(pack):
    """``ensure_flow`` resolves once per pack - the second call is a no-op."""
    assert pack.ensure_flow() is True, "the flownet comes with the KEEP net"
    # the flownet is gone -> the cached result is still returned
    pack.flownet = None
    assert pack.ensure_flow() is True


@pytest.mark.unit
def test_ensure_flow_reports_a_missing_flownet(loader_module):
    """A pack without a flownet reports that nothing is available (yet).

    P5.5 resolves the GMFlow checkpoint behind this hook.
    """
    net = _StubKeepNet()
    net.flownet = None
    pack = loader_module.KEEPModelPack(net, _StubFaceHelper(), None, None, "KEEP")
    assert pack.ensure_flow() is False
