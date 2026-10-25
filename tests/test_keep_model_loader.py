"""P4.5 - the model residency of the pack (ModelPatcher) and ``ensure_flow``.

The pack wraps its three model components (the KEEP net with its flownet, the
face detector, the face parser) in ``ModelPatcher`` instances and lets ComfyUI
arbitrate the VRAM.  The tests run against the **mocked**
``comfy.model_management`` of the conftest: every residency call is recorded,
so the tests can assert what the pack asks ComfyUI to do - and that it asks
exactly once per execute.
"""

from __future__ import annotations

import hashlib
import importlib
import os

import pytest
import torch


@pytest.fixture(scope="module")
def loader_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_model_loader")


@pytest.fixture(scope="module")
def node_module():
    return importlib.import_module("ComfyUI_KEEP.nodes.keep_face_upscale_image")


@pytest.fixture(scope="module")
def configs_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_model_configs")


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
def test_ensure_flow_is_idempotent(loader_module, stub_models):
    """``ensure_flow`` resolves once per pack - the second call is a no-op.

    A checkpoint that carries the flownet weights (the shipped one, P0.4
    section 2) needs no resolution at all.
    """
    net, helper = stub_models
    pack = loader_module.KEEPModelPack(
        net, helper, None, None, "KEEP", flownet_from_checkpoint=True
    )
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


# ---------------------------------------------------------------------------
# P5.3 - the fail-fast weight validation
# ---------------------------------------------------------------------------
PAYLOAD = b"not a real checkpoint"
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def synthetic_manifest(loader_module, configs_module, monkeypatch):
    """The manifest with the hash of the test payload.

    The real hashes cannot be reproduced by a test, so the manifest the
    loader and the downloader see is replaced by entries that carry the hash
    of the payload the transport double writes.  Everything else - the
    resolution order, the fail-fast rule, the error messages - is the
    production code.
    """
    entries = tuple(
        configs_module.WeightEntry(
            name=entry.name,
            url=entry.url,
            sha256=PAYLOAD_SHA256,
            dest_folder=entry.dest_folder,
            subdir=entry.subdir,
            required=entry.required,
            available=entry.available,
            unverified=entry.unverified,
            sha256_prefix=entry.sha256_prefix,
        )
        for entry in configs_module.WEIGHT_MANIFEST
    )
    by_name = {entry.name: entry for entry in entries}
    monkeypatch.setattr(loader_module, "WEIGHT_MANIFEST_BY_NAME", by_name)
    monkeypatch.setattr(
        loader_module,
        "weight_entry_for",
        lambda model_type: by_name[
            "KEEP_Asian-4765ebe0.pth" if model_type == "Asian" else "KEEP-b76feb75.pth"
        ],
    )
    monkeypatch.setattr(
        loader_module.keep_download, "WEIGHT_MANIFEST", entries
    )
    return by_name


@pytest.fixture
def transport(monkeypatch):
    """The transport double: it records the URLs and writes the payload."""
    download_module = importlib.import_module("ComfyUI_KEEP.src.keep_download")
    calls: list[str] = []

    def _download(url, dst, hash_prefix=None, progress=True):
        calls.append(url)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as handle:
            handle.write(PAYLOAD)

    monkeypatch.setattr(download_module, "download_url_to_file", _download)
    return calls


@pytest.fixture
def loader(loader_module, monkeypatch, synthetic_manifest, transport):
    """A loader whose model build is stubbed out (the weights are real)."""

    class _StubArch:
        """A stand-in for the vendored arch that behaves like ``nn.Module``.

        The real arch **owns** the flownet as a submodule
        (``keep_arch.py:882``), so the state dict of the net always carries
        the ``flownet.*`` keys - and a checkpoint without them fails the
        **strict** load.  The stub keeps that behaviour (a no-op recorder
        would hide exactly that defect).
        """

        #: The state dict of the stub net (the flownet is part of it).
        STATE = {
            "weight": torch.zeros(1),
            "flownet.model.conv1.weight": torch.ones(1),
        }

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            # the real arch owns the flownet as a submodule (keep_arch.py:882)
            self.flownet = _StubModule("flownet")
            self.state = dict(self.STATE)

        def to(self, device):
            return self

        def state_dict(self):
            return dict(self.state)

        def load_state_dict(self, state_dict, strict=True):
            missing = [key for key in self.state if key not in state_dict]
            unexpected = [key for key in state_dict if key not in self.state]
            if strict and (missing or unexpected):
                raise RuntimeError(
                    "Error(s) in loading state_dict for KEEP: missing "
                    f"{missing}, unexpected {unexpected}"
                )
            self.loaded = state_dict
            return self

        def eval(self):
            return self

    class _StubHelper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(loader_module, "ARCH_REGISTRY", {"KEEP": _StubArch})
    monkeypatch.setattr(
        loader_module.torch,
        "load",
        lambda *args, **kwargs: {"params_ema": {"weight": torch.zeros(1)}},
    )
    monkeypatch.setattr(loader_module, "_face_restore_helper", lambda: _StubHelper)
    return loader_module.KEEPModelLoader()


@pytest.mark.unit
def test_a_missing_required_weight_triggers_a_download(loader, transport):
    """The weights are resolved (and downloaded) before the net is built."""
    pack = loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    assert pack.keep_net is not None
    urls = [url.rsplit("/", 1)[-1] for url in transport]
    assert "KEEP-b76feb75.pth" in urls, urls
    # D5: only the **selected** detector and the face parsing net are
    # fetched - the other detectors of the manifest are never downloaded.
    assert "detection_Resnet50_Final.pth" in urls, urls
    assert "parsing_parsenet.pth" in urls, urls


@pytest.mark.unit
def test_a_failed_download_raises_the_actionable_message(
    loader_module, loader, monkeypatch, transport
):
    """A missing required weight names the file, the path and the URL."""
    download_module = importlib.import_module("ComfyUI_KEEP.src.keep_download")

    def _fail(url, dst, hash_prefix=None, progress=True):
        raise RuntimeError("the server is not reachable")

    monkeypatch.setattr(download_module, "download_url_to_file", _fail)

    with pytest.raises(FileNotFoundError) as excinfo:
        loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    message = str(excinfo.value)
    assert "KEEP-b76feb75.pth" in message
    assert "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP-b76feb75.pth" in message
    assert "keep_models" in message and "KEEP" in message


@pytest.mark.unit
def test_an_unresolvable_optional_gmflow_weight_does_not_raise(
    loader_module, loader, monkeypatch
):
    """The GMFlow checkpoint is optional - its absence never breaks the pack.

    The flow weights are only *needed* when the checkpoint does not carry
    them; the resolution is deferred to ``ensure_flow()`` (P5.5).
    """
    monkeypatch.setattr(
        loader_module.keep_download,
        "ensure_optional",
        lambda entry, progress=True: None,
    )

    pack = loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    assert pack is not None
    assert pack.keep_net.kwargs["flownet_path"] is None


@pytest.mark.unit
def test_the_unavailable_asian_entry_does_not_break_the_default_pack(loader, transport):
    """The Asian checkpoint is not required - the default pack still builds."""
    pack = loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    assert pack is not None
    urls = [url.rsplit("/", 1)[-1] for url in transport]
    assert "KEEP_Asian-4765ebe0.pth" not in urls, urls


@pytest.mark.unit
def test_selecting_the_asian_model_raises_an_actionable_error(loader):
    """The Asian checkpoint is not published upstream (HTTP 404)."""
    with pytest.raises(FileNotFoundError) as excinfo:
        loader.load_keep_model_pack("Asian", "retinaface_resnet50")

    message = str(excinfo.value)
    assert "KEEP_Asian-4765ebe0.pth" in message
    assert "not published upstream" in message
    assert "KEEP_Asian-4765ebe0.pth" in message
    assert "keep_models" in message


# ---------------------------------------------------------------------------
# P5.5 - the flownet wiring
# ---------------------------------------------------------------------------
def _downloaded(transport) -> list[str]:
    return [url.rsplit("/", 1)[-1] for url in transport]


@pytest.mark.unit
def test_the_arch_gets_a_resolved_flownet_path_without_flownet_weights(loader, transport):
    """A checkpoint **without** flownet weights -> the GMFlow file is resolved.

    The resolved path is put into the architecture config **before** the net
    is built, so the vendored arch loads the trained weights instead of
    random ones.
    """
    pack = loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    flownet_path = pack.keep_net.kwargs["flownet_path"]
    assert flownet_path is not None, "the GMFlow checkpoint was not resolved"
    assert flownet_path.endswith("gmflow_sintel-0c07dcb3.pth"), flownet_path
    assert "gmflow_sintel-0c07dcb3.pth" in _downloaded(transport)


@pytest.mark.unit
def test_a_checkpoint_without_flownet_weights_loads_strictly(loader):
    """The fallback branch must not break the strict load.

    The real arch **always** owns the flownet (``keep_arch.py:882``), so the
    state dict of the net always carries the ``flownet.*`` keys.  A
    checkpoint without them used to make the strict load raise
    ``RuntimeError: Missing key(s) in state_dict: flownet.*`` - and the
    GMFlow weights the arch had just loaded were thrown away.  The flownet
    tensors of the built net fill the gap, so the load stays strict **and**
    the trained flow weights survive.
    """
    pack = loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    # the strict load happened (the stub arch raises on a missing key)
    assert pack.keep_net.loaded is not None
    # the flownet of the built net (the trained GMFlow weights) is in the
    # loaded state dict - it was not discarded
    assert torch.equal(
        pack.keep_net.loaded["flownet.model.conv1.weight"],
        pack.keep_net.state_dict()["flownet.model.conv1.weight"],
    )


@pytest.mark.unit
def test_the_strict_load_still_rejects_an_unexpected_key(loader_module, loader, monkeypatch):
    """The flownet merge does not weaken the strict load."""
    monkeypatch.setattr(
        loader_module.torch,
        "load",
        lambda *args, **kwargs: {
            "params_ema": {"weight": torch.zeros(1), "unexpected.key": torch.zeros(1)}
        },
    )

    with pytest.raises(RuntimeError):
        loader.load_keep_model_pack("KEEP", "retinaface_resnet50")


@pytest.mark.unit
def test_a_checkpoint_with_flownet_weights_downloads_no_gmflow(
    loader_module, loader, transport, monkeypatch
):
    """The shipped checkpoint carries the flownet -> nothing is downloaded."""
    monkeypatch.setattr(
        loader_module.torch,
        "load",
        lambda *args, **kwargs: {
            "params_ema": {
                "flownet.model.conv1.weight": torch.zeros(1),
                "weight": torch.zeros(1),
            }
        },
    )

    pack = loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    assert pack.keep_net.kwargs["flownet_path"] is None
    assert "gmflow_sintel-0c07dcb3.pth" not in _downloaded(transport)
    # the flownet is restored from the checkpoint (the strict load)
    assert "flownet.model.conv1.weight" in pack.keep_net.loaded
    assert pack.flownet_from_checkpoint is True
    # the hook is a no-op - the weights are already there
    assert pack.ensure_flow() is True
    assert "gmflow_sintel-0c07dcb3.pth" not in _downloaded(transport)


class _RecordingInner:
    """The inner module of the flownet (it records the strict load)."""

    def __init__(self) -> None:
        self.loaded: tuple | None = None

    def load_state_dict(self, weights, strict=True):
        self.loaded = (weights, strict)
        return self


class _RecordingFlownet:
    """A flownet that records the weights it receives."""

    def __init__(self) -> None:
        self.model = _RecordingInner()


@pytest.mark.unit
def test_ensure_flow_loads_the_gmflow_weights_strictly(
    loader_module, stub_models, monkeypatch, tmp_path
):
    """A net without flow weights gets them from the GMFlow file."""
    net, helper = stub_models
    net.flownet = _RecordingFlownet()
    pack = loader_module.KEEPModelPack(
        net, helper, None, None, "KEEP",
        flownet_from_checkpoint=False, flownet_path=None,
    )

    gmflow_path = tmp_path / "gmflow_sintel-0c07dcb3.pth"
    gmflow_path.write_bytes(b"weights")
    monkeypatch.setattr(
        loader_module.keep_download,
        "ensure_optional",
        lambda entry, progress=True: str(gmflow_path),
    )
    monkeypatch.setattr(
        loader_module.torch,
        "load",
        lambda *args, **kwargs: {"model": {"w": torch.zeros(1)}},
    )

    assert pack.ensure_flow() is True
    weights, strict = net.flownet.model.loaded
    assert strict is True, "the flow weights must be loaded with strict=True"
    assert "w" in weights
    # the resolution is cached - a second call does nothing
    assert pack.ensure_flow() is True


@pytest.mark.mock_integration
def test_the_image_node_does_not_call_ensure_flow(
    loader_module, node_module, monkeypatch
):
    """The image path leaves the flow resolution deferred."""
    net = _StubKeepNet()
    pack = loader_module.KEEPModelPack(
        net, _StubFaceHelper(), None, None, "KEEP"
    )
    calls: list[str] = []
    monkeypatch.setattr(pack, "ensure_flow", lambda: calls.append("ensure_flow"))
    _patch_the_node(monkeypatch, node_module, _StubProcessor)

    node_module.KEEP_FaceUpscaleImageNode.execute(
        image=torch.zeros((1, 8, 8, 3)),
        keep_model=pack,
        final_upscale_factor=1.0,
        has_aligned_face=False,
        only_center_face=True,
        draw_bounding_box=False,
    )

    assert calls == [], "the image path must not resolve the flow weights"
