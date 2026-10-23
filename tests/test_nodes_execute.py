"""P3.2 / P4.1 - the mock-integration execution of the node classes.

The loader node is a V3 node (P4.1) - its ``execute`` is a classmethod that
returns an ``io.NodeOutput``.  The tests drive it with a **stubbed loader**
(the model build is stubbed out - no weights are needed) and assert the
contract of the node: the pack comes back, the widgets reach the loader
unchanged and the second call is served from the loader's cache.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def node_module():
    return importlib.import_module("ComfyUI_KEEP.nodes.keep_face_upscale_image")


@pytest.fixture(scope="module")
def loader_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_model_loader")


@pytest.fixture(scope="module")
def loader_node_module():
    return importlib.import_module("ComfyUI_KEEP.nodes.keep_model_loader")


@pytest.fixture(scope="module")
def sequence_node_module():
    return importlib.import_module("ComfyUI_KEEP.nodes.keep_process_image_sequence")


@pytest.fixture
def stub_pack(loader_module, monkeypatch):
    """A real ``KEEPModelPack`` (the node checks the type) with stubs."""
    pack = loader_module.KEEPModelPack(
        keep_net=None,
        face_helper=None,
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


@pytest.fixture
def stubbed_loader(loader_module, monkeypatch, tmp_path):
    """A real ``KEEPModelLoader`` whose model build is stubbed out.

    The stub replaces the four heavy parts of the build (the architecture, the
    checkpoint download, the checkpoint load and the face helper) - the cache
    logic of the loader itself stays **real**.
    """
    class _StubArch:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loaded = None

        def to(self, device):
            return self

        def load_state_dict(self, state_dict, strict=True):
            self.loaded = state_dict
            return self

        def eval(self):
            return self

    class _StubHelper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(loader_module, "ARCH_REGISTRY", {"KEEP": _StubArch})
    monkeypatch.setattr(
        loader_module,
        "load_file_from_url_comfy",
        lambda **kwargs: str(tmp_path / "checkpoint.pth"),
    )
    monkeypatch.setattr(
        loader_module.torch,
        "load",
        lambda *args, **kwargs: {"params_ema": {"weight": torch.zeros(1)}},
    )
    monkeypatch.setattr(loader_module, "_face_restore_helper", lambda: _StubHelper)
    monkeypatch.setattr(
        loader_module.keep_paths, "resolve_model_root", lambda *args, **kwargs: str(tmp_path)
    )
    return loader_module.KEEPModelLoader()


# ---------------------------------------------------------------------------
# P4.1 - the model loader node
# ---------------------------------------------------------------------------
@pytest.mark.mock_integration
def test_the_loader_node_returns_the_pack(loader_node_module, stubbed_loader, monkeypatch):
    """The node returns the pack of the loader as a ``NodeOutput``."""
    from comfy_api.latest import io

    monkeypatch.setattr(loader_node_module, "GLOBAL_KEEP_MODEL_LOADER", stubbed_loader)

    output = loader_node_module.KEEP_ModelLoaderNode.execute(
        model="KEEP",
        detection_model="retinaface_resnet50",
    )

    assert isinstance(output, io.NodeOutput)
    assert len(output.result) == 1
    pack = output.result[0]
    assert pack.model_type_str == "KEEP"
    assert pack.keep_net is not None


@pytest.mark.mock_integration
def test_the_loader_node_serves_the_second_call_from_the_cache(
    loader_node_module, stubbed_loader, monkeypatch
):
    """The second call with the same widgets reuses the cached models."""
    monkeypatch.setattr(loader_node_module, "GLOBAL_KEEP_MODEL_LOADER", stubbed_loader)

    first = loader_node_module.KEEP_ModelLoaderNode.execute(
        model="KEEP", detection_model="retinaface_resnet50"
    ).result[0]
    second = loader_node_module.KEEP_ModelLoaderNode.execute(
        model="KEEP", detection_model="retinaface_resnet50"
    ).result[0]

    assert first is not second, "every call returns its own pack"
    assert first.keep_net is second.keep_net, "the models are cached"
    assert first.face_helper is second.face_helper, "the models are cached"
    assert len(stubbed_loader.loaded_models) == 1, stubbed_loader.loaded_models


@pytest.mark.mock_integration
def test_the_loader_node_passes_the_widgets_to_the_loader(
    loader_node_module, monkeypatch
):
    """The widget values reach the loader unchanged."""
    received: list[dict] = []

    class _RecordingLoader:
        def load_keep_model_pack(self, **kwargs):
            received.append(kwargs)
            return "pack"

    monkeypatch.setattr(loader_node_module, "GLOBAL_KEEP_MODEL_LOADER", _RecordingLoader())

    output = loader_node_module.KEEP_ModelLoaderNode.execute(
        model="KEEP",
        detection_model="retinaface_mobile0.25",
        bg_upscale_model="bg",
        face_upscale_model="face",
    )

    assert output.result == ("pack",)
    assert received == [
        {
            "model_type_str": "KEEP",
            "detection_model_str": "retinaface_mobile0.25",
            "bg_upscale_model": "bg",
            "face_upscale_model": "face",
        }
    ]


# ---------------------------------------------------------------------------
# P4.2 - the single image node
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_validate_inputs_refuses_a_foreign_pack(node_module):
    """A pack of the wrong type is refused with an actionable message."""
    message = node_module.KEEP_FaceUpscaleImageNode.validate_inputs(
        image=torch.rand((1, 4, 4, 3), dtype=torch.float32),
        keep_model=object(),
        final_upscale_factor=1.0,
        has_aligned_face=True,
        only_center_face=True,
        draw_bounding_box=False,
    )
    assert isinstance(message, str), message
    assert "KEEPModelPack" in message, message
    assert "Load KEEP Models" in message, message


@pytest.mark.unit
def test_validate_inputs_accepts_a_real_pack(node_module, stub_pack):
    """A real pack passes the validation."""
    assert (
        node_module.KEEP_FaceUpscaleImageNode.validate_inputs(
            image=torch.rand((1, 4, 4, 3), dtype=torch.float32),
            keep_model=stub_pack,
            final_upscale_factor=1.0,
            has_aligned_face=True,
            only_center_face=True,
            draw_bounding_box=False,
        )
        is True
    )


@pytest.mark.mock_integration
def test_upscale_face_image_returns_the_processed_image(node_module, stub_pack, monkeypatch):
    """The node returns the processed image and uses the pack once."""
    from comfy_api.latest import io

    processed = np.full((4, 6, 3), 128, dtype=np.uint8)
    processor_calls: list[dict] = []

    class _StubProcessor:
        def __init__(self, model_pack):
            processor_calls.append({"pack": model_pack})

        def process_image(self, **kwargs):
            processor_calls.append({"kwargs": kwargs})
            return processed

    monkeypatch.setattr(node_module, "KEEPFaceProcessor", _StubProcessor)

    image = torch.rand((1, 4, 6, 3), dtype=torch.float32)
    result = node_module.KEEP_FaceUpscaleImageNode.execute(
        image,
        stub_pack,
        final_upscale_factor=1.0,
        has_aligned_face=True,
        only_center_face=True,
        draw_bounding_box=False,
    )

    assert isinstance(result, io.NodeOutput)
    assert len(result.result) == 1
    output = result.result[0]
    assert tuple(output.shape) == (1, 4, 6, 3), output.shape
    # the stub's image (128/255) came back
    assert torch.allclose(output, torch.full((1, 4, 6, 3), 128 / 255), atol=1e-3), output

    # the processor was built once and called once
    assert len(processor_calls) == 2, processor_calls
    assert processor_calls[0]["pack"] is stub_pack
    assert processor_calls[1]["kwargs"]["final_upscale_factor"] == 1.0
    # the pack was loaded and offloaded exactly once each
    assert stub_pack.calls == ["load_device", "offload"], stub_pack.calls


@pytest.mark.mock_integration
def test_upscale_face_image_scales_the_output(node_module, stub_pack, monkeypatch):
    """The output of the node has the scaled shape ``(1, H*f, W*f, 3)``."""
    factor = 2.0

    class _StubProcessor:
        def __init__(self, model_pack):
            pass

        def process_image(self, cv2_image_orig, final_upscale_factor, **kwargs):
            height, width = cv2_image_orig.shape[:2]
            return np.full(
                (int(height * final_upscale_factor), int(width * final_upscale_factor), 3),
                64,
                dtype=np.uint8,
            )

    monkeypatch.setattr(node_module, "KEEPFaceProcessor", _StubProcessor)

    result = node_module.KEEP_FaceUpscaleImageNode.execute(
        torch.rand((1, 4, 6, 3), dtype=torch.float32),
        stub_pack,
        final_upscale_factor=factor,
        has_aligned_face=True,
        only_center_face=True,
        draw_bounding_box=False,
    )

    output = result.result[0]
    assert tuple(output.shape) == (1, 8, 12, 3), output.shape


@pytest.mark.mock_integration
def test_upscale_face_image_raises_on_a_failure(node_module, stub_pack, monkeypatch):
    """A failing processing raises (B8: no ``(None,)`` behind an IMAGE socket)."""

    class _FailingProcessor:
        def __init__(self, model_pack):
            pass

        def process_image(self, **kwargs):
            raise RuntimeError("the restoration failed")

    monkeypatch.setattr(node_module, "KEEPFaceProcessor", _FailingProcessor)

    with pytest.raises(RuntimeError, match="the restoration failed"):
        node_module.KEEP_FaceUpscaleImageNode.execute(
            torch.rand((1, 4, 6, 3), dtype=torch.float32),
            stub_pack,
            final_upscale_factor=1.0,
            has_aligned_face=True,
            only_center_face=True,
            draw_bounding_box=False,
        )

    # the pack is offloaded even when the processing fails
    assert stub_pack.calls == ["load_device", "offload"], stub_pack.calls


@pytest.mark.mock_integration
def test_upscale_face_image_reports_the_progress(node_module, stub_pack, monkeypatch):
    """The node reports its progress through the ComfyUI execution API."""
    progress_module = importlib.import_module("ComfyUI_KEEP.src.keep_progress")

    class _StubProcessor:
        def __init__(self, model_pack):
            pass

        def process_image(self, **kwargs):
            return np.zeros((4, 6, 3), dtype=np.uint8)

    monkeypatch.setattr(node_module, "KEEPFaceProcessor", _StubProcessor)

    # the cached API instance of the engine (built on the first report)
    api = progress_module.get_api_sync()
    api.execution.calls.clear()

    node_module.KEEP_FaceUpscaleImageNode.execute(
        torch.rand((1, 4, 6, 3), dtype=torch.float32),
        stub_pack,
        final_upscale_factor=1.0,
        has_aligned_face=True,
        only_center_face=True,
        draw_bounding_box=False,
    )

    assert api.execution.calls == [(0.0, 1.0), (1.0, 1.0)], api.execution.calls


# ---------------------------------------------------------------------------
# P4.3 - the image sequence node
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_sequence_node_refuses_a_foreign_pack(sequence_node_module):
    """A pack of the wrong type is refused with an actionable message."""
    message = sequence_node_module.KEEP_ProcessImageSequenceNode.validate_inputs(
        images=torch.rand((2, 4, 4, 3), dtype=torch.float32),
        keep_model=object(),
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_bounding_box=False,
        max_clip_length=20,
    )
    assert isinstance(message, str), message
    assert "KEEPModelPack" in message, message


@pytest.mark.mock_integration
def test_the_sequence_node_calls_the_processor_once(
    sequence_node_module, stub_pack, monkeypatch
):
    """30 frames with ``max_clip_length=20`` - the processor is called once."""
    from comfy_api.latest import io

    calls: list[dict] = []

    class _StubProcessor:
        def __init__(self, model_pack):
            calls.append({"pack": model_pack})

        def process_image_sequence(self, **kwargs):
            calls.append({"kwargs": kwargs})
            return kwargs["image_sequence_tensor"]

    monkeypatch.setattr(sequence_node_module, "KEEPFaceProcessor", _StubProcessor)

    frames = torch.rand((30, 4, 6, 3), dtype=torch.float32)
    output = sequence_node_module.KEEP_ProcessImageSequenceNode.execute(
        frames,
        stub_pack,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_bounding_box=False,
        max_clip_length=20,
    )

    assert isinstance(output, io.NodeOutput)
    assert len(output.result) == 1
    assert output.result[0].shape == frames.shape
    # the processor was built once and called once
    assert len(calls) == 2, calls
    assert calls[0]["pack"] is stub_pack
    assert calls[1]["kwargs"]["max_clip_length"] == 20
    assert calls[1]["kwargs"]["image_sequence_tensor"] is frames


@pytest.mark.mock_integration
def test_the_sequence_node_calls_ensure_flow_on_a_multi_frame_run(
    sequence_node_module, stub_pack, monkeypatch
):
    """The flow hook is called exactly once - between load and offload."""

    class _StubProcessor:
        def __init__(self, model_pack):
            pass

        def process_image_sequence(self, **kwargs):
            return kwargs["image_sequence_tensor"]

    monkeypatch.setattr(sequence_node_module, "KEEPFaceProcessor", _StubProcessor)

    sequence_node_module.KEEP_ProcessImageSequenceNode.execute(
        torch.rand((30, 4, 6, 3), dtype=torch.float32),
        stub_pack,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_bounding_box=False,
        max_clip_length=20,
    )

    assert stub_pack.calls == ["load_device", "ensure_flow", "offload"], stub_pack.calls


@pytest.mark.mock_integration
def test_the_sequence_node_skips_ensure_flow_for_a_single_frame(
    sequence_node_module, stub_pack, monkeypatch
):
    """A single frame needs no optical flow - the hook is not called."""

    class _StubProcessor:
        def __init__(self, model_pack):
            pass

        def process_image_sequence(self, **kwargs):
            return kwargs["image_sequence_tensor"]

    monkeypatch.setattr(sequence_node_module, "KEEPFaceProcessor", _StubProcessor)

    sequence_node_module.KEEP_ProcessImageSequenceNode.execute(
        torch.rand((1, 4, 6, 3), dtype=torch.float32),
        stub_pack,
        final_upscale_factor=1.0,
        has_aligned_frames=False,
        only_center_face=True,
        draw_bounding_box=False,
        max_clip_length=20,
    )

    assert stub_pack.calls == ["load_device", "offload"], stub_pack.calls


@pytest.mark.mock_integration
def test_the_sequence_node_raises_on_a_failure(
    sequence_node_module, stub_pack, monkeypatch
):
    """A failing processing raises and the pack is offloaded anyway."""

    class _FailingProcessor:
        def __init__(self, model_pack):
            pass

        def process_image_sequence(self, **kwargs):
            raise RuntimeError("the sequence failed")

    monkeypatch.setattr(sequence_node_module, "KEEPFaceProcessor", _FailingProcessor)

    with pytest.raises(RuntimeError, match="the sequence failed"):
        sequence_node_module.KEEP_ProcessImageSequenceNode.execute(
            torch.rand((2, 4, 6, 3), dtype=torch.float32),
            stub_pack,
            final_upscale_factor=1.0,
            has_aligned_frames=False,
            only_center_face=True,
            draw_bounding_box=False,
            max_clip_length=20,
        )

    assert stub_pack.calls == ["load_device", "ensure_flow", "offload"], stub_pack.calls
