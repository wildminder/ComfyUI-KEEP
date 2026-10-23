"""P3.2 - the mock-integration execution of the node classes.

The nodes are still V1 classes in this phase (P4 migrates them to the V3
schema), so their ``execute`` path is the plain method.  The tests drive it
with a **stub pack** and a stubbed processor and assert the contract of the
node: the image that goes in comes back out, and the pack is loaded and
offloaded exactly once.
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
    pack.calls = calls
    return pack


@pytest.mark.mock_integration
def test_upscale_face_image_returns_the_processed_image(node_module, stub_pack, monkeypatch):
    """The node returns the processed image and uses the pack once."""
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
    result = node_module.KEEP_FaceUpscaleImageNode().upscale_face_image(
        image,
        stub_pack,
        final_upscale_factor=1.0,
        has_aligned_face=True,
        only_center_face=True,
        draw_bounding_box=False,
    )

    assert len(result) == 1
    output = result[0]
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
def test_upscale_face_image_rejects_a_foreign_pack(node_module):
    """A pack of the wrong type is refused (and the error is logged)."""
    result = node_module.KEEP_FaceUpscaleImageNode().upscale_face_image(
        torch.rand((1, 4, 4, 3), dtype=torch.float32),
        object(),
        final_upscale_factor=1.0,
        has_aligned_face=True,
        only_center_face=True,
        draw_bounding_box=False,
    )
    assert result == (None,)
