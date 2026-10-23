"""P3.2 - the face processor of ``src/keep_processor.py``.

The tests drive the processor with **stubs** (a stub face helper and a stub
KEEP net) and assert the things the processor is responsible for:

* the chunk math of the sequence path (``max_clip_length``),
* the paste-back (the output frame count and shape),
* the ``tiled_scale`` wrapper of the optional ComfyUI upscalers.
"""

from __future__ import annotations

import importlib
import types

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def processor_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_processor")


class _StubKeepNet:
    """A KEEP net that records the clips it is called with."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, frames, need_upscale=False):
        self.calls.append((tuple(frames.shape), need_upscale))
        return torch.zeros_like(frames)


class _StubFaceHelper:
    """The facelib helper, reduced to the calls the processor makes."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.upscale_factor = 1.0
        self.is_gray = False
        self.cropped_faces: list = []
        self.restored_faces: list = []
        self.all_landmarks_5: list = []
        self.calls: list[str] = []

    def clean_all(self):
        self.calls.append("clean_all")
        self.cropped_faces = []
        self.restored_faces = []

    def read_image(self, image):
        self.calls.append("read_image")

    def align_warp_face(self):
        self.calls.append("align_warp_face")

    def add_restored_face(self, face, **kwargs):
        self.calls.append("add_restored_face")
        self.restored_faces = [face]

    def get_inverse_affine(self, *args, **kwargs):
        self.calls.append("get_inverse_affine")

    def paste_faces_to_input_image(self, *args, **kwargs):
        self.calls.append("paste_faces_to_input_image")
        return np.zeros((512, 512, 3), dtype=np.uint8)


def _make_pack(keep_net, face_helper):
    """A model pack the processor can consume."""
    return types.SimpleNamespace(
        keep_net=keep_net,
        face_helper=face_helper,
        bg_upscale_model=None,
        face_upscale_model=None,
        device=torch.device("cpu"),
        model_type_str="KEEP",
    )


@pytest.mark.unit
def test_the_sequence_is_split_into_clips_of_max_clip_length(processor_module):
    """The chunk math of the sequence path (``max_clip_length``)."""
    keep_net = _StubKeepNet()
    face_helper = _StubFaceHelper()
    processor = processor_module.KEEPFaceProcessor(_make_pack(keep_net, face_helper))

    frames = torch.rand((4, 16, 16, 3), dtype=torch.float32)
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=True,
        only_center_face=True,
        draw_box=False,
        max_clip_length=2,
    )

    # 4 frames, clips of 2 -> two calls of two frames each
    assert len(keep_net.calls) == 2, keep_net.calls
    for shape, need_upscale in keep_net.calls:
        assert shape == (1, 2, 3, 512, 512), shape
        assert need_upscale is False
    # the paste-back produced one frame per input frame
    assert tuple(output.shape) == (4, 512, 512, 3), output.shape


@pytest.mark.unit
def test_an_empty_sequence_is_returned_unchanged(processor_module):
    """No frames -> no work, the input is returned."""
    keep_net = _StubKeepNet()
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(keep_net, _StubFaceHelper())
    )
    frames = torch.zeros((0, 16, 16, 3), dtype=torch.float32)
    output = processor.process_image_sequence(
        image_sequence_tensor=frames,
        final_upscale_factor=1.0,
        has_aligned_frames=True,
        only_center_face=True,
        draw_box=False,
        max_clip_length=20,
    )
    assert output is frames
    assert keep_net.calls == []


@pytest.mark.unit
def test_the_upscaler_goes_through_tiled_scale(processor_module):
    """The optional ComfyUI upscaler is called through ``tiled_scale``."""
    calls: list[dict] = []

    class _StubUpscaleModel:
        scale = 2.0

        def __init__(self):
            self.model = self._model

        def _model(self, samples):
            calls.append({"shape": tuple(samples.shape)})
            return samples * 2.0

    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _StubFaceHelper())
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = processor._run_upscaler(_StubUpscaleModel(), image)

    assert len(calls) == 1, calls
    assert calls[0]["shape"] == (1, 3, 8, 8), calls
    assert result.shape == (8, 8, 3), result.shape


@pytest.mark.unit
def test_without_an_upscaler_the_image_is_untouched(processor_module):
    """No upscaler -> the image is returned unchanged."""
    processor = processor_module.KEEPFaceProcessor(
        _make_pack(_StubKeepNet(), _StubFaceHelper())
    )
    image = np.full((4, 4, 3), 7, dtype=np.uint8)
    assert processor._run_upscaler(None, image) is image
