"""D5 - only the **selected** face detector is downloaded (and co-located).

The pack used to download **every** face detector of the manifest - the
weights of the three detectors the user did **not** select were fetched as
well (``yolov5l-face.pth`` alone is 187 MB).  Since D5 the loader resolves the
**selected** detector and the face parsing net: the helper is built with
``use_parse=True``, so the parsing net is the only other weight it loads.

The two weights must live in **one** root - the vendored helper builds the
weight paths from a single root (``detection/__init__.py:53``,
``parsing/__init__.py:30``).  A weight that lives in another root is
**copied** there (``shutil.copy2``), never downloaded a second time.

The tests below are the **proof of the defect**: with the production change
stashed they fail - every detector is downloaded, an unsupported selection is
accepted and the weights are not co-located.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os

import pytest
import torch

#: The payload the transport double writes (the real hashes cannot be
#: reproduced by a test - the manifest the loader sees carries the hash of
#: this payload instead).
PAYLOAD = b"not a real weight"
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()

#: The five weights of the facelib folder (the manifest of the pack).
FACELIB_FILES = (
    "detection_Resnet50_Final.pth",
    "detection_mobilenet0.25_Final.pth",
    "yolov5n-face.pth",
    "yolov5l-face.pth",
    "parsing_parsenet.pth",
)

#: The recorded widget contract of the model loader node (the order and the
#: default are compatibility guarantees of the pack).
EXPECTED_DETECTION_MODELS = (
    "retinaface_resnet50",
    "retinaface_mobile0.25",
    "YOLOv5l",
    "YOLOv5n",
)


@pytest.fixture(scope="module")
def loader_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_model_loader")


@pytest.fixture(scope="module")
def configs_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_model_configs")


@pytest.fixture(scope="module")
def download_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_download")


@pytest.fixture(scope="module")
def logger_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_logger")


@pytest.fixture(scope="module")
def vendored_model_names():
    """The ``MODEL_NAMES`` of the vendored helper - the mapping it uses."""
    detection = importlib.import_module(
        "ComfyUI_KEEP.modules.deps.facelib.detection"
    )
    return dict(detection.MODEL_NAMES)


@pytest.fixture
def synthetic_manifest(loader_module, configs_module, download_module, monkeypatch):
    """The manifest with the hash of the payload the transport writes.

    Only the **hash** is replaced: the names, the urls and the destinations
    are the production ones - the selection and the download set are the
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
        lambda model_type: by_name["KEEP-b76feb75.pth"],
    )
    monkeypatch.setattr(download_module, "WEIGHT_MANIFEST", entries)
    return by_name


@pytest.fixture
def transport(download_module, monkeypatch):
    """The transport double: it records the urls and writes the payload."""
    calls: list[str] = []

    def _download(url, dst, hash_prefix=None, progress=True):
        calls.append(url)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as handle:
            handle.write(PAYLOAD)

    monkeypatch.setattr(download_module, "download_url_to_file", _download)
    return calls


class _StubModule:
    """A minimal stand-in for a torch module (it only has to move)."""

    def __init__(self, name: str = "module") -> None:
        self.name = name
        self.moved_to: list = []

    def to(self, device):
        self.moved_to.append(device)
        return self


class _StubArch:
    """A stand-in for the vendored arch that behaves like ``nn.Module``.

    The real arch **owns** the flownet as a submodule (``keep_arch.py:882``),
    so the state dict of the net always carries the ``flownet.*`` keys - and
    a checkpoint without them fails the **strict** load.  The stub keeps that
    behaviour (a no-op recorder would hide exactly that defect).
    """

    STATE = {
        "weight": torch.zeros(1),
        "flownet.model.conv1.weight": torch.ones(1),
    }

    def __init__(self, **kwargs):
        self.kwargs = kwargs
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
    """A stand-in for the vendored face helper (it records its arguments)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_rootpath = kwargs.get("model_rootpath")
        self.det_model = kwargs.get("det_model")


class _PrintingHelper(_StubHelper):
    """A helper that prints - like the vendored one does while it is built."""

    def __init__(self, **kwargs):
        print("facelib: the detection model is ready")
        super().__init__(**kwargs)


@pytest.fixture
def loader(loader_module, monkeypatch, synthetic_manifest, transport):
    """A loader whose model build is stubbed out (the weights are real)."""
    monkeypatch.setattr(loader_module, "ARCH_REGISTRY", {"KEEP": _StubArch})
    monkeypatch.setattr(
        loader_module.torch,
        "load",
        lambda *args, **kwargs: {"params_ema": {"weight": torch.zeros(1)}},
    )
    monkeypatch.setattr(loader_module, "_face_restore_helper", lambda: _StubHelper)
    return loader_module.KEEPModelLoader()


def _downloaded(transport) -> list[str]:
    """The file names of the transport calls."""
    return [url.rsplit("/", 1)[-1] for url in transport]


def _facelib_downloads(transport) -> list[str]:
    """The facelib weights that were downloaded (the checkpoint excluded)."""
    return [name for name in _downloaded(transport) if name in FACELIB_FILES]


# ---------------------------------------------------------------------------
# the selection
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_only_the_selected_detector_and_the_parsing_net_are_downloaded(
    loader, transport
):
    """The download set is the selected detector **and** the parsing net.

    The helper is built with ``use_parse=True``, so the parsing net is
    needed - the three other detectors are not.
    """
    loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    assert _facelib_downloads(transport) == [
        "detection_Resnet50_Final.pth",
        "parsing_parsenet.pth",
    ], _downloaded(transport)


@pytest.mark.unit
def test_the_detectors_that_are_not_selected_are_never_downloaded(
    loader, transport
):
    """The weights of the unselected detectors are never fetched."""
    loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    downloaded = _downloaded(transport)
    for name in (
        "detection_mobilenet0.25_Final.pth",
        "yolov5n-face.pth",
        "yolov5l-face.pth",
    ):
        assert name not in downloaded, (name, downloaded)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("detector", "expected"),
    (
        ("retinaface_resnet50", "detection_Resnet50_Final.pth"),
        ("retinaface_mobile0.25", "detection_mobilenet0.25_Final.pth"),
        ("YOLOv5l", "yolov5l-face.pth"),
        ("YOLOv5n", "yolov5n-face.pth"),
    ),
)
def test_every_detector_downloads_its_own_weight(
    loader, transport, detector, expected
):
    """The selected detector decides the weight - and only that one."""
    loader.load_keep_model_pack("KEEP", detector)

    assert _facelib_downloads(transport) == [expected, "parsing_parsenet.pth"], (
        _downloaded(transport)
    )


@pytest.mark.unit
def test_an_unsupported_detector_is_rejected_before_anything_is_downloaded(
    loader, transport
):
    """An unsupported selection raises a ``ValueError`` that names the
    supported detectors - and nothing is downloaded."""
    with pytest.raises(ValueError) as excinfo:
        loader.load_keep_model_pack("KEEP", "retinaface_resnet999")

    message = str(excinfo.value)
    for supported in EXPECTED_DETECTION_MODELS:
        assert supported in message, (supported, message)
    assert transport == [], "the selection must be validated before the download"


@pytest.mark.unit
def test_the_widget_list_matches_the_mapping_of_the_vendored_helper(
    configs_module, vendored_model_names
):
    """The drift guard: the widget list and the vendored mapping agree.

    The mapping of a name to its weight file is the one of the vendored
    helper - a second table would drift.
    """
    assert tuple(configs_module.DETECTION_MODELS) == EXPECTED_DETECTION_MODELS
    assert set(configs_module.DETECTION_MODELS) == set(vendored_model_names)


@pytest.mark.unit
def test_the_selection_maps_to_the_weight_of_the_vendored_helper(
    loader_module, vendored_model_names
):
    """Every detector of the widget maps to the file of the helper."""
    for name in EXPECTED_DETECTION_MODELS:
        entry = loader_module.detection_weight_entry(name)
        assert entry.name == vendored_model_names[name]


# ---------------------------------------------------------------------------
# the co-location (one root for the helper)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_weights_are_co_located_in_the_root_of_the_detector(
    loader, transport, mock_folder_paths, tmp_path
):
    """A weight that lives in another root is **copied** - not downloaded.

    The vendored helper builds the weight paths from a **single** root, so
    the parsing net is copied into the root of the detector when it lives
    somewhere else.  The copy is a file copy: no second download.
    """
    extra_root = tmp_path / "extra-models" / "facedetection"
    extra_root.mkdir(parents=True)
    (extra_root / "detection_Resnet50_Final.pth").write_bytes(PAYLOAD)

    primary_root = tmp_path / "models" / "facedetection"
    primary_root.mkdir(parents=True)
    (primary_root / "parsing_parsenet.pth").write_bytes(PAYLOAD)

    mock_folder_paths.add_model_folder_path(
        "facedetection", str(extra_root), is_default=False
    )

    pack = loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    # the parsing net now lives next to the detector ...
    assert (extra_root / "parsing_parsenet.pth").is_file()
    # ... and the helper is built with that root
    assert pack.face_helper.model_rootpath == str(extra_root)
    # the copy is not a download - neither weight was fetched
    assert _facelib_downloads(transport) == [], _downloaded(transport)


@pytest.mark.unit
def test_the_weights_stay_where_they_are_when_they_share_a_root(
    loader, transport, mock_folder_paths, tmp_path
):
    """Two weights in the same root are not copied."""
    root = tmp_path / "models" / "facedetection"
    root.mkdir(parents=True)
    (root / "detection_Resnet50_Final.pth").write_bytes(PAYLOAD)
    (root / "parsing_parsenet.pth").write_bytes(PAYLOAD)

    pack = loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    assert pack.face_helper.model_rootpath == str(root)
    assert _facelib_downloads(transport) == [], _downloaded(transport)


@pytest.mark.unit
def test_the_co_location_never_overwrites_a_file_of_the_user(
    loader, transport, mock_folder_paths, tmp_path
):
    """A parsing net that is already in the root of the detector is kept.

    The co-location must not overwrite a file of the user silently: a file
    that does not match the manifest is kept as a backup.
    """
    primary_root = tmp_path / "models" / "facedetection"
    primary_root.mkdir(parents=True, exist_ok=True)
    (primary_root / "parsing_parsenet.pth").write_bytes(PAYLOAD)

    extra_root = tmp_path / "extra" / "facedetection"
    extra_root.mkdir(parents=True)
    (extra_root / "detection_Resnet50_Final.pth").write_bytes(PAYLOAD)
    user_file = extra_root / "parsing_parsenet.pth"
    user_file.write_bytes(b"the user's own parsing net")

    mock_folder_paths.add_model_folder_path(
        "facedetection", str(extra_root), is_default=False
    )

    loader.load_keep_model_pack("KEEP", "retinaface_resnet50")

    # the file of the user is kept as a backup ...
    backup = extra_root / "parsing_parsenet.pth.bak"
    assert backup.is_file(), sorted(p.name for p in extra_root.iterdir())
    assert backup.read_bytes() == b"the user's own parsing net"
    # ... and the verified file is in place
    assert user_file.read_bytes() == PAYLOAD


# ---------------------------------------------------------------------------
# D4 - the console writes of the vendored helper reach the logger
# ---------------------------------------------------------------------------
class _Collector(logging.Handler):
    """A handler that collects the messages of the pack logger."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.mark.unit
def test_the_prints_of_the_helper_reach_the_logger(
    loader_module, logger_module, loader, monkeypatch
):
    """The vendored helper prints while it is built - the writes become log
    records of the pack logger (and carry the tag)."""
    monkeypatch.setattr(
        loader_module, "_face_restore_helper", lambda: _PrintingHelper
    )

    collector = _Collector()
    logger_module.logger.addHandler(collector)
    try:
        loader.load_keep_model_pack("KEEP", "retinaface_resnet50")
    finally:
        logger_module.logger.removeHandler(collector)

    assert any(
        "the detection model is ready" in message for message in collector.messages
    ), collector.messages
    assert any(
        message.startswith("[KEEP]") for message in collector.messages
    ), collector.messages
