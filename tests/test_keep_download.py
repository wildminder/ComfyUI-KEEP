"""P5.2 - the manifest-driven downloader.

The downloader turns a manifest entry into a file on disk.  The tests replace
the transport (``torch.hub.download_url_to_file``) with a double, so the whole
tier runs **without a network**:

* a fresh download writes the file and verifies its sha256;
* a cached and valid file is returned **without** a download;
* a sha256 mismatch deletes the file and downloads it **once** more;
* a failed download leaves **no** partial file and an actionable error;
* a weight in an extra model root is used where it is (the primary root wins);
* an entry without a hash is downloaded **unverified** - with a loud warning;
* an entry that is not published upstream raises an actionable error naming the
  file, the expected path and the dead URL.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def download_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_download")


@pytest.fixture(scope="module")
def configs_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_model_configs")


@pytest.fixture
def keep_records(download_module, caplog):
    """The records of the pack logger - in **both** logger modes.

    Without a ComfyUI the pack logger owns a private handler and does not
    propagate, so the records never reach the root logger of pytest: the
    capture handler is attached to the pack logger itself.
    """
    logger = download_module.logger
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def _entry(configs_module, name="KEEP-b76feb75.pth", payload=b"payload", **overrides):
    """A manifest entry with the sha256 of ``payload``."""
    fields = {
        "name": name,
        "url": f"https://example.invalid/{name}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dest_folder": "keep_models",
        "subdir": "KEEP",
        "required": True,
    }
    fields.update(overrides)
    return configs_module.WeightEntry(**fields)


class _Transport:
    """The transport double: it records the calls and writes the payload."""

    def __init__(self, payload: bytes = b"payload", fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.calls: list[str] = []
        self.progress: list[bool] = []

    def __call__(self, url, dst, hash_prefix=None, progress=True):
        self.calls.append(url)
        self.progress.append(progress)
        if self.fail:
            # the real transport leaves a temporary file behind
            with open(dst + ".tmp", "wb") as handle:
                handle.write(self.payload)
            raise RuntimeError("the server is not reachable")
        with open(dst, "wb") as handle:
            handle.write(self.payload)


@pytest.mark.unit
def test_a_fresh_download_is_written_and_verified(
    download_module, configs_module, monkeypatch
):
    """The file lands in the resolved destination and its hash is checked."""
    entry = _entry(configs_module)
    transport = _Transport()
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    path = download_module.ensure_required(entry)

    assert os.path.isfile(path)
    assert path.endswith(os.path.join("keep_models", "KEEP", entry.name)), path
    assert transport.calls == [entry.url]
    assert download_module.sha256_of(path) == entry.sha256


@pytest.mark.unit
def test_a_cached_and_valid_file_is_not_downloaded_again(
    download_module, configs_module, monkeypatch
):
    """A file that is already there is used - no second download."""
    entry = _entry(configs_module)
    target = download_module.target_path(entry)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(b"payload")

    transport = _Transport()
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    path = download_module.ensure_required(entry)

    assert path == target
    assert transport.calls == [], "the cached file must not be downloaded again"


@pytest.mark.unit
def test_a_sha_mismatch_deletes_the_file_and_downloads_it_once(
    download_module, configs_module, monkeypatch
):
    """The corrupted file is removed and fetched again (exactly once)."""
    entry = _entry(configs_module)
    target = download_module.target_path(entry)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(b"corrupted")

    transport = _Transport()
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    path = download_module.ensure_required(entry)

    assert path == target
    assert transport.calls == [entry.url], "the file must be downloaded exactly once"
    assert download_module.sha256_of(path) == entry.sha256


@pytest.mark.unit
def test_a_persistent_sha_mismatch_raises_and_leaves_no_file(
    download_module, configs_module, monkeypatch
):
    """A file that never matches the manifest is deleted and reported."""
    entry = _entry(configs_module, payload=b"payload")
    transport = _Transport(payload=b"something else")
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    with pytest.raises(ValueError) as excinfo:
        download_module.ensure_required(entry)

    message = str(excinfo.value)
    assert entry.name in message
    assert entry.url in message
    assert len(transport.calls) == 2, "one retry, then the error"
    target = download_module.target_path(entry)
    assert not os.path.exists(target), "the mismatching file must be deleted"


@pytest.mark.unit
def test_a_failed_download_leaves_no_partial_file(
    download_module, configs_module, monkeypatch
):
    """A failed download cleans up and names the URL and the destination."""
    entry = _entry(configs_module)
    transport = _Transport(fail=True)
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    with pytest.raises(FileNotFoundError) as excinfo:
        download_module.ensure_required(entry)

    message = str(excinfo.value)
    assert entry.url in message
    assert entry.name in message
    target = download_module.target_path(entry)
    assert entry.dest_folder in target
    assert not os.path.exists(target), "no partial file may be left behind"
    assert not os.path.exists(target + ".tmp"), "no temporary file may be left behind"


@pytest.mark.unit
def test_a_weight_of_an_extra_model_root_is_used_where_it_is(
    download_module, configs_module, monkeypatch, mock_folder_paths
):
    """The multi-root handling: the file is found in the extra root."""
    entry = _entry(configs_module)
    extra = str(Path(mock_folder_paths.models_dir).parent / "extra-models")
    os.makedirs(os.path.join(extra, "KEEP"), exist_ok=True)
    mock_folder_paths.add_model_folder_path("keep_models", extra)
    extra_file = os.path.join(extra, "KEEP", entry.name)
    with open(extra_file, "wb") as handle:
        handle.write(b"payload")

    transport = _Transport()
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    path = download_module.ensure_required(entry)

    assert path == extra_file, path
    assert transport.calls == [], "the file must not be downloaded again"
    # nothing was written into the primary root
    primary = os.path.join(
        mock_folder_paths.models_dir, "keep_models", "KEEP", entry.name
    )
    assert not os.path.exists(primary), primary


@pytest.mark.unit
def test_the_primary_root_wins_over_an_extra_model_root(
    download_module, configs_module, monkeypatch, mock_folder_paths
):
    """The primary root is searched first."""
    entry = _entry(configs_module)
    extra = str(Path(mock_folder_paths.models_dir).parent / "extra-models")
    os.makedirs(os.path.join(extra, "KEEP"), exist_ok=True)
    mock_folder_paths.add_model_folder_path("keep_models", extra)
    with open(os.path.join(extra, "KEEP", entry.name), "wb") as handle:
        handle.write(b"payload")

    primary_file = os.path.join(
        mock_folder_paths.models_dir, "keep_models", "KEEP", entry.name
    )
    os.makedirs(os.path.dirname(primary_file), exist_ok=True)
    with open(primary_file, "wb") as handle:
        handle.write(b"payload")

    transport = _Transport()
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    path = download_module.ensure_required(entry)

    assert path == primary_file, path
    assert transport.calls == []


@pytest.mark.unit
def test_a_missing_weight_is_downloaded_next_to_the_pack(
    download_module, configs_module, monkeypatch, mock_folder_paths
):
    """The target follows the root that holds the **other** weights of the pack.

    A user who keeps the weights of the pack in an extra model root must not
    get a second copy of them in the primary root of the installation.
    """
    entry = _entry(configs_module)
    extra = str(Path(mock_folder_paths.models_dir).parent / "extra-models")
    os.makedirs(os.path.join(extra, "KEEP"), exist_ok=True)
    mock_folder_paths.add_model_folder_path("keep_models", extra)
    # the sibling of the entry - the other weight of the pack - lives there
    with open(os.path.join(extra, "KEEP", "KEEP_Asian-4765ebe0.pth"), "wb") as handle:
        handle.write(b"payload")

    transport = _Transport()
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    path = download_module.ensure_required(entry)

    assert path == os.path.join(extra, "KEEP", entry.name), path
    assert transport.calls == [entry.url]
    # nothing was written into the primary root
    primary = os.path.join(
        mock_folder_paths.models_dir, "keep_models", "KEEP", entry.name
    )
    assert not os.path.exists(primary), primary


@pytest.mark.unit
def test_the_download_target_is_the_primary_root_when_no_root_holds_the_pack(
    download_module, configs_module, monkeypatch, mock_folder_paths
):
    """Without a pack home the primary root is the target."""
    entry = _entry(configs_module)
    transport = _Transport()
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    path = download_module.ensure_required(entry)

    primary = os.path.join(
        mock_folder_paths.models_dir, "keep_models", "KEEP", entry.name
    )
    assert path == primary, path


@pytest.mark.unit
def test_the_transport_is_called_without_a_progress_bar(
    download_module, configs_module, monkeypatch
):
    """``torch.hub`` writes its own (untagged) bar to stderr - it is off."""
    entry = _entry(configs_module)
    transport = _Transport()
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    download_module.ensure_required(entry)

    assert transport.progress == [False], transport.progress


@pytest.mark.unit
def test_an_entry_without_a_hash_is_downloaded_unverified_with_a_warning(
    download_module, configs_module, monkeypatch, keep_records
):
    """The Asian entry: no verification, but a loud warning."""
    entry = _entry(
        configs_module,
        name="KEEP_Asian-4765ebe0.pth",
        sha256=None,
        unverified=True,
        available=True,
        required=False,
    )
    transport = _Transport()
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    with keep_records.at_level("WARNING"):
        path = download_module.ensure_optional(entry)

    assert path is not None and os.path.isfile(path)
    assert transport.calls == [entry.url]
    warnings = [record for record in keep_records.records if record.levelname == "WARNING"]
    assert warnings, "the missing verification must be reported loudly"
    message = " ".join(record.getMessage() for record in warnings)
    assert entry.name in message
    assert "not verified" in message.lower() or "no verified sha256" in message.lower()


@pytest.mark.unit
def test_a_sha_mismatch_in_an_extra_root_never_deletes_the_file_of_the_user(
    download_module, configs_module, monkeypatch, mock_folder_paths
):
    """A weight of the user (an extra model root) is never deleted.

    The pack owns the files of its own installation; a weight in an extra
    model root belongs to the user.  A file that does not match the manifest
    is therefore kept untouched and the verified file is downloaded into the
    primary root.
    """
    entry = _entry(configs_module)
    extra = str(Path(mock_folder_paths.models_dir).parent / "extra-models")
    os.makedirs(os.path.join(extra, "KEEP"), exist_ok=True)
    mock_folder_paths.add_model_folder_path("keep_models", extra)
    extra_file = os.path.join(extra, "KEEP", entry.name)
    with open(extra_file, "wb") as handle:
        handle.write(b"corrupted")

    transport = _Transport()
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    path = download_module.ensure_required(entry)

    # the file of the user is untouched
    with open(extra_file, "rb") as handle:
        assert handle.read() == b"corrupted"
    # the verified file is in the primary root of the installation
    primary = os.path.join(
        mock_folder_paths.models_dir, "keep_models", "KEEP", entry.name
    )
    assert path == primary, path
    assert download_module.sha256_of(path) == entry.sha256
    assert transport.calls == [entry.url]


@pytest.mark.unit
def test_a_failed_download_keeps_the_file_of_the_user(
    download_module, configs_module, monkeypatch, mock_folder_paths
):
    """A failed download destroys nothing - not in an extra root either."""
    entry = _entry(configs_module)
    extra = str(Path(mock_folder_paths.models_dir).parent / "extra-models")
    os.makedirs(os.path.join(extra, "KEEP"), exist_ok=True)
    mock_folder_paths.add_model_folder_path("keep_models", extra)
    extra_file = os.path.join(extra, "KEEP", entry.name)
    with open(extra_file, "wb") as handle:
        handle.write(b"corrupted")

    transport = _Transport(fail=True)
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    with pytest.raises(FileNotFoundError):
        download_module.ensure_required(entry)

    with open(extra_file, "rb") as handle:
        assert handle.read() == b"corrupted", "the file of the user must survive"
    # no partial file is left behind
    assert not os.path.exists(extra_file + ".part")
    assert not os.path.exists(extra_file + ".part.tmp")


@pytest.mark.unit
def test_a_failed_download_keeps_the_file_of_the_installation(
    download_module, configs_module, monkeypatch
):
    """The download is atomic: the old file survives a failed download."""
    entry = _entry(configs_module)
    target = download_module.target_path(entry)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(b"corrupted")

    transport = _Transport(fail=True)
    monkeypatch.setattr(download_module, "download_url_to_file", transport)

    with pytest.raises(FileNotFoundError):
        download_module.ensure_required(entry)

    with open(target, "rb") as handle:
        assert handle.read() == b"corrupted", "the old file must survive"
    assert not os.path.exists(target + ".part")
    assert not os.path.exists(target + ".part.tmp")


@pytest.mark.unit
def test_an_unpublished_entry_raises_an_actionable_error(
    download_module, configs_module
):
    """The Asian checkpoint is not published upstream (HTTP 404)."""
    entry = _entry(
        configs_module,
        name="KEEP_Asian-4765ebe0.pth",
        sha256=None,
        unverified=True,
        available=False,
        required=True,
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        download_module.ensure_required(entry)

    message = str(excinfo.value)
    assert entry.name in message
    assert entry.url in message
    assert "not published upstream" in message
    assert "manually" in message
    # the expected path is part of the message
    assert os.path.join("keep_models", "KEEP", entry.name) in message


@pytest.mark.unit
def test_a_manually_placed_unpublished_file_is_used(
    download_module, configs_module
):
    """A file the user placed by hand is used - no download."""
    entry = _entry(
        configs_module,
        name="KEEP_Asian-4765ebe0.pth",
        sha256=None,
        unverified=True,
        available=False,
        required=True,
    )
    target = download_module.target_path(entry)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(b"manually placed")

    path = download_module.ensure_required(entry)

    assert path == target
