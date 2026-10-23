"""Root ``conftest.py`` - the ComfyUI-KEEP test harness (P1.2).

This file is the harness the whole modernization plan runs on.  It does three
things, in this order:

1. **Package alias.**  The repository directory name (``ComfyUI-KEEP``) is not a
   valid Python identifier, so the package is registered under the alias the
   plan mandates (``ComfyUI_KEEP``) with ``__path__`` **and**
   ``submodule_search_locations`` set - the same trick the reference pack
   ComfyUI-DyPE uses.  The alias is registered *without executing* the root
   ``__init__.py``; that is allowed only for ``modules/*`` consumers, because
   the root module imports ComfyUI.  :func:`execute_package_init` runs the real
   entry point (node tests, P3.4/P4.4) once the mocks below are installed.

2. **ComfyUI mocks.**  ``comfy_api.latest`` (``io``/``ui`` as *sibling*
   exports, ``ComfyExtension``, ``ComfyAPI``/``ComfyAPISync``),
   ``comfy.model_management``, ``comfy.model_patcher``, ``comfy.utils`` and
   ``folder_paths`` are installed into ``sys.modules`` before any test module
   is imported.  The mocks mirror the *verified* surface of ComfyUI v0.42.0
   (``docs/audit/2026-10-23-comfy-api-surface.md``); in particular ``ui`` is
   **not** an attribute of ``io``, and ``execution`` is an instance attribute
   (class-level access raises ``AttributeError``), exactly like the real API.

3. **Fixtures.**  ``sample_image`` (deterministic seeded 64x64 RGB tensor) and
   the autouse ``mock_folder_paths`` reset, which gives every test its own
   ``models_dir`` and its own folder registry.

P1.2 **replaces** the P0 bootstrap: the alias set is exactly the plan's, the
ComfyUI mocks are added, and the pytest collection artifact
(``sys.modules["__init__"]``) is dropped - it was only needed before P1.1's
``--import-mode=importlib``.
"""

from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import pytest
import torch

#: Absolute path of the repository root (this file's directory).
PROJECT_ROOT = Path(__file__).resolve().parent

# The repo root on sys.path so `from src.xxx import ...` / `import modules.xxx`
# work in the tests (mirrors the reference pack).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Regular packages, not test modules - they are imported explicitly by the
# tests (via the alias) and must not be collected by pytest.
collect_ignore = ["__init__.py", "src", "nodes"]

#: The package alias mandated by the plan (P1.2).
PACKAGE_ALIAS = "ComfyUI_KEEP"


# ---------------------------------------------------------------------------
# 1. package alias
# ---------------------------------------------------------------------------
def _package_spec(alias: str) -> importlib.machinery.ModuleSpec:
    """Spec of the root ``__init__.py`` under ``alias``."""
    return importlib.util.spec_from_file_location(
        alias,
        str(PROJECT_ROOT / "__init__.py"),
        submodule_search_locations=[str(PROJECT_ROOT)],
    )


def _make_placeholder_logger(alias: str) -> logging.Logger:
    """The logger the root ``__init__.py`` would install.

    ``modules/*`` does ``from .. import logger``; the attribute only exists
    once the root ``__init__.py`` has run.  The register-without-execute mode
    therefore provides the same logger object (same name, level and handler),
    so ``modules/*`` consumers work - the plan's P3 logger constraint, until
    P3.4 moves the logger into ``src/keep_logger.py``.
    """
    logger = logging.getLogger(alias)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def register_package_alias(alias: str = PACKAGE_ALIAS) -> types.ModuleType:
    """Register the package under ``alias`` **without executing** it.

    The module object gets ``__path__`` and ``submodule_search_locations``, so
    ``import <alias>.modules.utils`` resolves inside the repository - but the
    root ``__init__.py`` itself is not run (it needs ComfyUI).
    """
    existing = sys.modules.get(alias)
    if existing is not None:
        return existing
    module = importlib.util.module_from_spec(_package_spec(alias))
    module.__path__ = [str(PROJECT_ROOT)]  # type: ignore[attr-defined]
    module.logger = _make_placeholder_logger(alias)  # type: ignore[attr-defined]
    sys.modules[alias] = module
    return module


def execute_package_init(alias: str = PACKAGE_ALIAS) -> types.ModuleType:
    """Execute the root ``__init__.py`` under ``alias`` and return the module.

    This is the path the node tests use (P1.2 mocks + the P3.4/P4.4 V3
    entrypoint): the package is really executed, so ``comfy_entrypoint`` /
    ``NODE_CLASS_MAPPINGS`` and the ``logger`` the ``modules/*`` code imports
    via ``from .. import logger`` are real.  The register-without-execute stub
    is replaced.
    """
    _install_comfy_mocks()
    module = importlib.util.module_from_spec(_package_spec(alias))
    sys.modules[alias] = module
    module.__spec__.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# 2. ComfyUI mocks
# ---------------------------------------------------------------------------
class _MockSchema:
    """Mock of ``comfy_api.latest.io.Schema`` (a dataclass in the real API)."""

    def __init__(
        self,
        node_id: str | None = None,
        display_name: str | None = None,
        category: str = "sd",
        inputs: list | None = None,
        outputs: list | None = None,
        hidden: list | None = None,
        description: str = "",
        search_aliases: list | None = None,
        is_input_list: bool = False,
        is_output_node: bool = False,
        is_deprecated: bool = False,
        is_experimental: bool = False,
        is_dev_only: bool = False,
        is_api_node: bool = False,
        **extra: object,
    ) -> None:
        self.node_id = node_id
        self.display_name = display_name
        self.category = category
        self.inputs = list(inputs) if inputs is not None else []
        self.outputs = list(outputs) if outputs is not None else []
        self.hidden = list(hidden) if hidden is not None else []
        self.description = description
        self.search_aliases = list(search_aliases) if search_aliases is not None else []
        self.is_input_list = is_input_list
        self.is_output_node = is_output_node
        self.is_deprecated = is_deprecated
        self.is_experimental = is_experimental
        self.is_dev_only = is_dev_only
        self.is_api_node = is_api_node
        # Unknown keyword arguments are kept (the real dataclass would raise);
        # being forgiving keeps the mock usable while a schema is drafted.
        for key, value in extra.items():
            setattr(self, key, value)


class _MockInput:
    """Mock of ``comfy_api.latest.io.Input``."""

    def __init__(
        self,
        id: str,
        display_name: str | None = None,
        optional: bool = False,
        tooltip: str | None = None,
        lazy: bool | None = None,
        extra_dict: dict | None = None,
        raw_link: bool | None = None,
        advanced: bool | None = None,
        **extra: object,
    ) -> None:
        self.id = id
        self.display_name = display_name
        self.optional = optional
        self.tooltip = tooltip
        self.lazy = lazy
        self.extra_dict = dict(extra_dict) if extra_dict is not None else {}
        self.rawLink = raw_link
        self.advanced = advanced
        for key, value in extra.items():
            setattr(self, key, value)

    def as_dict(self) -> dict:
        return {
            "display_name": self.display_name,
            "optional": self.optional,
            "tooltip": self.tooltip,
            "lazy": self.lazy,
            "rawLink": self.rawLink,
            "advanced": self.advanced,
            **self.extra_dict,
        }


class _MockWidgetInput(_MockInput):
    """Mock of the widget inputs (``io.Int.Input``, ``io.Combo.Input``, ...).

    The real widget inputs carry ``default``/``min``/``max``/``step``/``options``
    - the very fields the P4 schema-snapshot tests compare against the V1
    ``INPUT_TYPES``.
    """

    def __init__(
        self,
        id: str,
        default=None,
        min=None,  # noqa: A002 - the real API uses these names
        max=None,  # noqa: A002 - the real API uses these names
        step=None,
        options=None,
        control_after_generate=None,
        **kwargs,
    ) -> None:
        super().__init__(id, **kwargs)
        self.default = default
        self.min = min
        self.max = max
        self.step = step
        self.options = options
        self.control_after_generate = control_after_generate

    def as_dict(self) -> dict:
        return super().as_dict() | {
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "options": self.options,
        }


class _MockOutput:
    """Mock of ``comfy_api.latest.io.Output``."""

    def __init__(
        self,
        display_name: str | None = None,
        tooltip: str | None = None,
        **extra: object,
    ) -> None:
        self.display_name = display_name
        self.tooltip = tooltip
        for key, value in extra.items():
            setattr(self, key, value)


class _MockNodeOutput:
    """Mock of ``comfy_api.latest.io.NodeOutput``.

    ``result`` returns the positional args (or ``None`` when empty), exactly
    like the real class, so node code can return ``io.NodeOutput(value)``.
    """

    def __init__(
        self,
        *args: object,
        ui: object | None = None,
        expand: dict | None = None,
        block_execution: str | None = None,
    ) -> None:
        self.args = args
        self.ui = ui
        self.expand = expand
        self.block_execution = block_execution

    @property
    def result(self):
        return self.args if self.args else None

    def __getitem__(self, index):
        return self.args[index]

    def __iter__(self):
        return iter(self.args)


class _MockComfyNode:
    """Mock of ``comfy_api.latest.io.ComfyNode``.

    The real base class raises ``NotImplementedError`` for the abstract hooks
    and provides ``check_lazy_status``; the mock keeps that behaviour so a
    node that forgets to implement a hook fails loudly instead of silently
    passing.
    """

    @classmethod
    def define_schema(cls) -> _MockSchema:
        raise NotImplementedError

    @classmethod
    def execute(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def validate_inputs(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def check_lazy_status(cls, **kwargs) -> list[str]:
        return [name for name in kwargs if kwargs[name] is None]

    @classmethod
    def GET_SCHEMA(cls) -> _MockSchema:
        """Validate/finalize the schema once per class (real API behaviour)."""
        if "_schema_cache" not in cls.__dict__:
            cls._schema_cache = cls.define_schema()  # type: ignore[attr-defined]
        return cls._schema_cache  # type: ignore[attr-defined]


class _MockTypeIO:
    """Base of the mock socket types (``io.Image``, ``io.Int``, ...)."""

    Input = _MockInput
    Output = _MockOutput
    type_name: str = ""


class _MockIO:
    """Mock of ``comfy_api.latest.io`` (the *sibling* of ``ui``)."""

    Schema = _MockSchema
    Input = _MockInput
    Output = _MockOutput
    NodeOutput = _MockNodeOutput
    ComfyNode = _MockComfyNode

    class Boolean(_MockTypeIO):
        type_name = "BOOLEAN"
        Input = _MockWidgetInput

    class Int(_MockTypeIO):
        type_name = "INT"
        Input = _MockWidgetInput

    class Float(_MockTypeIO):
        type_name = "FLOAT"
        Input = _MockWidgetInput

    class String(_MockTypeIO):
        type_name = "STRING"
        Input = _MockWidgetInput

    class Combo(_MockTypeIO):
        type_name = "COMBO"
        Input = _MockWidgetInput

    class MultiCombo(_MockTypeIO):
        type_name = "MULTICOMBO"
        Input = _MockWidgetInput

    class Image(_MockTypeIO):
        type_name = "IMAGE"

    class Mask(_MockTypeIO):
        type_name = "MASK"

    class Latent(_MockTypeIO):
        type_name = "LATENT"

    class Conditioning(_MockTypeIO):
        type_name = "CONDITIONING"

    class Model(_MockTypeIO):
        type_name = "MODEL"

    class ModelPatch(_MockTypeIO):
        type_name = "MODEL_PATCH"

    class Vae(_MockTypeIO):
        type_name = "VAE"

    class UpscaleModel(_MockTypeIO):
        type_name = "UPSCALE_MODEL"

    class Audio(_MockTypeIO):
        type_name = "AUDIO"

    class Video(_MockTypeIO):
        type_name = "VIDEO"

    class Hidden:
        unique_id = "unique_id"
        prompt = "prompt"

    @staticmethod
    def Custom(io_type: str):
        """Mock of ``io.Custom("TYPE")`` - a socket type by name."""
        return type(
            f"Custom_{io_type}",
            (_MockTypeIO,),
            {"type_name": io_type},
        )


class _MockUI:
    """Mock of ``comfy_api.latest.ui`` (the *sibling* of ``io``)."""

    class PreviewImage:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class PreviewAudio:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class SaveImage:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs


class _MockComfyExtension:
    """Mock of ``comfy_api.latest.ComfyExtension``."""

    async def on_load(self) -> None:
        return None

    async def get_node_list(self) -> list:
        return []


class _SyncExecution:
    """``ComfyAPISync().execution`` - a *synchronous* progress recorder."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def set_progress(
        self,
        value: float,
        max_value: float,
        node_id: str | None = None,
        preview_image: object | None = None,
        ignore_size_limit: bool = False,
    ) -> None:
        self.calls.append((value, max_value))


class _AsyncExecution:
    """``ComfyAPI().execution`` - an *asynchronous* progress recorder."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    async def set_progress(
        self,
        value: float,
        max_value: float,
        node_id: str | None = None,
        preview_image: object | None = None,
        ignore_size_limit: bool = False,
    ) -> None:
        self.calls.append((value, max_value))


class _MockComfyAPI:
    """Mock of ``comfy_api.latest.ComfyAPI`` (async execution API)."""

    def __init__(self) -> None:
        # Instance attribute on purpose: the real API raises AttributeError on
        # class-level access (`ComfyAPI.execution`), see the P4.2 audit.
        self.execution = _AsyncExecution()


class _MockComfyAPISync:
    """Mock of ``comfy_api.latest.ComfyAPISync`` (synchronous execution API)."""

    def __init__(self) -> None:
        self.execution = _SyncExecution()


class _MockModelPatcher:
    """Mock of ``comfy.model_patcher.ModelPatcher``."""

    def __init__(self, model=None, *args, **kwargs) -> None:
        self.model = model if model is not None else types.SimpleNamespace()
        if not hasattr(self.model, "diffusion_model"):
            self.model.diffusion_model = types.SimpleNamespace()
        self._object_patches: dict = {}
        self._unet_wrapper = None

    def clone(self) -> "_MockModelPatcher":
        new = _MockModelPatcher()
        new.model = copy.copy(self.model)
        new.model.diffusion_model = copy.copy(self.model.diffusion_model)
        new._object_patches = dict(self._object_patches)
        new._unet_wrapper = self._unet_wrapper
        return new

    def add_object_patch(self, path: str, obj) -> None:
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn) -> None:
        self._unet_wrapper = fn


class _MockModelManagement(types.ModuleType):
    """Mock of ``comfy.model_management`` - records the residency calls."""

    def __init__(self) -> None:
        super().__init__("comfy.model_management")
        self.calls: list[tuple] = []

    def load_models_gpu(self, models, memory_required=None, **kwargs) -> None:
        self.calls.append(("load_models_gpu", models, memory_required))

    def load_model_gpu(self, model):
        self.calls.append(("load_model_gpu", model))
        return model

    def unload_all_models(self) -> None:
        self.calls.append(("unload_all_models",))

    def soft_empty_cache(self) -> None:
        self.calls.append(("soft_empty_cache",))

    def get_torch_device(self):
        self.calls.append(("get_torch_device",))
        return torch.device("cpu")

    def unet_offload_device(self):
        self.calls.append(("unet_offload_device",))
        return torch.device("cpu")

    def cast_to_device(self, tensor, device=None, dtype=None):
        return tensor

    def unet_dtype(self, *args, **kwargs):
        return torch.float32

    def should_use_fp16(self, *args, **kwargs) -> bool:
        return False

    def get_free_memory(self, *args, **kwargs):
        return 0, 0


class _MockProgressBar:
    """Mock of ``comfy.utils.ProgressBar``."""

    def __init__(self, total: int = 0) -> None:
        self.total = total
        self.current = 0

    def update_absolute(self, value, total=None, preview=None) -> None:
        self.current = value
        if total is not None:
            self.total = total

    def update(self, value) -> None:
        self.current += value

    def close(self) -> None:
        return None


def _mock_tiled_scale(samples, function, tile_x=512, tile_y=512, overlap=64, upscale_amount=1, **kwargs):
    """Mock of ``comfy.utils.tiled_scale`` - a passthrough (no tiling)."""
    return function(samples)


class _MockFolderPaths(types.ModuleType):
    """Mock of ComfyUI's top-level ``folder_paths`` module.

    ``models_dir`` and the folder registry live in memory, so the autouse
    ``mock_folder_paths`` fixture can hand every test its own directories.
    """

    def __init__(self) -> None:
        super().__init__("folder_paths")
        self.base_path = str(PROJECT_ROOT)
        self.models_dir = str(PROJECT_ROOT / "models")
        self.folder_names_and_paths: dict[str, list[tuple[str, bool]]] = {}
        self.folder_names_and_paths_with_mtime: dict[str, list] = {}
        self.custom_folder_paths: dict[str, list[tuple[str, bool]]] = {}
        self.calls: list[tuple] = []

    def add_model_folder_path(self, name: str, path: str, is_default: bool = False) -> None:
        self.calls.append(("add_model_folder_path", name, path, is_default))
        self.folder_names_and_paths.setdefault(name, []).append((str(path), is_default))

    def get_folder_paths(self, name: str) -> list[str]:
        self.calls.append(("get_folder_paths", name))
        return [path for path, _ in self.folder_names_and_paths.get(name, [])]

    def get_full_path_or_raise(self, name: str, filename: str) -> str:
        self.calls.append(("get_full_path_or_raise", name, filename))
        for path, _ in self.folder_names_and_paths.get(name, []):
            candidate = os.path.join(path, filename)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"filename not found in {name}: {filename!r} "
            f"(searched: {self.folder_names_and_paths.get(name, [])})"
        )

    def get_filename_list(self, name: str) -> list[str]:
        self.calls.append(("get_filename_list", name))
        found: list[str] = []
        for path, _ in self.folder_names_and_paths.get(name, []):
            if not os.path.isdir(path):
                continue
            for entry in sorted(os.listdir(path)):
                if os.path.isfile(os.path.join(path, entry)):
                    found.append(entry)
        return found

    def exists_annotated_filepath(self, name: str, filename: str) -> tuple[str, str | None]:
        return self.get_full_path_or_raise(name, filename), None


def _install_comfy_mocks() -> None:
    """Install the ComfyUI mocks into ``sys.modules``.

    Nothing is overwritten: if a real ComfyUI is already imported (the
    ``headless`` tier runs the helper in its own interpreter), the real
    modules stay in charge.
    """
    if "comfy_api.latest" in sys.modules or "comfy" in sys.modules:
        return

    # --- comfy_api.latest -------------------------------------------------
    mock_comfy_api = types.ModuleType("comfy_api")
    mock_comfy_api.__path__ = []  # type: ignore[attr-defined]
    mock_comfy_api_latest = types.ModuleType("comfy_api.latest")

    mock_io = _MockIO()
    mock_ui = _MockUI()

    mock_comfy_api_latest.io = mock_io
    mock_comfy_api_latest.ui = mock_ui
    mock_comfy_api_latest.IO = mock_io
    mock_comfy_api_latest.UI = mock_ui
    mock_comfy_api_latest.ComfyExtension = _MockComfyExtension
    mock_comfy_api_latest.ComfyAPI = _MockComfyAPI
    mock_comfy_api_latest.ComfyAPISync = _MockComfyAPISync

    # --- comfy.model_management / comfy.model_patcher / comfy.utils ------
    mock_model_management = _MockModelManagement()

    mock_model_patcher = types.ModuleType("comfy.model_patcher")
    mock_model_patcher.ModelPatcher = _MockModelPatcher  # type: ignore[attr-defined]

    mock_comfy_utils = types.ModuleType("comfy.utils")
    mock_comfy_utils.ProgressBar = _MockProgressBar  # type: ignore[attr-defined]
    mock_comfy_utils.tiled_scale = _mock_tiled_scale  # type: ignore[attr-defined]

    # --- comfy (top level) ------------------------------------------------
    mock_comfy = types.ModuleType("comfy")
    mock_comfy.__path__ = []  # type: ignore[attr-defined]
    mock_comfy.model_management = mock_model_management  # type: ignore[attr-defined]
    mock_comfy.model_patcher = mock_model_patcher  # type: ignore[attr-defined]
    mock_comfy.utils = mock_comfy_utils  # type: ignore[attr-defined]

    # --- folder_paths (top level, not a comfy submodule) ------------------
    mock_folder_paths = _MockFolderPaths()

    sys.modules["comfy"] = mock_comfy
    sys.modules["comfy.model_management"] = mock_model_management
    sys.modules["comfy.model_patcher"] = mock_model_patcher
    sys.modules["comfy.utils"] = mock_comfy_utils
    sys.modules["comfy_api"] = mock_comfy_api
    sys.modules["comfy_api.latest"] = mock_comfy_api_latest
    sys.modules["folder_paths"] = mock_folder_paths


# Install the mocks immediately (before pytest imports any test module).
_install_comfy_mocks()

# The plan's package alias (P1.2) - registered, not executed.
register_package_alias()

#: The name pytest imports the repository root under.
#:
#: The repository root is itself a Python package (it has an ``__init__.py``),
#: so pytest's ``Package.setup()`` imports that file during collection - under
#: the name ``__init__`` - **independently of the import mode**.  The root
#: module uses *relative* imports (``from .modules import utils``), so it can
#: only be imported as part of a package; registering the alias module under
#: this name as well keeps pytest from executing the entry point a second
#: time (which would fail with "attempted relative import with no known
#: parent package").
#:
#: This is the P0 "collection artifact" that P1.2 planned to drop - it cannot
#: be dropped while the repository root is a package (see
#: ``docs/audit/2026-10-23-p1-test-infrastructure.md``).
COLLECTION_MODULE_NAME = "__init__"
sys.modules.setdefault(COLLECTION_MODULE_NAME, sys.modules[PACKAGE_ALIAS])


# ---------------------------------------------------------------------------
# 3. fixtures
# ---------------------------------------------------------------------------
def _make_sample_image() -> torch.Tensor:
    """Deterministic seeded 64x64 RGB IMAGE tensor ``(1, 64, 64, 3)``."""
    generator = torch.Generator().manual_seed(1234)
    return torch.rand((1, 64, 64, 3), generator=generator, dtype=torch.float32)


@pytest.fixture
def sample_image() -> torch.Tensor:
    """A deterministic ``(1, 64, 64, 3)`` float32 RGB tensor in ``[0, 1]``."""
    return _make_sample_image()


@pytest.fixture(autouse=True)
def mock_folder_paths(tmp_path):
    """Autouse: every test gets its own ``models_dir`` and folder registry.

    The reset happens *before* the test (so a test never sees the directory of
    another test) and is undone afterwards.

    The fixture is defensive on purpose: a test that installs its own
    ``folder_paths`` stand-in (the P0 dependency-decision probe does) keeps
    working - only the attributes that exist are isolated.
    """
    folder_paths = sys.modules.get("folder_paths")
    if folder_paths is None:
        yield None
        return

    original_models_dir = getattr(folder_paths, "models_dir", None)
    registry = getattr(folder_paths, "folder_names_and_paths", None)
    original_registry = (
        {key: list(value) for key, value in registry.items()} if registry is not None else None
    )
    with_mtime = getattr(folder_paths, "folder_names_and_paths_with_mtime", None)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    folder_paths.models_dir = str(models_dir)
    if registry is not None:
        registry.clear()
    if with_mtime is not None:
        with_mtime.clear()

    yield folder_paths

    if original_models_dir is not None:
        folder_paths.models_dir = original_models_dir
    if registry is not None and original_registry is not None:
        registry.clear()
        registry.update(original_registry)
    if with_mtime is not None:
        with_mtime.clear()
