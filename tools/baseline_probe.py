"""P0.1 baseline probe - records how the *current* ComfyUI-KEEP package behaves.

This tool changes no production code. It reproduces, in a throw-away interpreter,
the two things the modernization plan needs a verified baseline for:

1. the package **imports** through ComfyUI's own custom-node loader
   (``importlib.util.spec_from_file_location`` + alias, no server), and
2. the **model build fails**, at the eager basicsr arch scan, with
   ``ImportError: cannot import name 'T' from 're'`` - swallowed by the
   swallow-and-stub blocks of B9, which is why the import itself still succeeds.

Run it with the embedded interpreter:

    C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe tools/baseline_probe.py

Useful flags:

    --json PATH            write the machine-readable report to PATH
    --comfyui-root PATH    ComfyUI source tree to put on sys.path
                           (default: $COMFYUI_ROOT, else the local runtime tree,
                           else the read-only source reference tree)
    --simulate-missing-ffmpeg
                           block ``import ffmpeg`` and record (instead of perform)
                           the ``pip.main([...])`` call of the vendored
                           ``basicsr/utils/video_util.py`` import-time shim.

Exit code is 0 when the probe ran; the *content* of the report (not the exit
code) is the baseline evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Local ComfyUI trees, tried in order when neither ``--comfyui-root`` nor
#: ``COMFYUI_ROOT`` is given.  The first entry is the runtime tree that owns the
#: embedded interpreter used by this repository; the second is the read-only
#: v0.42.0 source reference.  Override with ``--comfyui-root`` / ``COMFYUI_ROOT``.
DEFAULT_COMFYUI_ROOTS = (
    r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI",
    r"C:\_Dev\ComfyUI_dev\ComfyUI-source\ComfyUI",
)


def resolve_comfyui_root(explicit: str | None) -> str | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env_root = os.environ.get("COMFYUI_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.extend(DEFAULT_COMFYUI_ROOTS)
    for candidate in candidates:
        if candidate and (Path(candidate) / "folder_paths.py").is_file():
            return str(Path(candidate))
    return None


def _exc_info(exc: BaseException) -> dict:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _symbol_origin(module, name: str) -> str | None:
    """``__module__`` of ``module.name`` (``None`` when the symbol is absent)."""
    symbol = getattr(module, name, None)
    return getattr(symbol, "__module__", None)


def _is_local_stub(module, name: str) -> bool:
    """``True`` when ``module.name`` is a stub defined in ``module`` itself.

    A symbol that does not exist at all (P2.1 removed the vendored
    ``VideoReader`` / ``VideoWriter``) is **not** a stub.
    """
    origin = _symbol_origin(module, name)
    return origin is not None and origin == module.__name__


class _BlockFFmpegFinder:
    """meta_path finder that makes ``import ffmpeg`` fail (simulates a host
    without ``ffmpeg-python``) so the vendored pip shim can be observed."""

    def find_module(self, fullname, path=None):  # pragma: no cover - py2 API
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ffmpeg" or fullname.startswith("ffmpeg."):
            raise ImportError(f"blocked by probe: {fullname} is unavailable")
        return None


def _install_pip_main_recorder():
    """Replace ``pip.main`` with a recorder so the shim can be *observed*
    without actually installing anything into the environment."""
    calls = []

    import pip  # noqa: PLC0415 - optional dependency, imported defensively

    original = pip.main

    def recorder(args, *a, **kw):
        calls.append(list(args))
        return 0

    pip.main = recorder
    return calls, original, pip


def load_package_like_comfyui(package_dir: Path):
    """Load the package exactly the way ``ComfyUI/nodes.py:load_custom_node`` does.

    ComfyUI (v0.42.0, ``nodes.py:2255-2276``) uses, for a directory package:
        sys_module_name = module_path.replace(".", "_x_")
        module_spec = spec_from_file_location(sys_module_name, <dir>/__init__.py)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[sys_module_name] = module
        module_spec.loader.exec_module(module)
    """
    module_path = str(package_dir)
    sys_module_name = module_path.replace(".", "_x_")
    module_spec = importlib.util.spec_from_file_location(
        sys_module_name, os.path.join(module_path, "__init__.py")
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[sys_module_name] = module
    module_spec.loader.exec_module(module)
    return sys_module_name, module


def _v3_nodes(package) -> tuple[list[str], dict[str, str]]:
    """The node ids and display names of a **V3** package (P4.4).

    The extension is loaded the way ``nodes.py:2316`` does it: through the
    module level ``comfy_entrypoint``.  An empty result means the package is
    not a V3 extension.
    """
    entrypoint = getattr(package, "comfy_entrypoint", None)
    if entrypoint is None:
        return [], {}

    async def collect():
        extension = await entrypoint()
        await extension.on_load()
        node_list = await extension.get_node_list()
        return [cls.GET_SCHEMA() for cls in node_list]

    try:
        schemas = asyncio.run(collect())
    except Exception:  # noqa: BLE001 - the probe reports everything
        return [], {}
    return (
        sorted(schema.node_id for schema in schemas),
        {schema.node_id: schema.display_name for schema in schemas},
    )


def probe(simulate_missing_ffmpeg: bool = False) -> dict:
    report: dict = {
        "repo_root": str(REPO_ROOT),
        "python": sys.version,
        "python_executable": sys.executable,
        "simulate_missing_ffmpeg": simulate_missing_ffmpeg,
    }

    comfyui_root = resolve_comfyui_root(None)
    report["comfyui_root"] = comfyui_root
    if comfyui_root is None:
        report["fatal"] = (
            "No ComfyUI source tree found. Set COMFYUI_ROOT or pass --comfyui-root."
        )
        return report

    sys.path.insert(0, comfyui_root)

    # Record pip.main calls *before* the package import so the vendored
    # import-time shim cannot install anything into the shared environment.
    pip_calls, pip_main_original, pip_module = _install_pip_main_recorder()
    if simulate_missing_ffmpeg:
        sys.meta_path.insert(0, _BlockFFmpegFinder())
    report["pip_main_calls"] = pip_calls

    # ------------------------------------------------------------------
    # 1. import through ComfyUI's loader
    # ------------------------------------------------------------------
    package_dir = REPO_ROOT
    try:
        sys_module_name, package = load_package_like_comfyui(package_dir)
        report["import"] = {"ok": True, "error": None, "sys_module_name": sys_module_name}
        report["node_class_mappings"] = sorted(
            getattr(package, "NODE_CLASS_MAPPINGS", {}).keys()
        )
        report["node_display_name_mappings"] = dict(
            getattr(package, "NODE_DISPLAY_NAME_MAPPINGS", {})
        )
        # P4.4: the pack registers as a V3 extension - the node ids live in
        # the schemas of the node classes, not in a V1 mapping.
        v3_ids, v3_display_names = _v3_nodes(package)
        report["v3_node_ids"] = v3_ids
        report["v3_display_names"] = v3_display_names
    except BaseException as exc:  # noqa: BLE001 - the probe reports everything
        report["import"] = {"ok": False, "error": _exc_info(exc)}
        pip_module.main = pip_main_original
        return report

    utils_mod = sys.modules[f"{sys_module_name}.src.keep_model_configs"]
    loader_mod = sys.modules[f"{sys_module_name}.src.keep_model_loader"]
    processor_mod = sys.modules[f"{sys_module_name}.src.keep_processor"]

    # ------------------------------------------------------------------
    # 2. which swallow-and-stub block swallowed which failure (B9)
    # ------------------------------------------------------------------
    arch_registry = getattr(utils_mod, "ARCH_REGISTRY", None)
    registry_type = type(arch_registry).__name__ if arch_registry is not None else None
    try:
        has_keep = arch_registry.get("KEEP") is not None
    except Exception as exc:  # noqa: BLE001
        has_keep = f"raised {type(exc).__name__}: {exc}"
    report["arch_registry"] = {
        "class_name": registry_type,
        "module": type(arch_registry).__module__ if arch_registry is not None else None,
        "is_stub": registry_type == "DummyArchRegistry",
        "get_keep": has_keep,
    }
    report["stubs"] = {
        "utils_arch_registry_dummy": registry_type == "DummyArchRegistry",
        "utils_load_file_from_url_fallback": (
            # P4.7: the dead import is gone - an absent symbol is not a stub.
            getattr(utils_mod, "load_file_from_url", None) is not None
            and getattr(utils_mod, "load_file_from_url").__module__
            == utils_mod.__name__
        ),
        "processor_img2tensor_is_dummy": (
            getattr(processor_mod, "img2tensor").__module__ == processor_mod.__name__
        ),
        "processor_tensor2img_is_dummy": (
            getattr(processor_mod, "tensor2img").__module__ == processor_mod.__name__
        ),
        "processor_is_gray_is_dummy": (
            getattr(processor_mod, "is_gray").__module__ == processor_mod.__name__
        ),
        # P2.1 deleted the vendored ``video_util`` module (its import-time pip
        # shim); ``VideoReader`` / ``VideoWriter`` are therefore *absent* from
        # the processor instead of being dummy stubs.  An absent symbol is not
        # a stub - the dead ``process_video`` path raises instead.
        "processor_video_reader_is_dummy": _is_local_stub(
            processor_mod, "VideoReader"
        ),
        "processor_video_writer_is_dummy": _is_local_stub(
            processor_mod, "VideoWriter"
        ),
        "loader_face_restore_helper_is_dummy": (
            getattr(loader_mod, "FaceRestoreHelper").__module__ == loader_mod.__name__
        ),
    }
    report["stub_summary"] = {
        "fired": sorted(k for k, v in report["stubs"].items() if v),
        "not_fired": sorted(k for k, v in report["stubs"].items() if not v),
    }
    report["symbol_origins"] = {
        "img2tensor": getattr(processor_mod, "img2tensor").__module__,
        "tensor2img": getattr(processor_mod, "tensor2img").__module__,
        "is_gray": getattr(processor_mod, "is_gray").__module__,
        "VideoReader": _symbol_origin(processor_mod, "VideoReader"),
        "FaceRestoreHelper": getattr(loader_mod, "FaceRestoreHelper").__module__,
    }

    # ------------------------------------------------------------------
    # 3. state of the interpreter *right after* the package import:
    #    the pip shim (B10) and the global sys.path shadowing (B4).
    #    Captured before the cleanup below, because the eager arch scan
    #    test deliberately purges the partially imported 'basicsr'.
    # ------------------------------------------------------------------
    deps_path = str(REPO_ROOT / "modules" / "deps")
    try:
        ffmpeg_importable = importlib.util.find_spec("ffmpeg") is not None
    except ImportError:
        ffmpeg_importable = False
    report["pip_shim"] = {
        "video_util_imported": any(
            name.endswith("basicsr.utils.video_util") for name in sys.modules
        ),
        "ffmpeg_importable": ffmpeg_importable,
        "pip_main_calls": list(pip_calls),
    }
    report["sys_path_shadowing"] = {
        "deps_path_on_sys_path": deps_path in sys.path,
        "basicsr_in_sys_modules": "basicsr" in sys.modules,
        "basicsr_utils_in_sys_modules": "basicsr.utils" in sys.modules,
        "facelib_in_sys_modules": "facelib" in sys.modules,
    }

    # ------------------------------------------------------------------
    # 4. the *real* (unmasked) import of the vendored KEEP arch (B1/B2)
    # ------------------------------------------------------------------
    try:
        importlib.import_module(
            f"{sys_module_name}.modules.deps.basicsr.archs.keep_arch"
        )
        report["arch_scan"] = {"ok": True, "error": None}
    except BaseException as exc:  # noqa: BLE001
        report["arch_scan"] = {"ok": False, "error": _exc_info(exc)}

    # ------------------------------------------------------------------
    # 5. the diffusers import (B2) - only symbol source in the tree
    # ------------------------------------------------------------------
    try:
        importlib.import_module("diffusers.models.attention")
        report["diffusers"] = {"ok": True, "error": None}
    except BaseException as exc:  # noqa: BLE001
        report["diffusers"] = {"ok": False, "error": _exc_info(exc)}

    # ------------------------------------------------------------------
    # 6. a real model build, with a stubbed weight file present
    # ------------------------------------------------------------------
    import folder_paths  # noqa: PLC0415 - needs the ComfyUI root on sys.path

    with tempfile.TemporaryDirectory(prefix="keep-baseline-weights-") as tmp:
        models_dir = Path(tmp)
        stub_weight = models_dir / "keep_models" / "KEEP" / "KEEP-b76feb75.pth"
        stub_weight.parent.mkdir(parents=True, exist_ok=True)
        stub_weight.write_bytes(b"stub-weight-not-a-real-checkpoint")
        original_models_dir = folder_paths.models_dir
        folder_paths.models_dir = str(models_dir)
        # The probe must never download a weight: the stub file is not the
        # real one (its sha256 does not match the manifest since P5.3), so
        # the resolution would delete it and fetch ~634 MB from upstream.
        # The resolution is therefore stubbed - the build itself stays real.
        download_mod = sys.modules[f"{sys_module_name}.src.keep_download"]
        original_ensure_weight = download_mod.ensure_weight
        original_ensure_required = download_mod.ensure_required
        download_mod.ensure_weight = lambda entry, progress=True: str(stub_weight)
        download_mod.ensure_required = lambda entry, progress=True: str(
            stub_weight.parent / entry.name
        )
        try:
            loader = loader_mod.KEEPModelLoader()
            loader.load_keep_model_pack(
                model_type_str="KEEP", detection_model_str="retinaface_resnet50"
            )
            report["model_build"] = {
                "ok": True,
                "error": None,
                "stub_weight_path": str(stub_weight),
                "stub_weight_present": stub_weight.is_file(),
            }
        except BaseException as exc:  # noqa: BLE001
            report["model_build"] = {
                "ok": False,
                "error": _exc_info(exc),
                "stub_weight_path": str(stub_weight),
                "stub_weight_present": stub_weight.is_file(),
            }
        finally:
            download_mod.ensure_weight = original_ensure_weight
            download_mod.ensure_required = original_ensure_required
            folder_paths.models_dir = original_models_dir

    pip_module.main = pip_main_original
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--comfyui-root", dest="comfyui_root", default=None)
    parser.add_argument(
        "--simulate-missing-ffmpeg",
        action="store_true",
        help="block 'import ffmpeg' and record the vendored pip.main shim call",
    )
    args = parser.parse_args(argv)

    if args.comfyui_root:
        os.environ["COMFYUI_ROOT"] = args.comfyui_root

    report = probe(simulate_missing_ffmpeg=args.simulate_missing_ffmpeg)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if "fatal" not in report else 2


if __name__ == "__main__":
    raise SystemExit(main())
