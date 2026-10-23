"""P0.2 - local weight inventory scanner.

Scans the ComfyUI model folders the node uses and records, per expected
weight, whether it is present on this host, its size and its sha256.

The inventory is a *local* fact: a weight that is not on this host is recorded
as absent with ``sha256: null``.  No hash is ever invented.

Usage (embedded python):

    C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe tools/scan_weights.py
    C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe tools/scan_weights.py --models-root <abs>

The models root is resolved from (in order): ``--models-root``,
``$KEEP_MODEL_DIR``, ``$COMFYUI_ROOT/models``, then the local ComfyUI runtime
tree of this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS_ROOTS = (
    r"C:\_Dev\ComfyUI_dev\ComfyUI_t211c130p313\ComfyUI\models",
)

#: The 8 weights the node needs (plan section 5 / P5.1 manifest scope).
#: ``dest_folder`` / ``subdir`` reproduce the legacy on-disk layout exactly.
EXPECTED_WEIGHTS = (
    {
        "name": "KEEP-b76feb75.pth",
        "dest_folder": "keep_models",
        "subdir": "KEEP",
        "required": True,
        "url": "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP-b76feb75.pth",
    },
    {
        "name": "KEEP_Asian-4765ebe0.pth",
        "dest_folder": "keep_models",
        "subdir": "KEEP",
        "required": True,
        "url": "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP_Asian-4765ebe0.pth",
    },
    {
        "name": "detection_Resnet50_Final.pth",
        "dest_folder": "facedetection",
        "subdir": "",
        "required": True,
        "url": "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/detection_Resnet50_Final.pth",
    },
    {
        "name": "detection_mobilenet0.25_Final.pth",
        "dest_folder": "facedetection",
        "subdir": "",
        "required": True,
        "url": "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/detection_mobilenet0.25_Final.pth",
    },
    {
        "name": "yolov5n-face.pth",
        "dest_folder": "facedetection",
        "subdir": "",
        "required": True,
        "url": "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/yolov5n-face.pth",
    },
    {
        "name": "yolov5l-face.pth",
        "dest_folder": "facedetection",
        "subdir": "",
        "required": True,
        "url": "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/yolov5l-face.pth",
    },
    {
        "name": "parsing_parsenet.pth",
        "dest_folder": "facedetection",
        "subdir": "",
        "required": True,
        "url": "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/parsing_parsenet.pth",
    },
    {
        "name": "gmflow_sintel-0c07dcb3.pth",
        "dest_folder": "gmflow",
        "subdir": "",
        "required": False,
        "url": "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/gmflow_sintel-0c07dcb3.pth",
    },
)

#: The folders that are scanned (the plan names these three explicitly).
SCANNED_FOLDERS = ("keep_models", "facedetection", "gmflow")


def resolve_models_root(explicit: str | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_keep = os.environ.get("KEEP_MODEL_DIR")
    if env_keep:
        candidates.append(Path(env_keep))
    env_comfy = os.environ.get("COMFYUI_ROOT")
    if env_comfy:
        candidates.append(Path(env_comfy) / "models")
    candidates.extend(Path(p) for p in DEFAULT_MODELS_ROOTS)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(name: str, dest_folder: str, subdir: str) -> Path:
    parts = [dest_folder]
    if subdir:
        parts.append(subdir)
    parts.append(name)
    return Path(*parts)


def scan(models_root: Path) -> dict:
    """Return the inventory of the expected weights below ``models_root``."""
    entries = []
    for spec in EXPECTED_WEIGHTS:
        rel = relative_path(spec["name"], spec["dest_folder"], spec["subdir"])
        abs_path = models_root / rel
        present = abs_path.is_file()
        entries.append(
            {
                **spec,
                "relative_path": rel.as_posix(),
                "present": present,
                "size": abs_path.stat().st_size if present else None,
                "sha256": sha256_of(abs_path) if present else None,
            }
        )

    extras = []
    for folder in SCANNED_FOLDERS:
        folder_path = models_root / folder
        if not folder_path.is_dir():
            continue
        known = {
            (models_root / e["relative_path"]).resolve()
            for e in entries
            if e["present"]
        }
        for path in sorted(folder_path.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() in known:
                continue
            extras.append(
                {
                    "relative_path": path.relative_to(models_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_of(path),
                }
            )

    return {
        "models_root": str(models_root),
        "entries": entries,
        "unexpected_files": extras,
        "present_count": sum(1 for e in entries if e["present"]),
        "total_count": len(entries),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    models_root = resolve_models_root(args.models_root)
    if models_root is None:
        print(
            json.dumps(
                {
                    "fatal": "no models root found; set KEEP_MODEL_DIR or COMFYUI_ROOT",
                    "searched": list(DEFAULT_MODELS_ROOTS),
                },
                indent=2,
            )
        )
        return 2

    report = scan(models_root)
    text = json.dumps(report, indent=2)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
