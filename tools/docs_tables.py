"""The generated tables of the README (P6.2).

The **weights table** of the README is *generated* from the weight manifest
(:data:`src.keep_model_configs.WEIGHT_MANIFEST`) - the manifest is the single
source of truth of the URLs, the hashes, the sizes and the destinations, so
the table must never be written by hand.

The module has no ComfyUI dependency: it only formats the entries it gets.

    python -c "from tools.docs_tables import weights_table; ..."

``tests/test_docs_claims.py`` asserts that the README contains exactly the
table this module builds - a change of the manifest therefore fails the test
instead of silently making the README wrong.
"""

from __future__ import annotations

from typing import Iterable, Protocol


class _Entry(Protocol):
    """The fields of a manifest entry that the table uses."""

    name: str
    sha256: str | None
    dest_folder: str
    subdir: str
    required: bool
    available: bool


#: The file extensions that make a backticked token a **path claim** of the
#: repository.  The weight extensions are not part of it on purpose: the
#: weights live in the model folders of ComfyUI, not in the repository.
REPO_FILE_EXTENSIONS = (
    ".cfg",
    ".json",
    ".md",
    ".png",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
)

#: The path roots that are **outside** the repository (they are paths of the
#: ComfyUI installation).
EXTERNAL_PATH_ROOTS = ("ComfyUI", "custom_nodes", "models")


def format_size(size: int | None) -> str:
    """A human readable size (the manifest stores bytes)."""
    if size is None:
        return "—"
    return f"{size / (1024 * 1024):.1f} MiB"


def destination_of(entry: _Entry) -> str:
    """The on-disk destination of an entry (inside the ComfyUI ``models``)."""
    parts = ["models", entry.dest_folder]
    if entry.subdir:
        parts.append(entry.subdir)
    parts.append(entry.name)
    return "/".join(parts)


def note_of(entry: _Entry) -> str:
    """The note of an entry - derived from the flags of the manifest."""
    if not entry.available:
        return (
            "**not published upstream (HTTP 404)** — the file has to be "
            "placed manually"
        )
    if not entry.required:
        return "optional (defensive) — only used when the checkpoint has no flow weights"
    return "required"


def weights_rows(manifest: Iterable[_Entry]) -> list[str]:
    """The rows of the weights table (one markdown line per entry)."""
    rows = []
    for entry in manifest:
        sha256 = entry.sha256 if entry.sha256 else "—"
        rows.append(
            f"| `{entry.name}` | {format_size(getattr(entry, 'size', None))} "
            f"| `{sha256}` | `{destination_of(entry)}` | {note_of(entry)} |"
        )
    return rows


def weights_table(manifest: Iterable[_Entry]) -> str:
    """The whole weights table of the README (header + rows)."""
    lines = [
        "| File | Size | SHA-256 | Destination | Notes |",
        "|:---|---:|:---|:---|:---|",
    ]
    lines.extend(weights_rows(manifest))
    return "\n".join(lines)
