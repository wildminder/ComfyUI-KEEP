"""The generated tables of the README (P6.2).

The **weights table** of the README is *generated* from the weight manifest
(:data:`src.keep_model_configs.WEIGHT_MANIFEST`) - the manifest is the single
source of truth of the URLs, the hashes, the sizes and the destinations, so
the table must never be written by hand.

The **per-node parameter tables** are generated the same way: the V3 schema
of a node is the source of truth of the widget names, the defaults, the
ranges and the widget order, so :func:`documented_parameters` renders what
the README has to say about a node.

The module has no ComfyUI dependency: it only formats the entries it gets.

    python -c "from tools.docs_tables import weights_table; ..."

``tests/test_docs_claims.py`` asserts that the README contains exactly the
tables this module builds - a change of the manifest or of a schema therefore
fails the test instead of silently making the README wrong.

The module owns the **path rules** of the README as well
(:data:`REPO_FILE_EXTENSIONS` and :data:`EXTERNAL_PATH_ROOTS`): the test
imports them from here, so the checker and the generator cannot drift apart.
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

#: The socket types that are **widgets** in the V3 schema: a widget carries a
#: default value and is saved in the ``widgets_values`` of a workflow.
WIDGET_SOCKET_TYPES = frozenset(
    {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "MULTICOMBO"}
)

#: The dash the README writes inside a range (``0.5 – 4.0``).
RANGE_DASH = "–"


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


def _number(value) -> str:
    """A number as the README writes it (``1.0``, ``20``, ``0.5``)."""
    return str(value)


def parameter_default_cell(node_input) -> str:
    """The README cell of the default value of a widget.

    An optional socket has no default: its cell is ``*(optional)*``.
    """
    kind = node_input.get_io_type()
    if kind not in WIDGET_SOCKET_TYPES:
        return "*(optional)*"
    if kind == "BOOLEAN":
        return f"`{'true' if node_input.default else 'false'}`"
    if kind in {"FLOAT", "INT"}:
        cell = f"`{_number(node_input.default)}`"
        if node_input.min is not None or node_input.max is not None:
            low = _number(node_input.min) if node_input.min is not None else ""
            high = _number(node_input.max) if node_input.max is not None else ""
            span = f"{low} {RANGE_DASH} {high}" if low and high else (low or high)
            step = node_input.step
            if step is not None and step != 1:
                span += f", step {_number(step)}"
            cell += f" ({span})"
        return cell
    if kind == "MULTICOMBO":
        return f"`{', '.join(str(value) for value in node_input.default or [])}`"
    return f"`{node_input.default}`"


def documented_parameters(schema) -> list[tuple[str, str]]:
    """The parameters the README documents for a node: ``(name, default)``.

    Every **widget** of the schema is documented and every **optional**
    input (an optional socket has no default - it is marked as optional).
    The order is the order of the schema: the widget order is a
    compatibility guarantee of the pack.
    """
    parameters = []
    for node_input in schema.inputs:
        if node_input.get_io_type() not in WIDGET_SOCKET_TYPES and not node_input.optional:
            continue
        parameters.append((node_input.id, parameter_default_cell(node_input)))
    return parameters
