"""Render a preview image of a ComfyUI workflow json (P6.1).

The shipped example workflows carry a preview image next to the workflow
(``keep-workflow.json`` -> ``keep-workflow.png``).  The **sequence** example
of P6.1 has no ComfyUI screenshot - it was written by hand - so its preview is
*rendered* from the workflow graph itself by this script:

    python tools/render_workflow_preview.py \\
        example_workflows/keep-sequence-workflow.json \\
        example_workflows/keep-sequence-workflow.png

The rendered image is a **diagram of the graph** (nodes, titles, widget values
and the links between them) - not a screenshot of the ComfyUI frontend, and it
does not pretend to be one.  The caption of the image says so.

The script only needs Pillow (a runtime dependency of the pack) and the
standard library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (24, 24, 32)
GRID = (34, 34, 44)
BOX_FILL = (44, 46, 62)
BOX_OUTLINE = (108, 142, 191)
BOX_OUTLINE_OUTPUT = (126, 191, 133)
TITLE_COLOR = (235, 235, 245)
TYPE_COLOR = (159, 184, 216)
WIDGET_COLOR = (198, 198, 210)
LINK_COLOR = (150, 160, 190)
CAPTION_COLOR = (150, 152, 165)

MARGIN = 40
HEADER = 96
TARGET_WIDTH = 1600


def _font(size: int):
    """The scalable default font of Pillow (no font file needed)."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow < 10.1
        return ImageFont.load_default()


def _node_title(node: dict) -> str:
    return str(node.get("title") or node.get("type"))


def _fit_text(draw, text: str, font, max_width: float) -> str:
    """Shorten ``text`` until it fits into ``max_width`` pixels."""
    if draw.textlength(text, font=font) <= max_width:
        return text
    shortened = text
    while shortened and draw.textlength(shortened + "…", font=font) > max_width:
        shortened = shortened[:-1]
    return shortened + "…" if shortened else ""


def _widget_summary(node: dict) -> list[str]:
    """The widget values of a node as short strings."""
    values = node.get("widgets_values")
    if values is None:
        return []
    if isinstance(values, dict):
        values = list(values.values())
    summary = []
    for value in values:
        if isinstance(value, bool):
            summary.append("true" if value else "false")
        elif value is None:
            summary.append("")
        elif isinstance(value, (int, float)):
            summary.append(str(value))
        else:
            text = str(value)
            summary.append(text if len(text) <= 28 else text[:27] + "…")
    return summary


def _anchor_out(node: dict, index: int) -> tuple[float, float]:
    pos = node.get("pos", [0, 0])
    size = node.get("size", [200, 80])
    return float(pos[0]) + float(size[0]), float(pos[1]) + 26.0 + 18.0 * index


def _anchor_in(node: dict, index: int) -> tuple[float, float]:
    pos = node.get("pos", [0, 0])
    return float(pos[0]), float(pos[1]) + 26.0 + 18.0 * index


def render(workflow_path: Path, output_path: Path) -> Path:
    """Render the workflow graph to a PNG file."""
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    nodes = workflow.get("nodes", [])
    by_id = {node["id"]: node for node in nodes}

    max_x = max((float(n.get("pos", [0, 0])[0]) + float(n.get("size", [0, 0])[0]) for n in nodes), default=0.0)
    max_y = max((float(n.get("pos", [0, 0])[1]) + float(n.get("size", [0, 0])[1]) for n in nodes), default=0.0)

    scale = TARGET_WIDTH / max(max_x + MARGIN * 2, 1.0)
    width = int((max_x + MARGIN * 2) * scale)
    height = int((max_y + MARGIN * 2 + HEADER) * scale)

    image = Image.new("RGB", (max(width, 640), max(height, 360)), BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = _font(30)
    type_font = _font(19)
    widget_font = _font(16)
    caption_font = _font(15)

    def point(x: float, y: float) -> tuple[int, int]:
        return int((x + MARGIN) * scale), int((y + MARGIN + HEADER) * scale)

    # the grid
    step = 40
    x = 0.0
    while x < max_x + MARGIN * 2:
        px = int(x * scale)
        draw.line([(px, 0), (px, image.height)], fill=GRID, width=1)
        x += step
    y = 0.0
    while y < max_y + MARGIN * 2 + HEADER:
        py = int(y * scale)
        draw.line([(0, py), (image.width, py)], fill=GRID, width=1)
        y += step

    # the links (under the boxes)
    for link in workflow.get("links", []):
        link_id, source_id, source_slot, target_id, target_slot, _kind = link
        source = by_id.get(source_id)
        target = by_id.get(target_id)
        if source is None or target is None:
            continue
        x1, y1 = point(*_anchor_out(source, int(source_slot)))
        x2, y2 = point(*_anchor_in(target, int(target_slot)))
        mid = (x1 + x2) // 2
        draw.line([(x1, y1), (mid, y1), (mid, y2), (x2, y2)], fill=LINK_COLOR, width=2)

    # the nodes
    for node in nodes:
        pos = node.get("pos", [0, 0])
        size = node.get("size", [200, 80])
        x0, y0 = point(float(pos[0]), float(pos[1]))
        x1, y1 = point(float(pos[0]) + float(size[0]), float(pos[1]) + float(size[1]))
        is_output = node.get("type") in {"SaveImage", "PreviewImage", "VHS_VideoCombine"}
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=8,
            fill=BOX_FILL,
            outline=BOX_OUTLINE_OUTPUT if is_output else BOX_OUTLINE,
            width=2,
        )
        inner_width = max(x1 - x0 - 24, 20)
        draw.text((x0 + 12, y0 + 8), _fit_text(draw, _node_title(node), title_font, inner_width), font=title_font, fill=TITLE_COLOR)
        draw.text((x0 + 12, y0 + 42), _fit_text(draw, str(node.get("type")), type_font, inner_width), font=type_font, fill=TYPE_COLOR)
        for index, value in enumerate(_widget_summary(node)):
            draw.text((x0 + 12, y0 + 68 + 20 * index), _fit_text(draw, value, widget_font, inner_width), font=widget_font, fill=WIDGET_COLOR)

    # the header
    draw.text((MARGIN, 18), workflow_path.stem, font=title_font, fill=TITLE_COLOR)
    draw.text(
        (MARGIN, 56),
        "Graph diagram rendered from the workflow json - not a ComfyUI screenshot.",
        font=caption_font,
        fill=CAPTION_COLOR,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path, help="the workflow json to render")
    parser.add_argument("output", type=Path, help="the png to write")
    args = parser.parse_args(argv)
    written = render(args.workflow, args.output)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
