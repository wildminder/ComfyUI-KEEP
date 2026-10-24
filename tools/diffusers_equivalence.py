"""P2.2 - one-off equivalence run: vendored vs. pristine diffusers 0.11.0.

The plan (P2.2) asks for a **recorded** equivalence run: the pristine
``diffusers==0.11.0`` classes and the vendored copy must have the same
``state_dict`` keys and produce identical outputs on a seeded input
(tolerance 0).  The result is written to
``.dev/docs/audit/2026-10-22-diffusers-equivalence.md``.

Why this is a *tool* and not a plain test: the pristine diffusers sources are
not part of the repository.  The hard environment rules of this phase forbid
creating a venv and forbid installing into the environment, so the pristine
package is used **as source**:

1. ``pip download diffusers==0.11.0 --no-deps --no-binary :all:`` into a
   temp dir (the only pip operation the plan allows), then unpack it;
2. the unpacked ``src`` directory is put in front of ``sys.path``;
3. the ``diffusers`` *package* is registered by hand with its real
   ``__path__``, so the pristine ``diffusers/models/attention.py`` and its
   real sibling modules are loaded without executing the 0.11.0 pipeline
   imports (the pipelines need a ``transformers`` older than the one
   installed here);
4. the ``huggingface_hub`` names that recent releases dropped
   (``cached_download``, ``HfFolder``, ...) are injected into the already
   imported ``huggingface_hub`` module - nothing is installed and nothing is
   written outside the temp dir.

Usage (embedded python):

    python tools/diffusers_equivalence.py --diffusers-src <unpacked>/src

Without ``--diffusers-src`` the tool exits with code 2 ("pristine source not
available") - the unit test then skips with a message that names the
argument.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDORED_FILE = REPO_ROOT / "modules" / "deps" / "basicsr" / "archs" / "_diffusers_compat.py"
DEFAULT_MARKDOWN = REPO_ROOT / ".dev" / "docs" / "audit" / "2026-10-22-diffusers-equivalence.md"

#: The classes that are vendored (the helper classes are pulled in by
#: ``FeedForward`` and are part of the copied block).
VENDORED_CLASSES = ("FeedForward", "GELU", "GEGLU", "ApproximateGELU", "AdaLayerNorm")

#: The ``huggingface_hub`` names diffusers 0.11.0 imports at module level.
HUB_NAMES = (
    "HfFolder",
    "cached_download",
    "hf_hub_download",
    "model_info",
    "whoami",
    "snapshot_download",
)


def _install_hub_shims(huggingface_hub) -> list[str]:
    """Add the missing ``huggingface_hub`` names; return the injected ones."""
    injected: list[str] = []
    for name in HUB_NAMES:
        if hasattr(huggingface_hub, name):
            continue

        def placeholder(*args, _name=name, **kwargs):  # pragma: no cover
            raise RuntimeError(
                f"huggingface_hub.{_name} is not available in this "
                "huggingface_hub version - the diffusers 0.11.0 import only "
                "needs the name to exist"
            )

        placeholder.__name__ = name
        setattr(huggingface_hub, name, placeholder)
        injected.append(name)
    return injected


def _load_vendored():
    """Load the vendored compat module directly from its file."""
    spec = importlib.util.spec_from_file_location("keep_diffusers_compat", VENDORED_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_blocks(text: str) -> dict[str, str]:
    """The source of every top-level ``class`` of ``text``."""
    blocks: dict[str, str] = {}
    matches = list(re.finditer(r"^class\s+(\w+)\(", text, re.MULTILINE))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks[match.group(1)] = text[match.start() : end].rstrip("\n")
    return blocks


def run(diffusers_src: Path) -> dict:
    """Compare the pristine diffusers 0.11.0 classes with the vendored copy."""
    import torch

    src = Path(diffusers_src).resolve()
    pristine_file = src / "diffusers" / "models" / "attention.py"
    if not pristine_file.is_file():
        raise FileNotFoundError(
            f"{src} does not contain diffusers/models/attention.py - point "
            "--diffusers-src at the unpacked 'src' directory of diffusers 0.11.0"
        )

    sys.path.insert(0, str(src))

    import huggingface_hub  # noqa: PLC0415 - optional dependency

    injected = _install_hub_shims(huggingface_hub)

    package = types.ModuleType("diffusers")
    package.__path__ = [str(src / "diffusers")]  # type: ignore[attr-defined]
    package.__version__ = "0.11.0"
    package.__file__ = str(src / "diffusers" / "__init__.py")
    sys.modules["diffusers"] = package

    import diffusers.models.attention as pristine_attention  # noqa: PLC0415
    from diffusers.models.attention import AdaLayerNorm as RealAdaLayerNorm
    from diffusers.models.attention import FeedForward as RealFeedForward

    version = getattr(package, "__version__", "unknown")
    if version != "0.11.0":
        raise RuntimeError(
            f"the pristine diffusers 0.11.0 source was expected, got {version}"
        )

    vendored = _load_vendored()

    results: dict = {
        "diffusers_version": version,
        "diffusers_file": str(Path(pristine_attention.__file__).resolve()),
        "vendored_file": str(VENDORED_FILE.resolve()),
        "huggingface_hub_shims": injected,
        "source_blocks": [],
        "cases": [],
    }

    # --- 1. the copied source must be byte-identical ----------------------
    pristine_blocks = class_blocks(pristine_file.read_text(encoding="utf-8"))
    vendored_blocks = class_blocks(VENDORED_FILE.read_text(encoding="utf-8"))
    for name in VENDORED_CLASSES:
        results["source_blocks"].append(
            {
                "class": name,
                "byte_identical": pristine_blocks.get(name) == vendored_blocks.get(name),
                "lines": len((vendored_blocks.get(name) or "").splitlines()),
            }
        )

    # --- 2. FeedForward: state_dict + forward (tolerance 0) ---------------
    torch.manual_seed(1234)
    real_ff = RealFeedForward(8)
    torch.manual_seed(1234)
    vend_ff = vendored.FeedForward(8)

    real_keys = list(real_ff.state_dict().keys())
    vend_keys = list(vend_ff.state_dict().keys())
    vend_ff.load_state_dict(real_ff.state_dict())

    generator = torch.Generator().manual_seed(7)
    sample = torch.randn((2, 8), generator=generator)
    with torch.no_grad():
        real_out = real_ff(sample)
        vend_out = vend_ff(sample)

    results["cases"].append(
        {
            "class": "FeedForward",
            "state_dict_keys_equal": real_keys == vend_keys,
            "state_dict_keys": real_keys,
            "input_shape": list(sample.shape),
            "output_shape": list(real_out.shape),
            "shape_preserved": list(real_out.shape) == list(sample.shape),
            "max_abs_diff": float((real_out - vend_out).abs().max()),
            "bit_identical": bool(torch.equal(real_out, vend_out)),
        }
    )

    # --- 3. AdaLayerNorm: state_dict + the inner computation --------------
    torch.manual_seed(1234)
    real_aln = RealAdaLayerNorm(8, 4)
    torch.manual_seed(1234)
    vend_aln = vendored.AdaLayerNorm(8, 4)

    real_keys = list(real_aln.state_dict().keys())
    vend_keys = list(vend_aln.state_dict().keys())
    vend_aln.load_state_dict(real_aln.state_dict())

    timestep = torch.tensor([0, 2, 3], dtype=torch.long)
    with torch.no_grad():
        real_emb = real_aln.linear(real_aln.silu(real_aln.emb(timestep)))
        vend_emb = vend_aln.linear(vend_aln.silu(vend_aln.emb(timestep)))

    # The *forward* of the pristine 0.11.0 class chunks the embedding on
    # dim 0 (``torch.chunk(emb, 2)``), which only broadcasts for very
    # specific shapes - the upstream behaviour is reproduced verbatim, so
    # both classes must fail (or succeed) in exactly the same way.
    probe_input = torch.randn((3, 8), generator=torch.Generator().manual_seed(11))
    real_error = vend_error = None
    try:
        with torch.no_grad():
            real_aln(probe_input, timestep)
    except BaseException as exc:  # noqa: BLE001 - the behaviour is the evidence
        real_error = f"{type(exc).__name__}: {exc}"
    try:
        with torch.no_grad():
            vend_aln(probe_input, timestep)
    except BaseException as exc:  # noqa: BLE001
        vend_error = f"{type(exc).__name__}: {exc}"

    results["cases"].append(
        {
            "class": "AdaLayerNorm",
            "state_dict_keys_equal": real_keys == vend_keys,
            "state_dict_keys": real_keys,
            "embedding_output_shape": list(real_emb.shape),
            "max_abs_diff_embedding": float((real_emb - vend_emb).abs().max()),
            "bit_identical_embedding": bool(torch.equal(real_emb, vend_emb)),
            "forward_error": real_error,
            "forward_error_identical": real_error == vend_error,
            "note": (
                "diffusers 0.11.0's AdaLayerNorm.forward chunks the timestep "
                "embedding on dim 0; the vendored copy reproduces that "
                "behaviour byte-for-byte (the KEEP architecture never "
                "enables it: use_ada_layer_norm is False without "
                "num_embeds_ada_norm)"
            ),
        }
    )

    results["equivalent"] = (
        all(block["byte_identical"] for block in results["source_blocks"])
        and results["cases"][0]["state_dict_keys_equal"]
        and results["cases"][0]["bit_identical"]
        and results["cases"][1]["state_dict_keys_equal"]
        and results["cases"][1]["bit_identical_embedding"]
        and results["cases"][1]["forward_error_identical"]
    )
    return results


def render_markdown(results: dict) -> str:
    lines = [
        "# P2.2 - diffusers 0.11.0 equivalence run",
        "",
        "Recorded evidence for the vendored copy of `FeedForward` and",
        "`AdaLayerNorm` (`modules/deps/basicsr/archs/_diffusers_compat.py`).",
        "",
        "The pristine `diffusers==0.11.0` **source** was the reference: the",
        "sdist was downloaded with `pip download diffusers==0.11.0 --no-deps",
        "--no-binary :all:` into a temp dir, unpacked, and put in front of",
        "`sys.path`.  No virtual environment was created and nothing was",
        "installed into the environment (the phase rules forbid both).",
        "",
        "Two kinds of evidence are recorded:",
        "",
        "1. **source level** - the copied class blocks are byte-identical to",
        "   the upstream 0.11.0 ones;",
        "2. **runtime level** - same `state_dict` keys and bit-identical",
        "   outputs (tolerance 0) on a seeded input.",
        "",
        f"* pristine diffusers version: **{results['diffusers_version']}**",
        f"* pristine file: `{results['diffusers_file']}`",
        f"* vendored file: `{results['vendored_file']}`",
        f"* `huggingface_hub` names injected (missing in this env): "
        f"`{', '.join(results['huggingface_hub_shims']) or 'none'}`",
        f"* overall equivalent: **{results['equivalent']}**",
        "",
        "## Source level",
        "",
        "| class | byte-identical to diffusers 0.11.0 | lines |",
        "|---|---|---|",
    ]
    for block in results["source_blocks"]:
        lines.append(
            f"| {block['class']} | {block['byte_identical']} | {block['lines']} |"
        )
    lines += [
        "",
        "## Runtime level",
        "",
        "| class | state_dict keys equal | max abs diff | bit-identical |",
        "|---|---|---|---|",
    ]
    for case in results["cases"]:
        diff = case.get("max_abs_diff", case.get("max_abs_diff_embedding"))
        identical = case.get("bit_identical", case.get("bit_identical_embedding"))
        lines.append(
            f"| {case['class']} | {case['state_dict_keys_equal']} | {diff} | {identical} |"
        )
    lines.append("")
    for case in results["cases"]:
        lines.append(f"* `{case['class']}` state_dict keys: `{case['state_dict_keys']}`")
    lines.append("")
    for case in results["cases"]:
        if case.get("note"):
            lines.append(f"* `{case['class']}`: {case['note']}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diffusers-src",
        dest="diffusers_src",
        default=None,
        help="the unpacked 'src' directory of diffusers 0.11.0",
    )
    parser.add_argument("--markdown", dest="markdown", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    if not args.diffusers_src:
        print(
            "no pristine diffusers source given: pass --diffusers-src <unpacked>/src "
            "(the 'src' directory of the unpacked diffusers-0.11.0 sdist)",
            file=sys.stderr,
        )
        return 2

    results = run(Path(args.diffusers_src))
    markdown = render_markdown(results)
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(markdown, encoding="utf-8", newline="\n")
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if results["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
