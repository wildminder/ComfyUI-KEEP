"""P1.5 - simulate the CI environment and run the gate in it.

The CI job installs a **fixed** list of packages; everything else that happens
to be installed on a developer machine must not be needed by the suite.  A
package that the suite needs but the CI install list omits turns the CI job
red on its first run (this is how PyYAML was missed: the vendored tree imports
``yaml`` eagerly at ``modules/deps/basicsr/utils/options.py:5``).

This tool reproduces the CI environment - it blocks the packages the CI
install list does **not** contain - and runs the gate command of the workflow
in it.  The block list is the "dev machine only" set of the P1 audit; add more
with ``--block``.

Usage (embedded python, from the repository root):

    C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe tools/ci_env_sim.py
    C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/python_embeded/python.exe tools/ci_env_sim.py --block yaml

Exit code: the pytest exit code of the gate run.
"""

from __future__ import annotations

import argparse
import sys
from importlib.abc import MetaPathFinder

#: The gate command of ``.github/workflows/tests.yml`` (and of the plan).
GATE = "-m"
GATE_MARKERS = "unit or mock_integration or headless"

#: Packages that are installed on a development machine but **not** by the CI
#: install step (``.github/workflows/tests.yml``).  The suite must not need
#: them - if it does, the CI install list is incomplete.
DEV_ONLY_PACKAGES = (
    "addict",
    "cupy",
    "diffusers",
    "ffmpeg",
    "future",
    "lmdb",
    "mmcv",
    "skimage",
    "yapf",
)


class _BlockingFinder(MetaPathFinder):
    """Makes the blocked top-level modules unimportable."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = frozenset(names)

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001, D102
        if fullname.split(".")[0] in self.names:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block",
        action="append",
        default=[],
        metavar="MODULE",
        help="an additional top-level module to block (repeatable)",
    )
    parser.add_argument(
        "--list-blocked",
        action="store_true",
        help="print the blocked modules and exit",
    )
    args = parser.parse_args(argv)

    blocked = DEV_ONLY_PACKAGES + tuple(args.block)
    if args.list_blocked:
        for name in blocked:
            print(name)
        return 0

    sys.meta_path.insert(0, _BlockingFinder(blocked))
    print(f"[ci-env-sim] blocked modules: {', '.join(blocked)}")

    import pytest  # noqa: PLC0415 - imported after the blocker is installed

    return pytest.main(["-q", GATE, GATE_MARKERS])


if __name__ == "__main__":
    raise SystemExit(main())
