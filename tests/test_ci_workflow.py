"""P1.5 - the CI workflow.

A workflow file is not a test, but it *is* code: a workflow without install
steps cannot run at all, and a workflow that runs a different command than the
local gate tests something else than the developer tested.

The checks below therefore assert that

* the workflow is valid YAML and has a ``lint`` and a ``test`` job,
* the lint job installs a **pinned** ruff and runs ``ruff check .``,
* the test job carries the real install steps (CPU torch, numpy, opencv,
  einops, scipy, tomli on 3.10, **PyYAML**) and the matrix 3.10 / 3.12 / 3.13,
* the test job runs exactly the local gate command and leaves ``COMFYUI_ROOT``
  empty (so the environment dependent tests skip themselves),
* ``publish.yml`` is still the untouched registry workflow.

**The module never skips.**  The plan (P1.5) says: "workflow YAML parses via
embedded python's ``yaml`` (available: PyYAML in env? - if absent, parse with
a minimal regex check instead; **assert either way**)".  A module level
``pytest.importorskip("yaml")`` would skip *all* checks on a CI image without
PyYAML - exactly the image where the workflow is broken.  So the file parses
the workflow with PyYAML when it is importable and with a minimal indentation
parser otherwise, and the assertions below always run.  The fallback is
cross-checked against the PyYAML parse whenever PyYAML is available, so it
cannot rot.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

try:  # PyYAML is optional: the checks below must run without it.
    import yaml
except ImportError:  # pragma: no cover - exercised by the CI-env simulation
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"
PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish.yml"

#: The local gate of the plan (section 4).
LOCAL_GATE = 'pytest -q -m "unit or mock_integration or headless"'


# ---------------------------------------------------------------------------
# the fallback parser (no PyYAML)
# ---------------------------------------------------------------------------
_JOB_RE = re.compile(r"^  ([A-Za-z0-9_.-]+):\s*$")
_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+):\s*(.*)$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _block_after(lines: list[str], index: int) -> list[str]:
    """The indented block that follows ``lines[index]``."""
    parent = _indent(lines[index])
    block: list[str] = []
    for line in lines[index + 1 :]:
        if line.strip() and _indent(line) <= parent:
            break
        block.append(line)
    return block


def _dedent(lines: list[str]) -> str:
    """The content of a YAML block scalar, without its indentation."""
    indents = [_indent(line) for line in lines if line.strip()]
    if not indents:
        return ""
    margin = min(indents)
    return "\n".join(line[margin:] if line.strip() else "" for line in lines)


def _scalar(value: str) -> str:
    return value.strip().strip("\"'")


def _inline_list(value: str) -> list[str]:
    """``[a, b]`` -> ``["a", "b"]`` (the quotes are optional)."""
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return [_scalar(value)]
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [_scalar(item) for item in value[1:-1].split(",") if item.strip()]


def _parse_mapping(block: list[str]) -> dict[str, object]:
    """A flat ``key: value`` mapping (values may be inline lists)."""
    mapping: dict[str, object] = {}
    for line in block:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _KEY_RE.match(line)
        if not match:
            continue
        key, raw = match.group(1), match.group(2).strip()
        if raw.startswith("[") and raw.endswith("]"):
            mapping[key] = _inline_list(raw)
        else:
            mapping[key] = _scalar(raw)
    return mapping


def _parse_steps(body: list[str]) -> list[dict[str, str]]:
    """The ``run:`` text of every step of a job."""
    steps: list[dict[str, str]] = []
    index = 0
    while index < len(body):
        line = body[index]
        if not line.lstrip().startswith("- "):
            index += 1
            continue
        step_indent = _indent(line)
        attribute_lines: list[str] = []
        index += 1
        while index < len(body):
            candidate = body[index]
            if candidate.strip() and _indent(candidate) <= step_indent:
                break
            attribute_lines.append(candidate)
            index += 1
        for position, attribute in enumerate(attribute_lines):
            match = re.match(r"^(\s*)run:\s*(.*)$", attribute)
            if not match:
                continue
            inline = match.group(2).strip()
            if inline in {"|", ">", "|-", ">-"}:
                content = _dedent(_block_after(attribute_lines, position))
            else:
                content = inline
            steps.append({"run": content.rstrip()})
            break
    return steps


def _parse_job(body: list[str]) -> dict[str, object]:
    """One job: its ``run:`` steps, its ``env`` and its matrix."""
    env: dict[str, str] = {}
    matrix: dict[str, list[str]] = {}
    for index, line in enumerate(body):
        stripped = line.strip()
        if stripped == "env:":
            for key, value in _parse_mapping(_block_after(body, index)).items():
                env[key] = str(value)
        elif stripped == "matrix:":
            for key, value in _parse_mapping(_block_after(body, index)).items():
                matrix[key] = value if isinstance(value, list) else [str(value)]
    return {"steps": _parse_steps(body), "env": env, "matrix": matrix}


def _fallback_parse(text: str) -> dict[str, object]:
    """A minimal parser for the workflow file (no PyYAML needed).

    It understands the subset the workflow uses: the top-level ``on:``
    triggers, the ``jobs:`` mapping, the per-job ``steps`` with ``run:`` block
    scalars, the per-job ``env:`` mapping and ``strategy.matrix``.
    """
    lines = text.splitlines()
    jobs: dict[str, object] = {}
    triggers: list[str] = []

    in_jobs = False
    job_name: str | None = None
    job_body: list[str] = []

    def close_job() -> None:
        nonlocal job_name, job_body
        if job_name is not None:
            jobs[job_name] = _parse_job(job_body)
        job_name, job_body = None, []

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not in_jobs:
            if stripped == "jobs:":
                in_jobs = True
            elif stripped == "on:":
                for trigger_line in _block_after(lines, index):
                    match = re.match(r"^  ([A-Za-z0-9_.-]+):", trigger_line)
                    if match:
                        triggers.append(match.group(1))
            continue
        if stripped and not line.startswith(" "):
            break  # the next top-level key ends the jobs section
        match = _JOB_RE.match(line)
        if match:
            close_job()
            job_name = match.group(1)
        elif job_name is not None:
            job_body.append(line)
    close_job()

    return {"jobs": jobs, "triggers": sorted(triggers)}


def _yaml_parse(text: str) -> dict[str, object]:
    """The same normalized view, parsed with PyYAML."""
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict), "the workflow must parse into a mapping"

    jobs: dict[str, object] = {}
    for name, job in (loaded.get("jobs") or {}).items():
        steps = []
        for step in job.get("steps") or []:
            if isinstance(step, dict) and "run" in step:
                steps.append({"run": str(step["run"]).rstrip()})
        env = {str(key): str(value) for key, value in (job.get("env") or {}).items()}
        matrix: dict[str, list[str]] = {}
        raw_matrix = (job.get("strategy") or {}).get("matrix") or {}
        for key, value in raw_matrix.items():
            matrix[str(key)] = (
                [str(item) for item in value] if isinstance(value, list) else [str(value)]
            )
        jobs[str(name)] = {"steps": steps, "env": env, "matrix": matrix}

    # PyYAML parses the YAML 1.1 boolean "on" as the key True.
    triggers = sorted(str(key) for key in (loaded.get("on") or loaded.get(True) or {}))
    return {"jobs": jobs, "triggers": triggers}


@pytest.fixture(scope="module")
def workflow():
    """The workflow as a normalized mapping (PyYAML, or the fallback parser)."""
    assert WORKFLOW.is_file(), f"the CI workflow is missing: {WORKFLOW}"
    text = WORKFLOW.read_text(encoding="utf-8")
    if yaml is not None:
        return _yaml_parse(text)
    return _fallback_parse(text)


def _steps(job: dict) -> list[dict]:
    return list(job.get("steps", []))


def _run_text(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(job))


@pytest.mark.unit
def test_workflow_parses_and_has_the_two_jobs(workflow):
    """The file is valid YAML with a lint job and a test job."""
    jobs = workflow["jobs"]
    assert "lint" in jobs, "the workflow needs a lint job"
    assert "test" in jobs, "the workflow needs a test job"
    assert workflow["triggers"], "the workflow needs a trigger"


@pytest.mark.unit
def test_the_fallback_parser_agrees_with_the_yaml_parse():
    """The PyYAML-free fallback must see the same workflow as PyYAML.

    This keeps the fallback honest on every runner that *has* PyYAML (the CI
    job installs it); without the cross-check the fallback would only ever run
    on a machine that is already broken.
    """
    if yaml is None:
        pytest.skip("PyYAML is not installed - the fallback parser is the only one")
    text = WORKFLOW.read_text(encoding="utf-8")
    assert _fallback_parse(text) == _yaml_parse(text)


@pytest.mark.unit
def test_lint_job_installs_a_pinned_ruff_and_checks_the_repository(workflow):
    """The lint job installs ruff (pinned) and runs ``ruff check .``."""
    lint = workflow["jobs"]["lint"]
    text = _run_text(lint)

    assert "ruff==0.8.*" in text, "ruff must be pinned to the 0.8 series"
    assert "pip install" in text
    assert "ruff check ." in text


@pytest.mark.unit
def test_test_job_has_the_python_matrix(workflow):
    """The test job runs on python 3.10, 3.12 and 3.13."""
    matrix = workflow["jobs"]["test"].get("matrix", {})
    versions = [str(version) for version in matrix.get("python-version", [])]
    assert versions == ["3.10", "3.12", "3.13"], (
        f"unexpected python matrix: {versions}"
    )


@pytest.mark.unit
def test_test_job_installs_the_real_dependencies(workflow):
    """The test job installs what the package and the suite actually need."""
    text = _run_text(workflow["jobs"]["test"])

    assert "--index-url https://download.pytorch.org/whl/cpu" in text, (
        "torch must be installed from the CPU index"
    )
    assert "torchvision" in text, "the package imports torchvision"
    assert "opencv-python-headless" in text
    assert "einops" in text
    assert "scipy" in text
    assert "Pillow" in text
    assert "tqdm" in text
    assert "pytest" in text
    # The vendored tree imports yaml eagerly
    # (modules/deps/basicsr/utils/options.py:5), so the unit tier dies with
    # "No module named 'yaml'" when PyYAML is missing.
    assert "pyyaml" in text.lower(), (
        "PyYAML must be installed: the vendored tree imports it eagerly"
    )
    assert "numpy<2.3" in text, "numpy must be pinned for the 3.10 job"
    assert "tomli" in text, "tomli backs tomllib on python 3.10"
    assert '"3.10"' in text, "the 3.10-only installs must be conditional"


@pytest.mark.unit
def test_test_job_runs_the_local_gate_command(workflow):
    """The CI command is exactly the local gate command."""
    run_commands = [
        str(step.get("run", "")).strip()
        for step in _steps(workflow["jobs"]["test"])
        if "pytest" in str(step.get("run", "")) and "pip install" not in str(step.get("run", ""))
    ]
    assert run_commands, "the test job must run pytest"

    command = run_commands[0]
    assert command.endswith(LOCAL_GATE), (
        f"the CI command {command!r} does not end with the local gate {LOCAL_GATE!r}"
    )
    assert command.startswith("python -m pytest"), (
        "the CI must run pytest through the interpreter of the job"
    )


@pytest.mark.unit
def test_test_job_leaves_comfyui_root_empty(workflow):
    """``COMFYUI_ROOT`` is empty so the environment dependent tests skip."""
    env = workflow["jobs"]["test"].get("env", {})
    assert "COMFYUI_ROOT" in env, "the test job must define COMFYUI_ROOT"
    assert str(env["COMFYUI_ROOT"]) == "", (
        "COMFYUI_ROOT must stay empty: the gate runs without environment variables"
    )


@pytest.mark.unit
def test_publish_workflow_is_untouched():
    """``publish.yml`` is still the registry publish workflow."""
    assert PUBLISH_WORKFLOW.is_file()
    text = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    assert "Comfy-Org/publish-node-action" in text
    assert "tests.yml" not in text, "publish.yml must not depend on the test workflow"
