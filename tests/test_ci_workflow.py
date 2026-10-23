"""P1.5 - the CI workflow.

A workflow file is not a test, but it *is* code: a workflow without install
steps cannot run at all, and a workflow that runs a different command than the
local gate tests something else than the developer tested.

The checks below therefore assert that

* the workflow is valid YAML and has a ``lint`` and a ``test`` job,
* the lint job installs a **pinned** ruff and runs ``ruff check .``,
* the test job carries the real install steps (CPU torch, numpy, opencv,
  einops, scipy, tomli on 3.10) and the matrix 3.10 / 3.12 / 3.13,
* the test job runs exactly the local gate command and leaves ``COMFYUI_ROOT``
  empty (so the environment dependent tests skip themselves),
* ``publish.yml`` is still the untouched registry workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is required to parse the workflow")

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"
PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish.yml"

#: The local gate of the plan (section 4).
LOCAL_GATE = 'pytest -q -m "unit or mock_integration or headless"'


@pytest.fixture(scope="module")
def workflow():
    assert WORKFLOW.is_file(), f"the CI workflow is missing: {WORKFLOW}"
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), "the workflow must parse into a mapping"
    return loaded


def _steps(job: dict) -> list[dict]:
    return list(job.get("steps", []))


def _run_text(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(job))


@pytest.mark.unit
def test_workflow_parses_and_has_the_two_jobs(workflow):
    """The file is valid YAML with a lint job and a test job."""
    jobs = workflow.get("jobs", {})
    assert "lint" in jobs, "the workflow needs a lint job"
    assert "test" in jobs, "the workflow needs a test job"
    # PyYAML parses the YAML 1.1 boolean "on" as the key True.
    triggers = workflow.get("on", workflow.get(True))
    assert triggers, "the workflow needs a trigger"


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
    matrix = workflow["jobs"]["test"].get("strategy", {}).get("matrix", {})
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
