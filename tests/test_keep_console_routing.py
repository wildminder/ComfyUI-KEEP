"""The console routing of the pack - every console write reaches the logger.

The vendored facelib writes to the console with ``print()``.  Those writes
bypass the console of ComfyUI: they are untagged and they never reach the
log file of the host.  ``src/keep_console.py`` routes them to the logger of
the pack:

* the writes of the **owning thread** become log records - one record per
  complete line - and the tag filter adds the ``[KEEP]`` tag;
* the writes of **another thread** are forwarded to the original stream
  untouched;
* the streams are restored when the block ends - also after an exception;
* the output of ComfyUI is not swallowed: its root handler holds the stream
  object it was built with and writes straight through the swap.

The file also carries the guards of the **tqdm removal**: the processor must
not import ``tqdm`` any more - the progress of the node is reported through
the ComfyUI execution API (``src/keep_progress.py``).
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import io
import logging
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


@pytest.fixture(scope="module")
def console_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_console")


@pytest.fixture(scope="module")
def logger_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_logger")


class _Capture(logging.Handler):
    """A handler that keeps the records it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _records(target: logging.Logger):
    """The records the logger receives while the block runs."""
    handler = _Capture()
    target.addHandler(handler)
    try:
        yield handler.records
    finally:
        target.removeHandler(handler)


def _pack_logger(logger_module, name: str) -> logging.Logger:
    """A **dedicated** pack logger - never the global one.

    The name carries no dot on purpose: a dotted name would make the logger
    a child of ``ComfyUI-KEEP`` and the records would travel through the
    global pack logger.
    """
    return logger_module.configure_logger(logging.getLogger(name), comfyui_present=True)


def _messages(records) -> list[str]:
    """The messages of the records."""
    return [record.getMessage() for record in records]


# ---------------------------------------------------------------------------
# the capture
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_print_inside_the_block_becomes_a_tagged_record(
    console_module, logger_module
):
    """The console write of the owning thread reaches the logger."""
    logger = _pack_logger(logger_module, "KEEP-test-console-print")

    with _records(logger) as records:
        with console_module.capture_console_to_logger(logger):
            print("hello from the vendored tree")

    assert _messages(records) == ["[KEEP] hello from the vendored tree"]


@pytest.mark.unit
def test_the_default_logger_is_the_logger_of_the_pack(console_module, logger_module):
    """Without an argument the pack routes to the logger of the pack."""
    assert console_module.logger is logger_module.logger

    with _records(logger_module.logger) as records:
        with console_module.capture_console_to_logger():
            print("hello")

    assert _messages(records) == ["[KEEP] hello"]


@pytest.mark.unit
def test_a_partial_write_is_buffered_until_the_line_is_complete(
    console_module, logger_module
):
    """One record per **complete** line - a partial line waits for its newline."""
    logger = _pack_logger(logger_module, "KEEP-test-console-partial")

    with _records(logger) as records:
        with console_module.capture_console_to_logger(logger):
            sys.stdout.write("first ")
            assert _messages(records) == [], "an incomplete line must not be logged"
            sys.stdout.write("line\n")
            sys.stdout.write("second line\n")

    assert _messages(records) == ["[KEEP] first line", "[KEEP] second line"]


@pytest.mark.unit
def test_a_line_without_a_newline_is_flushed_at_the_end_of_the_block(
    console_module, logger_module
):
    """The output of a ``print(..., end="")`` is not lost."""
    logger = _pack_logger(logger_module, "KEEP-test-console-flush")

    with _records(logger) as records:
        with console_module.capture_console_to_logger(logger):
            sys.stdout.write("no newline")

    assert _messages(records) == ["[KEEP] no newline"]


@pytest.mark.unit
def test_the_stderr_of_the_owning_thread_is_captured_too(
    console_module, logger_module
):
    """``sys.stderr`` is captured as well - as a warning."""
    logger = _pack_logger(logger_module, "KEEP-test-console-stderr")

    with _records(logger) as records:
        with console_module.capture_console_to_logger(logger):
            print("on stderr", file=sys.stderr)

    assert len(records) == 1
    assert records[0].getMessage() == "[KEEP] on stderr"
    assert records[0].levelno == logging.WARNING


# ---------------------------------------------------------------------------
# the thread scope
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_write_of_another_thread_is_not_captured(
    console_module, logger_module, monkeypatch
):
    """Another thread keeps the console - its write is forwarded untouched."""
    logger = _pack_logger(logger_module, "KEEP-test-console-thread")
    original = io.StringIO()
    monkeypatch.setattr(sys, "stdout", original)

    inside = threading.Event()
    done = threading.Event()

    def _worker():
        inside.wait(5)
        print("from the worker")
        done.set()

    with _records(logger) as records:
        with console_module.capture_console_to_logger(logger):
            worker = threading.Thread(target=_worker)
            worker.start()
            inside.set()
            assert done.wait(5), "the worker did not finish"
            worker.join(5)

    assert _messages(records) == [], "the write of the worker must not be captured"
    assert original.getvalue() == "from the worker\n"


@pytest.mark.unit
def test_two_threads_capture_into_their_own_logger(
    console_module, logger_module
):
    """Two capture blocks may run at the same time - each one owns its logger."""
    logger_a = _pack_logger(logger_module, "KEEP-test-console-parallel-a")
    logger_b = _pack_logger(logger_module, "KEEP-test-console-parallel-b")

    inside_a = threading.Event()
    inside_b = threading.Event()
    errors: list[BaseException] = []

    def _run(name, target, mine, other):
        try:
            with console_module.capture_console_to_logger(target):
                mine.set()
                other.wait(5)
                print(f"from {name}")
        except BaseException as exc:  # noqa: BLE001 - the failure is asserted below
            errors.append(exc)

    with _records(logger_a) as records_a, _records(logger_b) as records_b:
        thread_a = threading.Thread(target=_run, args=("a", logger_a, inside_a, inside_b))
        thread_a.start()
        assert inside_a.wait(5), "the first thread did not enter the block"
        thread_b = threading.Thread(target=_run, args=("b", logger_b, inside_b, inside_a))
        thread_b.start()
        thread_a.join(5)
        thread_b.join(5)

    assert errors == []
    assert _messages(records_a) == ["[KEEP] from a"]
    assert _messages(records_b) == ["[KEEP] from b"]


# ---------------------------------------------------------------------------
# the restoration
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_streams_are_restored_after_the_block(console_module, logger_module):
    """The block leaves the streams of the process as it found them."""
    logger = _pack_logger(logger_module, "KEEP-test-console-restore")
    before = (sys.stdout, sys.stderr)

    with console_module.capture_console_to_logger(logger):
        assert (sys.stdout, sys.stderr) != before

    assert (sys.stdout, sys.stderr) == before


@pytest.mark.unit
def test_the_streams_are_restored_after_an_exception(
    console_module, logger_module
):
    """An exception inside the block does not leave the streams swapped."""
    logger = _pack_logger(logger_module, "KEEP-test-console-restore-error")
    before = (sys.stdout, sys.stderr)

    with pytest.raises(RuntimeError):
        with console_module.capture_console_to_logger(logger):
            raise RuntimeError("boom")

    assert (sys.stdout, sys.stderr) == before


@pytest.mark.unit
def test_a_nested_block_restores_the_outer_one(console_module, logger_module):
    """The blocks may be nested - the inner one does not break the outer."""
    logger = _pack_logger(logger_module, "KEEP-test-console-nested")
    outer = _pack_logger(logger_module, "KEEP-test-console-nested-outer")
    before = (sys.stdout, sys.stderr)

    with _records(logger) as records_inner, _records(outer) as records_outer:
        with console_module.capture_console_to_logger(outer):
            with console_module.capture_console_to_logger(logger):
                print("inner")
            print("outer")

    assert (sys.stdout, sys.stderr) == before
    assert _messages(records_outer) == ["[KEEP] outer"]
    assert _messages(records_inner) == ["[KEEP] inner"]


# ---------------------------------------------------------------------------
# the output of the host
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_output_of_comfyui_is_not_swallowed(
    console_module, logger_module, monkeypatch
):
    """The root handler of the host holds its own stream - it is not routed.

    ``app.logger.setup_logger()`` installs the ``LogInterceptor`` before it
    builds the root handler, so the handler writes straight through the swap.
    """
    logger = _pack_logger(logger_module, "KEEP-test-console-comfy")
    original = io.StringIO()
    monkeypatch.setattr(sys, "stdout", original)
    monkeypatch.setattr(sys, "stderr", original)

    handler = logging.StreamHandler(original)
    host_logger = logging.getLogger("KEEP-test-comfy-host")
    host_logger.setLevel(logging.WARNING)
    host_logger.addHandler(handler)
    try:
        with console_module.capture_console_to_logger(logger):
            host_logger.warning("comfy says hello")
    finally:
        host_logger.removeHandler(handler)

    assert "comfy says hello" in original.getvalue()


# ---------------------------------------------------------------------------
# the tqdm removal
# ---------------------------------------------------------------------------
def _imported_modules(path: Path) -> set[str]:
    """The top level modules a source file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


@pytest.mark.unit
def test_the_processor_does_not_import_tqdm():
    """The progress bars of the processor are gone - the pack logs its steps."""
    source = SRC_DIR / "keep_processor.py"
    assert "tqdm" not in _imported_modules(source)
    assert "tqdm(" not in source.read_text(encoding="utf-8")


@pytest.mark.unit
def test_the_processor_logs_the_four_steps():
    """Each sequence path logs its four steps - one message each, in order."""
    source = (SRC_DIR / "keep_processor.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SRC_DIR / "keep_processor.py"))
    four_steps = [
        "Detecting face landmarks",
        "Cropping and aligning faces",
        "Restoring faces with KEEP",
        "Pasting faces and finalizing frames",
    ]

    def _stage_messages(fn):
        calls = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "info"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ]
        messages = [node.args[0].value for node in sorted(calls, key=lambda call: call.lineno)]
        return [m for m in messages if m in four_steps]

    paths = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _stage_messages(node)
    ]
    assert paths, "no sequence path logs its four steps"
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in paths:
            assert _stage_messages(node) == four_steps, node.name
