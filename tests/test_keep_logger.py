"""The logger of the pack - ComfyUI's console, one ``[KEEP]`` tag.

The pack used to own a private stream handler and every call site carried a
literal ``"KEEP: "`` prefix.  The logger is now built around two facts:

* **the mode** - with a real ComfyUI the pack logger owns **no handler** and
  only propagates to the root logger that ComfyUI configured (the console
  shows the coloured ``"[INFO] [KEEP] ..."``); without ComfyUI the pack owns
  **exactly one** private handler on ``sys.stderr`` and prints the tagged line
  itself;
* **the tag** - ``[KEEP]`` is added by a single filter, so no call site
  carries a literal any more.

The tests exercise both modes through the public API (the mode is a parameter
of :func:`configure_logger`), so they do not need a ComfyUI server.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import io
import logging
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


@pytest.fixture(scope="module")
def logger_module():
    return importlib.import_module("ComfyUI_KEEP.src.keep_logger")


@contextlib.contextmanager
def _root_records():
    """The records the **root** logger receives while the block runs."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield records
    finally:
        root.removeHandler(handler)


@contextlib.contextmanager
def _captured_stream(handler: logging.Handler):
    """The text a handler writes while the block runs."""
    buffer = io.StringIO()
    original = handler.stream
    handler.stream = buffer
    try:
        yield buffer
    finally:
        handler.stream = original


def _pack_logger(logger_module, name: str, **kwargs) -> logging.Logger:
    """A **dedicated** pack logger - never the global one.

    The name carries no dot on purpose: a dotted name would make the logger a
    child of ``ComfyUI-KEEP`` and the records would travel through the global
    pack logger.
    """
    return logger_module.configure_logger(logging.getLogger(name), **kwargs)


def _messages(records, name: str) -> list[str]:
    """The messages of the records of one logger."""
    return [record.getMessage() for record in records if record.name == name]


def _code_strings(path: Path) -> list[str]:
    """The string literals of a module - the docstrings are prose, not code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


# ---------------------------------------------------------------------------
# the constants
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_tag_and_the_logger_name_are_defined_once(logger_module):
    """The tag is ``[KEEP]`` (braces) and the logger keeps its own name."""
    assert logger_module.TAG == "[KEEP]"
    assert logger_module.LOGGER_NAME == "ComfyUI-KEEP"
    assert logger_module.LOG_LEVEL == logging.INFO


@pytest.mark.unit
def test_tagged_builds_the_text_that_leaves_the_logger(logger_module):
    """An exception message carries the same tag as a log line."""
    assert logger_module.tagged("the weight could not be downloaded") == (
        "[KEEP] the weight could not be downloaded"
    )


# ---------------------------------------------------------------------------
# the tag filter
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_every_message_carries_the_tag_exactly_once(logger_module):
    """The filter tags the message - once, and after the arguments."""
    logger = _pack_logger(logger_module, "KEEP-test-tag", comfyui_present=True)
    with _root_records() as records:
        logger.info("hello")
        logger.info("hello %s", "world")
        logger.info(logger_module.tagged("already tagged"))

    assert _messages(records, "KEEP-test-tag") == [
        "[KEEP] hello",
        "[KEEP] hello world",
        "[KEEP] already tagged",
    ]


@pytest.mark.unit
def test_the_tag_is_added_by_a_filter_of_the_pack_logger(logger_module):
    """The filter lives on the logger - the call sites carry no literal."""
    filters = logger_module.logger.filters
    assert filters, "the pack logger must carry the tag filter"
    assert all(isinstance(a_filter, logging.Filter) for a_filter in filters)


# ---------------------------------------------------------------------------
# the mode: ComfyUI is in charge
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_under_comfyui_the_logger_owns_no_handler_and_propagates(logger_module):
    """ComfyUI formats the record - the pack must not own a handler."""
    logger = _pack_logger(logger_module, "KEEP-test-comfy", comfyui_present=True)

    assert logger.handlers == []
    assert logger.propagate is True
    assert logger.level == logging.INFO


@pytest.mark.unit
def test_under_comfyui_the_record_reaches_the_root_logger(logger_module):
    """The record travels to the root logger that ComfyUI configured."""
    logger = _pack_logger(logger_module, "KEEP-test-root", comfyui_present=True)
    with _root_records() as records:
        logger.info("hello")

    assert _messages(records, "KEEP-test-root") == ["[KEEP] hello"]


# ---------------------------------------------------------------------------
# the mode: no ComfyUI
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_without_comfyui_the_logger_owns_one_private_handler(logger_module):
    """The fallback owns exactly one handler and does not propagate."""
    logger = _pack_logger(logger_module, "KEEP-test-fallback", comfyui_present=False)

    assert len(logger.handlers) == 1
    assert logger.propagate is False
    assert logger.level == logging.INFO
    assert logger.handlers[0].get_name() == logger_module.FALLBACK_HANDLER_NAME


@pytest.mark.unit
def test_the_fallback_prints_one_tagged_line(logger_module):
    """The no-ComfyUI harness sees exactly one tagged line."""
    logger = _pack_logger(logger_module, "KEEP-test-print", comfyui_present=False)
    handler = logger.handlers[0]

    with _captured_stream(handler) as stream:
        logger.info("hello")

    assert stream.getvalue() == "[INFO] [KEEP] hello\n"


@pytest.mark.unit
def test_the_fallback_does_not_leak_into_the_root_logger(logger_module):
    """The fallback does not duplicate the record into pytest's capture."""
    logger = _pack_logger(logger_module, "KEEP-test-quiet", comfyui_present=False)
    handler = logger.handlers[0]

    with _captured_stream(handler), _root_records() as records:
        logger.info("hello")

    assert _messages(records, "KEEP-test-quiet") == []


@pytest.mark.unit
def test_the_fallback_handler_is_not_installed_twice(logger_module):
    """Configuring the logger twice does not duplicate the handler."""
    logger = _pack_logger(logger_module, "KEEP-test-twice", comfyui_present=False)
    logger = _pack_logger(logger_module, "KEEP-test-twice", comfyui_present=False)

    assert len(logger.handlers) == 1


# ---------------------------------------------------------------------------
# the probe: is a real ComfyUI in charge?
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_mocked_comfy_module_is_reported_as_absent(logger_module, monkeypatch):
    """The harness mock (``types.ModuleType`` without ``__file__``) is no ComfyUI."""
    monkeypatch.setitem(sys.modules, "app", None)
    monkeypatch.setitem(sys.modules, "app.logger", None)
    monkeypatch.setitem(sys.modules, "comfy", types.ModuleType("comfy"))

    assert logger_module.comfyui_is_present() is False


@pytest.mark.unit
def test_a_file_backed_comfy_module_is_reported_as_present(
    logger_module, monkeypatch, tmp_path
):
    """A real ``comfy`` package (a file on disk) is a ComfyUI."""
    package = tmp_path / "comfy"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    module = types.ModuleType("comfy")
    module.__file__ = str(package / "__init__.py")

    monkeypatch.setitem(sys.modules, "app", None)
    monkeypatch.setitem(sys.modules, "app.logger", None)
    monkeypatch.setitem(sys.modules, "comfy", module)

    assert logger_module.comfyui_is_present() is True


# ---------------------------------------------------------------------------
# the logger of the pack
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_the_pack_logger_matches_the_detected_mode(logger_module):
    """The global logger is in the mode of the tier the tests run in."""
    logger = logger_module.logger

    assert logger.level == logging.INFO
    if logger_module.comfyui_is_present():
        assert logger.handlers == []
        assert logger.propagate is True
    else:
        assert len(logger.handlers) == 1
        assert logger.propagate is False


@pytest.mark.unit
def test_the_tag_literal_lives_only_in_the_logger_module():
    """The tag is added in one place - the call sites carry no literal."""
    for path in sorted(SRC_DIR.glob("*.py")):
        if path.name == "keep_logger.py":
            continue
        for literal in _code_strings(path):
            assert "[KEEP]" not in literal, (path.name, literal)


@pytest.mark.unit
def test_the_sources_of_this_batch_carry_no_tag_literal():
    """The logger and the progress report carry no literal prefix."""
    for name in ("keep_logger.py", "keep_progress.py"):
        for literal in _code_strings(SRC_DIR / name):
            assert "KEEP: " not in literal, (name, literal)
