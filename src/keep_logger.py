"""The logger of the KEEP pack - ComfyUI's console, one ``[KEEP]`` tag.

The pack used to own a **private** stream handler (the format
``"[%(levelname)s] %(message)s"``) and every call site carried a literal
``"KEEP: "`` prefix.  The console therefore showed a plain stdlib line -
``"[INFO] KEEP: downloading ..."`` - that bypassed ComfyUI's own console.

The logger is built around **one** decision and **one** tag:

* **the decision** - is a real ComfyUI in charge of the console?  When it is,
  the pack logger owns **no handler** and only **propagates**: the record
  reaches the root logger that ``app.logger.setup_logger()`` configured
  (``ColoredFormatter("%(message)s")``), so the console shows the coloured
  ``"[INFO] [KEEP] ..."``.  When ComfyUI is absent (the plain test harness),
  the pack owns **one** private handler on ``sys.stderr`` and does **not**
  propagate: the pack still prints exactly one tagged line and its records do
  not leak into pytest's root capture.
* **the tag** - ``[KEEP]`` is prepended by a single filter
  (:class:`_TagFilter`), so no call site carries a literal any more.  A text
  that leaves the logger (an exception message) is built with :func:`tagged`.

The pack keeps its **own** logger name (``"ComfyUI-KEEP"``) instead of using
``logging.getLogger("comfy")``: the tag filter of the pack would then also
prefix ComfyUI's own records.  Propagating to the root logger gives the same
visual result without the collateral tagging.

The module keeps the P3.2 constraint: the engines import the logger directly
(``from .keep_logger import logger``) and import cleanly without the package
entry point having been executed.
"""

from __future__ import annotations

import logging
import os
import sys

#: The tag every message of the pack carries.
TAG = "[KEEP]"

#: The name of the package logger.
LOGGER_NAME = "ComfyUI-KEEP"

#: The level of the package logger.
LOG_LEVEL = logging.INFO

#: The name of the private handler of the fallback mode (a handler that
#: belongs to the pack is recognisable by it).
FALLBACK_HANDLER_NAME = "keep-fallback"


def comfyui_is_present() -> bool:
    """Is a real ComfyUI in charge of the console?

    ``True`` when ComfyUI's own logging module is importable (``app.logger`` -
    the module ``main.py`` uses to configure the root logger) or when the
    ``comfy`` module of ``sys.modules`` is a **real** one (a file on disk).

    The test harness mocks ``comfy`` with a bare ``types.ModuleType`` (no
    ``__file__``); such a mock is correctly reported as **absent** - the pack
    falls back to its own handler there.
    """
    try:
        import app.logger  # noqa: F401 - the import itself is the probe
    except Exception:  # noqa: BLE001 - any import problem means "no ComfyUI"
        pass
    else:
        return True

    comfy_file = getattr(sys.modules.get("comfy"), "__file__", None)
    return bool(comfy_file) and os.path.isfile(comfy_file)


class _TagFilter(logging.Filter):
    """Prefix the message of every record with the ``[KEEP]`` tag.

    The filter is the **only** place the tag is added to a log message - the
    call sites carry no literal.  The tag is added **once**: a message that
    already starts with it (an exception text built with :func:`tagged`) is
    left alone, and the filter is idempotent.

    The arguments of the record are resolved **before** the tag is added and
    are cleared afterwards: the record then carries the final text and cannot
    be formatted a second time (a stray ``%`` in a message would otherwise
    raise).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if not message.startswith(TAG):
            message = f"{TAG} {message}"
        record.msg = message
        record.args = None
        return True


def _fallback_handler() -> logging.StreamHandler:
    """The private handler of the no-ComfyUI mode."""
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(FALLBACK_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    return handler


def configure_logger(
    logger: logging.Logger, *, comfyui_present: bool | None = None
) -> logging.Logger:
    """Give a pack logger its mode and its tag filter.

    The mode is the **decision** of this module:

    * **ComfyUI present** - the logger owns no handler and propagates to the
      root logger that ComfyUI configured;
    * **ComfyUI absent** - the logger owns exactly one private handler on
      ``sys.stderr`` and does not propagate.

    ``comfyui_present`` overrides the detection; it is the seam the tests use
    to exercise both modes without a ComfyUI server.  The call is idempotent:
    the handler of a previous fallback run is removed first.
    """
    present = comfyui_is_present() if comfyui_present is None else comfyui_present

    logger.setLevel(LOG_LEVEL)
    if not any(isinstance(f, _TagFilter) for f in logger.filters):
        logger.addFilter(_TagFilter())

    for handler in list(logger.handlers):
        if handler.get_name() == FALLBACK_HANDLER_NAME:
            logger.removeHandler(handler)

    if present:
        # ComfyUI owns the console - the pack only propagates to it.
        logger.propagate = True
    else:
        # No ComfyUI: the pack prints the line itself, exactly once.
        logger.propagate = False
        logger.addHandler(_fallback_handler())
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """The package logger - or one of its children.

    A child carries the tag filter of its own: the filters of a parent logger
    do **not** apply to the records of a child.
    """
    logger = logging.getLogger(LOGGER_NAME if name is None else name)
    return configure_logger(logger)


def tagged(message: str) -> str:
    """The message with the pack's tag - for texts that leave the logger.

    An exception message is not a log record: it never passes the tag filter,
    so it is built here - with the same tag as the log lines.
    """
    return f"{TAG} {message}"


#: The package logger.
logger = get_logger()
