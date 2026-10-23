"""The central logger of the KEEP package (P3.2).

Before P3.2 the logger lived in the root ``__init__.py`` and every engine
imported it with ``from .. import logger``.  That import only works when the
root package has been **executed**, which forced the test harness (P1.2) to
register the package with a placeholder logger.

The logger is a module of its own now: every engine imports it directly
(``from .keep_logger import logger``), so the constraint is lifted - the
engines import cleanly without the package entry point.
"""

from __future__ import annotations

import logging
import sys

#: The name of the package logger.
LOGGER_NAME = "ComfyUI-KEEP"

#: The level of the package logger.
LOG_LEVEL = logging.INFO


def get_logger(name: str | None = None) -> logging.Logger:
    """The package logger - or one of its children.

    The logger owns **exactly one** handler: the check is made against the
    logger's own handlers (not against its ancestors), so the handler is
    installed even when the root logger already has one (pytest installs
    its own capture handler there).
    """
    logger = logging.getLogger(LOGGER_NAME if name is None else name)
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


#: The package logger.
logger = get_logger()
