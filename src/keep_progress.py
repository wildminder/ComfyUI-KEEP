"""Progress reporting through the ComfyUI execution API (P4.2).

The V3 API of ComfyUI reports the progress of a node through
``comfy_api.latest``:

* the **synchronous** form is ``ComfyAPISync().execution.set_progress(value,
  max_value)``,
* the **asynchronous** form is ``await ComfyAPI().execution.set_progress(...)``.

Facts that were verified against the real ComfyUI source (v0.42.0) and
recorded in ``docs/audit/2026-10-23-p4-v3-migration.md``:

* ``execution`` is an **instance** attribute - the class level access
  (``ComfyAPISync.execution``) raises ``AttributeError``;
* ``ComfyAPISync().execution.set_progress`` is a **normal function** (the
  async method is wrapped), ``ComfyAPI().execution.set_progress`` is a
  coroutine function;
* outside an execution context the call raises ``ValueError`` ("node_id must
  be provided if not in executing context") - the call is therefore wrapped
  in a silent ``try/except``: a progress report must never break a run.

One **cached** instance is used: building a ``ComfyAPISync`` creates a thread
pool, so a new instance per call would create a thread pool per call.

The ``comfy_api`` import is **lazy** (function level): the engines of the
pack must import without a ComfyUI server (the plan's section 3 rule).
"""

from __future__ import annotations

from .keep_logger import logger

#: The cached synchronous API instance (``None`` until it is built).
_API_SYNC = None


def get_api_sync():
    """The cached ``ComfyAPISync`` instance (built on first use)."""
    global _API_SYNC
    if _API_SYNC is None:
        from comfy_api.latest import ComfyAPISync  # lazy: src/ imports without ComfyUI

        _API_SYNC = ComfyAPISync()
    return _API_SYNC


def report_progress(value: float, max_value: float) -> bool:
    """Report the progress of the running node.

    Returns ``True`` when the progress was reported and ``False`` when the
    reporting failed (no execution context, no ComfyUI, ...) - reporting the
    progress must never break the execution.
    """
    try:
        api = get_api_sync()
        api.execution.set_progress(value, max_value)
        return True
    except Exception as exc:  # noqa: BLE001 - a progress report never breaks a run
        logger.debug(f"KEEP: the progress could not be reported ({exc})")
        return False


class ProgressReporter:
    """A small counter that reports its progress to the ComfyUI UI."""

    def __init__(self, total: float):
        self.total = total
        self.current = 0

    def update(self, value: float) -> None:
        """Advance the counter and report the new value."""
        self.current += value
        report_progress(self.current, self.total)
