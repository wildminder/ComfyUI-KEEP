"""The cooperative cancellation of the long loops (the cancel round, C1).

ComfyUI's Cancel button sets a global flag (``interrupt_current_processing``,
``comfy/model_management.py:2205-2213`` of the real source); a node
cooperates by **polling** ``throw_exception_if_processing_interrupted()``
(:2214), which raises ``InterruptProcessingException`` - deliberately a
**BaseException** subclass (:2196), so the broad ``except Exception``
handlers of the pipeline cannot swallow it.  The poll is a flag read: the
per-iteration cost of a check is nil.

The callers use the **module-qualified** call
(``keep_interrupt.check_interrupted()``) - never a
``from .keep_interrupt import check_interrupted``: the module attribute is
the single patch point of the tests
(``tests/test_cancel_interrupt.py`` patches exactly it).
"""


def check_interrupted() -> None:
    """Raise ``InterruptProcessingException`` when ComfyUI's cancel flag is set.

    The import is **lazy** (function level): ``src/`` imports without a
    ComfyUI server (the same rule as
    :func:`.keep_processor._soft_empty_cache`).  Outside ComfyUI the check
    is a no-op.
    """
    try:
        from comfy import model_management  # lazy: src/ imports without ComfyUI
    except ImportError:
        return
    model_management.throw_exception_if_processing_interrupted()
