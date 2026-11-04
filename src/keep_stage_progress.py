"""The progress lines of the KEEP sequence stages - through the logger.

The four sequence stages log their start line (``Detecting face
landmarks``, ``Cropping and aligning faces``, ``Restoring faces with
KEEP``, ``Pasting faces and finalizing frames``) and then report their
progress on the **same** channel - the pack logger.

The bar is **logger-only** (owner request, 2026-11-04): the pack never
writes to the console itself.  The earlier carriage-return mode wrote
``"\\r" + "[KEEP] " + line`` straight to ``sys.__stdout__`` - that bypassed
the logger of the host, so the progress never reached the log file of
ComfyUI, never reached the web UI and showed up in the terminal as a
plain, untagged console line.  Every render is now one ``logger.info``
record: the root logger of ComfyUI renders it exactly like the host's own
lines, and the pack owns no console stream at all.

:class:`StageBar` is one bar per stage per run: the per-chunk counts of
the chunked loops accumulate into the stage total (one construction
before the loop, one ``finish`` after it, no per-chunk reset).

The bar never touches the V3 execution-API reporter
(:mod:`ComfyUI_KEEP.src.keep_progress`) - the UI progress keeps its exact
totals and update sequence.  The bar's ``tick`` calls are inserted
**beside** the reporter's ``update`` calls, never instead.

The cadence is the design's time gate: a render fires only when
``now - last_render >= interval_s``, so a fast stage renders every
``interval_s`` (not per tick) and a slow stage at most once per tick.  One
clock read per call.

The bar is stdlib-only (``time``): no new dependency, and the tqdm
tripwire (``tests/test_keep_regressions.py``) stays green.  It holds no
global state, registers nothing, and never touches ``keep_interrupt`` or
the VRAM releases; abandoning a bar without ``finish()`` (the exception
paths) leaks nothing - the bar renders from ``tick``/``finish`` only, so
an exception leaves the stage silent, which is the intended behavior: a
100%-looking line after a failure would lie.
"""

from __future__ import annotations

import time

from .keep_logger import logger

#: The rate label of each accepted unit ("frames" -> "fps",
#: "faces" -> "faces/s").  Anything else fails the construction.
_UNIT_RATES = {"frames": "fps", "faces": "faces/s"}


def _mmss(seconds: float) -> str:
    """The floor ``mm:ss`` rendering of ``seconds``.

    The minutes are unbounded - 3661 s renders ``61:01``.
    """
    whole = int(seconds)
    return f"{whole // 60:02d}:{whole % 60:02d}"


class StageBar:
    """A per-stage progress line through the pack logger (the stage-bar design, D1).

    One bar counts **one** stage of one run: the processor constructs it
    right after the stage's log line and finishes it when the stage's work
    is done.  The rendered line is the design's pin::

        {label} {done}/{total} ({pct}%) {rate:.1f} {rate_unit} ETA {mm:ss} elapsed {mm:ss}

    * ``pct`` is the floor of ``100 * done / total`` - never 100 before
      ``done == total``;
    * ``rate`` is the overall average ``done / elapsed`` since the stage
      started (a rolling window would make the ETA jitter) - ``0.0`` when
      the elapsed time or the count is zero;
    * the ETA is ``remaining / rate`` - ``--:--`` while the rate is zero
      and ``00:00`` once ``done >= total``; the ``mm:ss`` fields floor and
      keep unbounded minutes.

    The tests inject the clock (``clock``) - no wall-clock value reaches
    an assertion.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        unit: str = "frames",
        interval_s: float = 2.0,
        clock=time.monotonic,
    ) -> None:
        if unit not in _UNIT_RATES:
            raise ValueError(
                f"unknown bar unit {unit!r} - expected one of {sorted(_UNIT_RATES)}"
            )
        self.label = label
        self.total = total
        self.unit = unit
        self._rate_unit = _UNIT_RATES[unit]
        self._interval_s = interval_s
        self._clock = clock
        self._done = 0
        self._start = clock()
        self._last_render = self._start
        self._finished = False

    def tick(self, n: int = 1) -> None:
        """Count ``n`` more processed units; render when the cadence allows.

        ``n <= 0``, a call after :meth:`finish` and a bar over zero work
        are no-ops.  The cadence is the design's time gate: a render fires
        only when ``now - last_render >= interval_s``, so a fast stage
        renders every ``interval_s`` (not per tick) and a slow stage at
        most once per tick.  One clock read per call.
        """
        if n <= 0 or self._finished or self.total <= 0:
            return
        self._done += n
        now = self._clock()
        if now - self._last_render >= self._interval_s:
            self._render(now)

    def finish(self) -> None:
        """The final render - exactly once, even when no cadence fired.

        The stage calls ``finish`` on its normal path only: when a stage
        raises (OOM, cancel), ``finish`` is never reached and no final
        line is logged - the exception is the report.  A bar over zero
        work stays silent (``total == 0`` is inert).
        """
        if self._finished or self.total <= 0:
            return
        self._finished = True
        now = self._clock()
        self._render(now)

    # -- the rendering -------------------------------------------------
    def _render(self, now: float) -> None:
        """One render - the cadence render of :meth:`tick` or the final one.

        The render is **one log record** through the pack logger: the tag
        filter of :mod:`keep_logger` adds the ``[KEEP]`` tag and the root
        logger of ComfyUI renders the line like its own.  The pack writes
        no byte to any console stream.
        """
        self._last_render = now
        logger.info(self._line(now))

    def _line(self, now: float) -> str:
        """The rendered line - the exact format the design pins."""
        elapsed = now - self._start
        if elapsed > 0 and self._done > 0:
            rate = self._done / elapsed
        else:
            rate = 0.0
        if self._done >= self.total:
            eta = "00:00"
        elif rate > 0:
            eta = _mmss((self.total - self._done) / rate)
        else:
            eta = "--:--"
        pct = (100 * self._done) // self.total
        return (
            f"{self.label} {self._done}/{self.total} ({pct}%) "
            f"{rate:.1f} {self._rate_unit} ETA {eta} elapsed {_mmss(elapsed)}"
        )
