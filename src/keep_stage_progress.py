"""The console progress bar of the KEEP sequence stages.

The four sequence stages log their start line (``Detecting face
landmarks``, ``Cropping and aligning faces``, ``Restoring faces with
KEEP``, ``Pasting faces and finalizing frames``) and then go silent until
the next stage line.  This module is the second, **console-only** progress
channel the stage-bar design adds:

* :class:`StageBar` - one bar per stage per run: the per-chunk counts of
  the chunked loops accumulate into the stage total (one construction
  before the loop, one ``finish`` after it, no per-chunk reset);
* the bar never touches the V3 execution-API reporter
  (:mod:`ComfyUI_KEEP.src.keep_progress`) - the UI progress keeps its exact
  totals and update sequence.  The bar's ``tick`` calls are inserted
  **beside** the reporter's ``update`` calls, never instead.

The output channel is decided **once** at construction:

* **carriage-return mode** - a real console is attached (the resolved
  stream claims ``isatty()``): every render rewrites the line in place -
  ``"\\r" + line``, padded with trailing spaces to erase the previous
  render's residue, then flushed; ``finish()`` writes the final render
  plus ``"\\n"`` so the next log line starts below it.  No ANSI escape
  codes (portable across the classic Windows console and Windows
  Terminal).  The line carries the pack tag ``[KEEP]`` so the bar is
  visually linked to the stage lines.  The writes go to
  ``sys.__stdout__`` - neither the D4 console router nor ComfyUI's log
  interceptor ever replaces ``__stdout__``, so a render never enters a
  capture session's line buffer and the one-record-per-line contract of
  the D4 capture is untouched.
* **logger mode** - piped output, log file, CI, pytest: every render is
  one complete line through ``logger.info`` on the pack logger (the tag
  filter adds the tag).  The routed streams receive **zero bytes** from
  the bar - no ``\\r`` garbage in pipes or log files.

The bar is stdlib-only (``time`` and ``sys``): no new dependency, and the
tqdm tripwire (``tests/test_keep_regressions.py``) stays green.  It holds
no global state, registers nothing, and never touches ``keep_interrupt``
or the VRAM releases; abandoning a bar without ``finish()`` (the
exception paths) leaks nothing - the bar renders from ``tick``/``finish``
only, so an exception leaves the stage silent, which is the intended
behavior: a 100%-looking line after a failure would lie.
"""

from __future__ import annotations

import sys
import time

from .keep_logger import TAG, logger

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
    """A per-stage console progress bar (the stage-bar design, D1).

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

    The tests inject the clock (``clock``) and the stream (``stream``) -
    no wall-clock value reaches an assertion.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        unit: str = "frames",
        interval_s: float = 2.0,
        clock=time.monotonic,
        stream=None,
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
        self._stream = stream if stream is not None else sys.__stdout__
        isatty = getattr(self._stream, "isatty", None)
        self._cr_mode = bool(
            self._stream is not None and callable(isatty) and isatty()
        )
        self._done = 0
        self._start = clock()
        self._last_render = self._start
        self._finished = False
        self._last_len = 0

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
        line is written - the exception is the report.  A bar over zero
        work stays silent (``total == 0`` is inert).
        """
        if self._finished or self.total <= 0:
            return
        self._finished = True
        now = self._clock()
        self._render(now, final=True)

    # -- the rendering -------------------------------------------------
    def _render(self, now: float, *, final: bool = False) -> None:
        """One render - the cadence render of :meth:`tick` or the final one.

        Carriage-return mode writes ``"\\r"`` + the tagged line, padded to
        the previous render's length so no residue survives, and flushes;
        the final render appends the newline.  Logger mode emits one
        complete line through the pack logger.
        """
        self._last_render = now
        line = self._line(now)
        if self._cr_mode:
            tagged_line = f"{TAG} {line}"
            terminator = "\n" if final else ""
            self._stream.write(f"\r{tagged_line.ljust(self._last_len)}{terminator}")
            self._stream.flush()
            self._last_len = len(tagged_line)
        else:
            logger.info(line)

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
