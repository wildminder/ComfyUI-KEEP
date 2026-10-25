"""Console routing for the KEEP pack - every console write reaches the logger.

The vendored facelib writes to the console with ``print()``: the model path
it resolves, the layers it fuses, the grayscale warning of ``read_image``.
Those writes bypass the console of ComfyUI: they are untagged, they do not
reach the log file of the host and they are invisible in the ComfyUI web UI.

The pack routes them to **its own** logger:

* the writes of the **owning thread** become log records - **one record per
  complete line** - and the tag filter of :mod:`keep_logger` adds the
  ``[KEEP]`` tag;
* the writes of **another thread** are forwarded to the original stream
  untouched: a node that runs in a worker thread never steals the console of
  the thread that entered the block;
* the streams are restored in the ``finally`` - also after an exception.

The output of ComfyUI is **not** swallowed: ``app.logger.setup_logger()``
installs its ``LogInterceptor`` **before** it builds the root handler, so the
root handler holds the interceptor object and writes straight through the
swap (``app/logger.py:107-120``).

``contextlib.redirect_stdout`` is deliberately not used: it captures the
writes of **every** thread and it would swallow the output of the host.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading

from .keep_logger import logger

#: The level of a captured line of ``sys.stdout``.
STDOUT_LEVEL = logging.INFO

#: The level of a captured line of ``sys.stderr``.
STDERR_LEVEL = logging.WARNING

#: The names of the streams the router replaces.
_STREAM_NAMES = ("stdout", "stderr")


class _Session:
    """One capture block - the logger, the levels and the pending lines.

    The session belongs to **one thread**: the buffer of a partial line is
    therefore never shared between two threads.
    """

    __slots__ = ("logger", "levels", "buffers")

    def __init__(self, target_logger: logging.Logger, stdout_level: int, stderr_level: int):
        self.logger = target_logger
        self.levels = {"stdout": stdout_level, "stderr": stderr_level}
        self.buffers = {"stdout": "", "stderr": ""}


class _RoutedStream:
    """The proxy that replaces ``sys.stdout`` / ``sys.stderr``.

    The proxy is a thin file-like object: it forwards everything the router
    does not capture and it delegates the unknown attributes to the original
    stream (``buffer``, ``closed``, ...).
    """

    def __init__(self, router: "_ConsoleRouter", name: str, original):
        # the original stream is set first: ``__getattr__`` delegates to it
        self._original = original
        self._router = router
        self._name = name

    # -- the file protocol --------------------------------------------
    def write(self, data):
        return self._router.write(self._name, data, self._original)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self._router.flush(self._name, self._original)

    def isatty(self) -> bool:
        isatty = getattr(self._original, "isatty", None)
        return bool(isatty()) if callable(isatty) else False

    def fileno(self) -> int:
        return self._original.fileno()

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._original, "errors", None)

    def __getattr__(self, name):
        # Everything the proxy does not implement itself belongs to the
        # original stream (the file protocol of ``sys.stdout``).
        return getattr(self._original, name)


class _ConsoleRouter:
    """The object that owns the swapped streams.

    The router is **process-global** (``sys.stdout`` is a module attribute),
    but the capture is **thread-scoped**: every thread that entered a block
    owns a session, and only the writes of that thread are routed to its
    logger.  The streams are swapped when the first session starts and
    restored when the last one ends - a second thread that enters the block
    while the first one is inside it therefore shares the proxy safely.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: dict[int, list[_Session]] = {}
        self._originals: dict[str, object] = {}
        self._proxies: dict[str, _RoutedStream] = {}
        self._local = threading.local()

    # -- the sessions -------------------------------------------------
    def enter(self, session: _Session) -> None:
        """Register the session of the calling thread."""
        with self._lock:
            self._sessions.setdefault(threading.get_ident(), []).append(session)
            self._install()

    def leave(self, session: _Session) -> None:
        """Unregister the session and flush its pending lines."""
        with self._lock:
            stack = self._sessions.get(threading.get_ident(), [])
            if session in stack:
                stack.remove(session)
            if not stack:
                self._sessions.pop(threading.get_ident(), None)
            pending = self._take_pending(session)
            if not self._sessions:
                self._uninstall()
        for name, line in pending:
            self._emit(session, name, line)

    def _take_pending(self, session: _Session) -> list[tuple[str, str]]:
        """The pending (incomplete) lines of a session - and clear them."""
        pending = []
        for name in _STREAM_NAMES:
            line = session.buffers[name]
            session.buffers[name] = ""
            if line:
                pending.append((name, line))
        return pending

    def _session(self) -> _Session | None:
        """The session of the calling thread - ``None`` when it has none."""
        with self._lock:
            stack = self._sessions.get(threading.get_ident())
            return stack[-1] if stack else None

    # -- the streams --------------------------------------------------
    def _install(self) -> None:
        """Swap the streams - once, for the first session."""
        if self._proxies:
            return
        for name in _STREAM_NAMES:
            original = getattr(sys, name)
            if original is None:  # pragma: no cover - a windowed interpreter
                continue
            self._originals[name] = original
            proxy = _RoutedStream(self, name, original)
            self._proxies[name] = proxy
            setattr(sys, name, proxy)

    def _uninstall(self) -> None:
        """Restore the streams - only the ones this router replaced."""
        for name, proxy in self._proxies.items():
            if getattr(sys, name) is proxy:
                setattr(sys, name, self._originals[name])
        self._proxies.clear()
        self._originals.clear()

    # -- the writes ---------------------------------------------------
    def write(self, name: str, data, original) -> int:
        """Route a write - or forward it to the original stream."""
        session = self._session()
        if session is None or not isinstance(data, str):
            # Another thread (or a non-text write): the original stream.
            return original.write(data)
        if getattr(self._local, "routing", False):
            # The logger writes through this proxy - do not recurse.
            return original.write(data)

        with self._lock:
            session.buffers[name] += data
            parts = session.buffers[name].split("\n")
            session.buffers[name] = parts.pop()
        for line in parts:
            self._emit(session, name, line)
        return len(data)

    def flush(self, name: str, original) -> None:
        """Flush the pending line of the calling thread and the original."""
        session = self._session()
        if session is not None:
            with self._lock:
                pending = session.buffers[name]
                session.buffers[name] = ""
            if pending:
                self._emit(session, name, pending)
        try:
            original.flush()
        except (ValueError, OSError):  # pragma: no cover - a closed stream
            pass

    def _emit(self, session: _Session, name: str, line: str) -> None:
        """Send one complete line to the logger of the session."""
        self._local.routing = True
        try:
            session.logger.log(session.levels[name], line)
        finally:
            self._local.routing = False


#: The router of the pack - one per process.
_ROUTER = _ConsoleRouter()


@contextlib.contextmanager
def capture_console_to_logger(
    target_logger: logging.Logger | None = None,
    *,
    stdout_level: int = STDOUT_LEVEL,
    stderr_level: int = STDERR_LEVEL,
):
    """Route the console writes of the **calling thread** to the logger.

    Inside the block the ``print()`` calls of the calling thread become log
    records of ``target_logger`` (the logger of the pack by default) - one
    record per complete line.  A partial line is buffered until its newline
    arrives; a line that is still incomplete when the block ends is flushed
    as the last record of the block.

    The writes of **another thread** are forwarded to the original stream
    untouched, and the streams are restored when the block ends - also after
    an exception.
    """
    session = _Session(
        target_logger if target_logger is not None else logger,
        stdout_level,
        stderr_level,
    )
    _ROUTER.enter(session)
    try:
        yield session
    finally:
        _ROUTER.leave(session)
