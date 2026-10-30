"""The guard around every adapter: timeout, exception capture, stdout silencing, uniform record shape."""
from __future__ import annotations

import contextlib
import io
import sys
import threading
import time
import traceback
import warnings
from typing import Any, Callable

from ..util import clip, jsonable

DEFAULT_TIMEOUT_S = 120.0

# Set when a worker thread outlived its timeout: the CLI then leaves with os._exit so a stuck C call cannot hang the job.
STUCK_THREADS: list[str] = []


class Skipped(Exception):
    """Raised by an adapter that does not apply in this run (missing credentials, disabled in config, no input data)."""


class _Sink(io.TextIOBase):
    """Swallows library chatter (banners, progress bars) while an adapter runs."""

    def write(self, s: str) -> int:
        return len(s)


def record(status: str, reason: str = "", seconds: float = 0.0, results: dict | None = None) -> dict:
    return {"status": status, "reason": clip(reason, 200) if reason else "", "seconds": round(float(seconds), 3),
            "results": jsonable(results or {})}


def run_guarded(fn: Callable[[Any], dict], ctx: Any, timeout: float | None = None, name: str = "adapter") -> dict:
    """Run fn(ctx) in a daemon thread; always return {status, reason, seconds, results}; never raise.

    status: ok | failed | skipped. Import failures carry the exception text (first 200 chars) as reason.
    """
    limit = float(timeout if timeout is not None else getattr(ctx, "timeout", lambda: DEFAULT_TIMEOUT_S)())
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["results"] = fn(ctx)
        except Skipped as exc:
            box["skipped"] = str(exc)
        except ImportError as exc:      # ModuleNotFoundError is a subclass
            box["failed"] = f"import failed: {type(exc).__name__}: {exc}"
        except BaseException as exc:    # noqa: BLE001 - a library may raise anything, including SystemExit
            box["failed"] = f"{type(exc).__name__}: {exc}"
            box["trace"] = "".join(traceback.format_exception_only(type(exc), exc)) + "".join(traceback.format_tb(exc.__traceback__)[-3:])

    t0 = time.monotonic()
    real_out, real_err = sys.stdout, sys.stderr
    thread = threading.Thread(target=target, name=f"lab-{name}", daemon=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.stdout = sys.stderr = _Sink()
        try:
            thread.start()
            thread.join(limit)
        finally:
            sys.stdout, sys.stderr = real_out, real_err
    seconds = time.monotonic() - t0
    if thread.is_alive():
        STUCK_THREADS.append(name)
        return record("failed", f"timeout after {limit:.0f}s", seconds)
    if "skipped" in box:
        return record("skipped", box["skipped"], seconds)
    if "failed" in box:
        return record("failed", box["failed"], seconds, {"traceback": clip(box.get("trace", ""), 600)})
    res = box.get("results")
    if not isinstance(res, dict):
        return record("failed", f"adapter returned {type(res).__name__}, expected dict", seconds)
    note = str(res.pop("_reason", "") or "")
    forced = res.pop("_status", None)       # an adapter may report "failed" and still hand back partial results
    return record(forced if forced in ("failed", "skipped") else "ok", note, seconds, res)


@contextlib.contextmanager
def quiet():
    """Silence stdout inside an adapter that is run outside run_guarded (tests)."""
    old = sys.stdout
    sys.stdout = _Sink()
    try:
        yield
    finally:
        sys.stdout = old
