"""Technical indicators with Pine Script v6 semantics, on float64 numpy arrays.

Every function returns an array of the input length with NaN where Pine returns na.
Warm-up bars follow Pine: sma/highest/lowest need n values, ema/rma seed with the sma of the first n values,
rsi and roc need one extra bar for the change, atr uses high-low on the first bar.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:  # numba is optional: without it the same code runs as plain Python
    from numba import njit

    def jit(fn):
        return njit(cache=True)(fn)
except ImportError:  # pragma: no cover - exercised only where numba is missing
    def jit(fn):
        return fn


def shift(src: np.ndarray, k: int = 1) -> np.ndarray:
    """src[k] in Pine terms: the value k bars ago, NaN where it does not exist."""
    out = np.full(src.shape[0], np.nan)
    if k < src.shape[0]:
        out[k:] = src[: src.shape[0] - k]
    return out


def _windows(src: np.ndarray, n: int) -> np.ndarray | None:
    return sliding_window_view(src, n) if src.shape[0] >= n else None


def sma(src: np.ndarray, n: int) -> np.ndarray:
    out = np.full(src.shape[0], np.nan)
    win = _windows(src, n)
    if win is not None:
        out[n - 1:] = win.mean(axis=1)
    return out


@jit
def _seeded_ma(src, n, alpha):
    out = np.full(src.shape[0], np.nan)
    prev = np.nan
    for i in range(src.shape[0]):
        if np.isnan(prev):
            cur = np.nan
            if i >= n - 1:
                total = 0.0
                ok = True
                for j in range(i - n + 1, i + 1):
                    if np.isnan(src[j]):
                        ok = False
                        break
                    total += src[j]
                if ok:
                    cur = total / n
        else:
            cur = alpha * src[i] + (1.0 - alpha) * prev
        out[i] = cur
        prev = cur
    return out


def ema(src: np.ndarray, n: int) -> np.ndarray:
    """ta.ema: alpha 2/(n+1), seeded with the sma of the first n values."""
    return _seeded_ma(np.ascontiguousarray(src, dtype=np.float64), n, 2.0 / (n + 1.0))


def rma(src: np.ndarray, n: int) -> np.ndarray:
    """ta.rma (Wilder): alpha 1/n, seeded with the sma of the first n values."""
    return _seeded_ma(np.ascontiguousarray(src, dtype=np.float64), n, 1.0 / n)


def ma(kind: str, src: np.ndarray, n: int) -> np.ndarray:
    if kind == "ema":
        return ema(src, n)
    if kind == "sma":
        return sma(src, n)
    raise ValueError(f"unknown moving average kind {kind!r}")


def rsi(src: np.ndarray, n: int) -> np.ndarray:
    """ta.rsi: rma of gains over rma of losses; 100 when there are no losses."""
    chg = np.diff(src, prepend=np.nan)
    up = np.where(np.isnan(chg), np.nan, np.maximum(chg, 0.0))
    dn = np.where(np.isnan(chg), np.nan, np.maximum(-chg, 0.0))
    ru, rd = rma(up, n), rma(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(rd == 0, 100.0, np.where(ru == 0, 0.0, 100.0 - 100.0 / (1.0 + ru / rd)))
    res[np.isnan(ru) | np.isnan(rd)] = np.nan
    return res


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray, first_bar_hl: bool = True) -> np.ndarray:
    """ta.tr. ta.atr uses ta.tr(true), where the first bar is high-low; ta.dmi uses the plain variable (na)."""
    prev = shift(close, 1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    if first_bar_hl and tr.shape[0]:
        tr[0] = high[0] - low[0]
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    return rma(true_range(high, low, close, True), n)


def highest(src: np.ndarray, n: int) -> np.ndarray:
    out = np.full(src.shape[0], np.nan)
    win = _windows(src, n)
    if win is not None:
        out[n - 1:] = win.max(axis=1)
    return out


def lowest(src: np.ndarray, n: int) -> np.ndarray:
    out = np.full(src.shape[0], np.nan)
    win = _windows(src, n)
    if win is not None:
        out[n - 1:] = win.min(axis=1)
    return out


def roc(src: np.ndarray, n: int) -> np.ndarray:
    """100 * (src - src[n]) / src[n]."""
    base = shift(src, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (src - base) / base


def stdev(src: np.ndarray, n: int) -> np.ndarray:
    """ta.stdev, biased (population) like Pine's default."""
    out = np.full(src.shape[0], np.nan)
    win = _windows(src, n)
    if win is not None:
        out[n - 1:] = win.std(axis=1, ddof=0)
    return out


def percentrank(src: np.ndarray, n: int) -> np.ndarray:
    """ta.percentrank: percent of the previous n values that are <= the current value."""
    out = np.full(src.shape[0], np.nan)
    if src.shape[0] > n:
        prev = sliding_window_view(src[:-1], n)
        out[n:] = 100.0 * (prev <= src[n:, None]).sum(axis=1) / n
        out[n:][np.isnan(src[n:])] = np.nan
    return out


def bollinger(close: np.ndarray, n: int, mult: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = sma(close, n)
    dev = mult * stdev(close, n)
    return basis, basis + dev, basis - dev


def keltner(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int, mult: float
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = ema(close, n)
    rng = mult * atr(high, low, close, n)
    return mid, mid + rng, mid - rng


def _fixnan(src: np.ndarray) -> np.ndarray:
    """Forward-fill NaN with the last valid value (Pine fixnan)."""
    idx = np.where(np.isnan(src), 0, np.arange(src.shape[0]))
    np.maximum.accumulate(idx, out=idx)
    return src[idx]


def dmi(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ta.dmi(n, n): returns (+DI, -DI, ADX)."""
    up = np.diff(high, prepend=np.nan)
    down = -np.diff(low, prepend=np.nan)
    plus_dm = np.where(np.isnan(up), np.nan, np.where((up > down) & (up > 0), up, 0.0))
    minus_dm = np.where(np.isnan(down), np.nan, np.where((down > up) & (down > 0), down, 0.0))
    tr = rma(true_range(high, low, close, False), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = _fixnan(100.0 * rma(plus_dm, n) / tr)
        minus = _fixnan(100.0 * rma(minus_dm, n) / tr)
        total = plus + minus
        adx = 100.0 * rma(np.abs(plus - minus) / np.where(total == 0, 1.0, total), n)
    return plus, minus, adx


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    return dmi(high, low, close, n)[2]


def _cmp(a: np.ndarray, b: np.ndarray | float, op) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return op(a, b)


def crossover(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    """a > b now and a[1] <= b[1]."""
    b_arr = np.broadcast_to(np.asarray(b, dtype=np.float64), a.shape)
    return _cmp(a, b_arr, np.greater) & _cmp(shift(a), shift(b_arr), np.less_equal)


def crossunder(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    """a < b now and a[1] >= b[1]."""
    b_arr = np.broadcast_to(np.asarray(b, dtype=np.float64), a.shape)
    return _cmp(a, b_arr, np.less) & _cmp(shift(a), shift(b_arr), np.greater_equal)
