"""Vectorised raw signals for a sleeve: entries after filters, exit signals and the sizing ATR."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind

MEMO_LIMIT = 96


class Bars:
    """OHLC arrays of one instrument plus a small memo of indicator results keyed by (name, params)."""

    def __init__(self, df: pd.DataFrame):
        self.index = df.index
        self.open = df["open"].to_numpy(np.float64)
        self.high = df["high"].to_numpy(np.float64)
        self.low = df["low"].to_numpy(np.float64)
        self.close = df["close"].to_numpy(np.float64)
        self._memo: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return self.close.shape[0]

    def memo(self, key: tuple, fn: Callable[[], object]):
        if key in self._memo:
            self._memo.move_to_end(key)
            return self._memo[key]
        val = fn()
        self._memo[key] = val
        if len(self._memo) > MEMO_LIMIT:
            self._memo.popitem(last=False)
        return val

    def atr(self, n: int) -> np.ndarray:
        return self.memo(("atr", n), lambda: ind.atr(self.high, self.low, self.close, n))

    def ma(self, kind: str, n: int) -> np.ndarray:
        return self.memo(("ma", kind, n), lambda: ind.ma(kind, self.close, n))

    def hh(self, n: int) -> np.ndarray:
        return self.memo(("hh", n), lambda: ind.highest(self.high, n))

    def ll(self, n: int) -> np.ndarray:
        return self.memo(("ll", n), lambda: ind.lowest(self.low, n))

    def roc(self, n: int) -> np.ndarray:
        return self.memo(("roc", n), lambda: ind.roc(self.close, n))

    def rsi(self, n: int) -> np.ndarray:
        return self.memo(("rsi", n), lambda: ind.rsi(self.close, n))

    def adx(self, n: int) -> np.ndarray:
        return self.memo(("adx", n), lambda: ind.adx(self.high, self.low, self.close, n))

    def boll(self, n: int, mult: float):
        return self.memo(("boll", n, mult), lambda: ind.bollinger(self.close, n, mult))

    def keltner(self, n: int, mult: float):
        return self.memo(("kelt", n, mult), lambda: ind.keltner(self.high, self.low, self.close, n, mult))

    def atr_rank(self, n: int, lookback: int) -> np.ndarray:
        return self.memo(("atr_rank", n, lookback),
                         lambda: ind.percentrank(self.atr(n) / self.close, lookback))


@dataclass
class Signals:
    long: np.ndarray
    short: np.ndarray
    exit_long: np.ndarray
    exit_short: np.ndarray
    atr: np.ndarray


def entry_signals(entry: dict, b: Bars) -> tuple[np.ndarray, np.ndarray]:
    """Raw (long, short) entry conditions evaluated on bar close."""
    kind, c = entry["type"], b.close
    if kind == "donchian_breakout":
        n = entry["len"]
        return c > ind.shift(b.hh(n)), c < ind.shift(b.ll(n))
    if kind == "ma_cross":
        fast, slow = b.ma(entry["kind"], entry["fast"]), b.ma(entry["kind"], entry["slow"])
        return ind.crossover(fast, slow), ind.crossunder(fast, slow)
    if kind == "keltner_breakout":
        _, upper, lower = b.keltner(entry["len"], entry["mult"])
        return ind.crossover(c, upper), ind.crossunder(c, lower)
    if kind == "momentum_roc":
        r = b.roc(entry["len"])
        return ind.crossover(r, entry["thr"]), ind.crossunder(r, -entry["thr"])
    if kind == "rsi_reversion":
        r = b.rsi(entry["len"])
        return ind.crossover(r, entry["lower"]), ind.crossunder(r, entry["upper"])
    if kind == "bollinger_reversion":
        _, upper, lower = b.boll(entry["len"], entry["mult"])
        return ind.crossover(c, lower), ind.crossunder(c, upper)
    raise ValueError(f"unknown entry type {kind!r}")


def _gt(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return a > b


def _lt(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return a < b


def filter_masks(flt: dict, b: Bars) -> tuple[np.ndarray, np.ndarray]:
    """(long_ok, short_ok) for one entry filter. NaN indicator values fail the filter."""
    kind, c = flt["type"], b.close
    if kind == "trend_ma":
        m = b.ma(flt["kind"], flt["len"])
        return _gt(c, m), _lt(c, m)
    if kind in ("adx_min", "adx_max"):
        a = b.adx(flt["len"])
        ok = (a >= flt["min"]) if kind == "adx_min" else (a <= flt["max"])
        ok = ok & ~np.isnan(a)
        return ok, ok
    if kind == "atr_rank":
        pr = b.atr_rank(flt["len"], flt["lookback"])
        ok = (pr >= flt["min"]) & (pr <= flt["max"]) & ~np.isnan(pr)
        return ok, ok
    if kind == "roc_dir":
        r = b.roc(flt["len"])
        return _gt(r, 0.0), _lt(r, 0.0)
    raise ValueError(f"unknown filter type {kind!r}")


def exit_signals(sig: dict | None, b: Bars) -> tuple[np.ndarray, np.ndarray]:
    """(exit_long, exit_short) conditions evaluated on bar close."""
    n = b.close.shape[0]
    if sig is None:
        return np.zeros(n, bool), np.zeros(n, bool)
    c = b.close
    if sig["type"] == "donchian_exit":
        return _lt(c, ind.shift(b.ll(sig["len"]))), _gt(c, ind.shift(b.hh(sig["len"])))
    m = b.ma(sig["kind"], sig["len"])
    return _lt(c, m), _gt(c, m)


def build_signals(sleeve: dict, b: Bars) -> Signals:
    """Entry signals gated by filters and direction, exit signals, and the sizing ATR for one instrument."""
    long, short = entry_signals(sleeve["entry"], b)
    for flt in sleeve.get("filters", []):
        ok_long, ok_short = filter_masks(flt, b)
        long, short = long & ok_long, short & ok_short
    if sleeve["direction"] == "long":
        short = np.zeros_like(short)
    elif sleeve["direction"] == "short":
        long = np.zeros_like(long)
    both = long & short
    atr = b.atr(sleeve["atr_len"])
    valid = ~np.isnan(atr) & (atr > 0)
    ex_long, ex_short = exit_signals(sleeve["exit"].get("exit_signal"), b)
    return Signals(long & ~both & valid, short & ~both & valid, ex_long, ex_short, atr)
