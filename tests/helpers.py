"""Shared builders for tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from weathervane.signals_core import Bars, Signals


def bars_from(o, h, l, c, start: str = "2020-01-01") -> Bars:
    idx = pd.bdate_range(start, periods=len(o), name="date")
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)
    return Bars(df)


def signals_for(n: int, long=(), short=(), exit_long=(), exit_short=(), atr: float = 2.0) -> Signals:
    def mask(ix):
        m = np.zeros(n, bool)
        m[list(ix)] = True
        return m
    return Signals(mask(long), mask(short), mask(exit_long), mask(exit_short), np.full(n, atr))


def sleeve_with(stop_atr=1.5, target_atr=None, trail_atr=None, time_stop_bars=None, reverse=True) -> dict:
    return {"id": "t", "exit": {"stop_atr": stop_atr, "target_atr": target_atr, "trail_atr": trail_atr,
                                "time_stop_bars": time_stop_bars, "exit_signal": None,
                                "reverse_on_opposite": reverse}}


def random_walk_df(n: int = 1500, seed: int = 1, start: str = "2010-01-01", vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, vol, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, vol / 4, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0}, index=idx)
