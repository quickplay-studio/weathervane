"""Daily R streams of the champion sleeves and allocation weights computed from them.

The streams come from the core simulator (read-only use of live.run_live) on cached data, or on synthetic data when no
cache exists (flagged in the result). Weights are research suggestions only: equal risk is what the engine uses.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import config, data as datamod, live, simulator as sm
from .util import clip

log = logging.getLogger("weathervane.lab")
MIN_DAYS = 250


def portfolio_streams(pid: str, lookback_days: int = 1260) -> tuple[pd.DataFrame, dict]:
    """(DataFrame date x stream of daily R, info). Streams are sleeves, or sleeve:instrument when there is one sleeve."""
    cfg = config.load_engine()
    universe = config.load_universe()
    pf = config.load_portfolios()[pid]
    champion = config.load_champion(pid)
    source = "cache"
    data = datamod.load(list(pf.instruments), cfg)
    if not data:
        data, source = datamod.load(list(pf.instruments), cfg, synthetic=True), "synthetic"
    runs = live.run_live(champion, data, universe)
    if not runs:
        raise RuntimeError(f"{pid}: no sleeve could be simulated")
    sleeve_ids = sorted({r.sleeve["id"] for r in runs})
    unit = "sleeve" if len(sleeve_ids) > 1 else "sleeve:instrument"
    grid = sm.master_grid([r.bars.index for r in runs])
    columns: dict[str, np.ndarray] = {}
    if unit == "sleeve":
        for sid in sleeve_ids:
            rc, _, _ = sm.aggregate([r.sim for r in runs if r.sleeve["id"] == sid], grid)
            columns[sid] = rc
    else:
        for r in runs:
            rc, _, _ = sm.aggregate([r.sim], grid)
            columns[f"{r.sleeve['id']}:{r.instrument.id}"] = rc
    df = pd.DataFrame(columns, index=pd.DatetimeIndex(grid, name="date")).tail(lookback_days)
    info = {"unit": unit, "data_source": source, "champion": champion.get("id"), "days": int(len(df))}
    return df, info


def _normalise(w: np.ndarray) -> np.ndarray:
    total = w.sum()
    return w / total if abs(total) > 1e-12 else np.full(len(w), 1.0 / len(w))


def suggest_weights(streams: pd.DataFrame) -> dict:
    """Equal risk, inverse volatility, and the two formulas of QuantPy's Portfolio.get_w (reimplemented).

    QuantPy 'characteristic' is the minimum-variance weights V^-1 1 / (1' V^-1 1); 'sharpe' is V^-1 e / (e' V^-1 e)
    with e the sum of each stream's returns (it does not sum to one). The long-only version clips negatives to zero and
    renormalises. Streams with zero variance are left out of the optimisation and get weight 0.
    """
    live_cols = [c for c in streams.columns if float(streams[c].std(ddof=1)) > 0]
    names = list(streams.columns)
    n = len(names)
    out: dict = {"streams": names, "equal": {c: 1.0 / n for c in names}, "excluded": [c for c in names if c not in live_cols]}
    if len(live_cols) < 2:
        out["note"] = "fewer than two streams with variance; no optimisation"
        return out
    x = streams[live_cols].to_numpy(float)
    vol = x.std(axis=0, ddof=1)
    inv = _normalise(1.0 / vol)
    cov = np.cov(x, rowvar=False)
    icov = np.linalg.pinv(cov)
    ones = np.ones(len(live_cols))
    e = x.sum(axis=0)
    min_var = icov @ ones / (ones @ icov @ ones)
    tangency = icov @ e / (e @ icov @ e) if abs(e @ icov @ e) > 1e-12 else np.full(len(live_cols), np.nan)
    long_only = _normalise(np.clip(np.nan_to_num(tangency), 0.0, None))

    def pack(w: np.ndarray) -> dict:
        d = {c: 0.0 for c in names}
        d.update({c: float(v) for c, v in zip(live_cols, w)})
        return d

    out.update(inverse_vol=pack(inv), min_variance=pack(min_var), mvo_raw=pack(tangency), mvo_long_only=pack(long_only))
    out["stats"] = {c: {"mean_r": float(streams[c].mean()), "vol_r": float(streams[c].std(ddof=1)),
                        "sharpe": float(streams[c].mean() / streams[c].std(ddof=1) * np.sqrt(252)) if streams[c].std(ddof=1) > 0 else 0.0}
                    for c in names}
    return out


def all_portfolios(lookback_days: int = 1260) -> dict:
    """Weights for every portfolio that has a champion; a failing portfolio is reported, not raised."""
    results: dict = {}
    for pid in config.load_portfolios():
        try:
            df, info = portfolio_streams(pid, lookback_days)
            if len(df) < MIN_DAYS:
                raise RuntimeError(f"only {len(df)} days of simulated returns")
            results[pid] = {**info, **suggest_weights(df)}
        except Exception as exc:
            results[pid] = {"error": clip(f"{type(exc).__name__}: {exc}", 160)}
    return results
