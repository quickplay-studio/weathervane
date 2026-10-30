"""pmorissette/ffn: performance statistics and return conventions checked against the core metrics.

Core and ffn use the same Sharpe definition (sample stdev of daily returns, 252 periods, zero risk-free rate), so those
are compared directly. ffn's drawdown is multiplicative while the core one is additive, so it is reported, not flagged.
"""
from __future__ import annotations

import numpy as np

from .. import crosscheck as cc
from ... import metrics
from ..context import LabContext
from ._common import version_of


def run(ctx: LabContext) -> dict:
    import ffn
    from ffn.core import calc_max_drawdown

    stats_by_instrument: dict[str, dict] = {}

    def build(f: cc.Frame) -> list[cc.Check]:
        px = f.close_series
        returns = ffn.to_returns(px).dropna()
        stats = ffn.calc_stats(px)
        stats_by_instrument[f.instrument] = {
            "cagr": float(stats.cagr), "daily_vol": float(stats.daily_vol), "daily_sharpe": float(stats.daily_sharpe),
            "max_drawdown": float(stats.max_drawdown), "total_return": float(stats.total_return)}
        r = cc.simple_returns(f)[1:]
        ffn_drawdown = np.array([abs(float(calc_max_drawdown(px)))])
        return [
            cc.Check("returns1", r, returns.to_numpy(float), "ffn.to_returns"),
            cc.Check("sharpe", np.array([metrics.sharpe(r)]), np.array([float(stats.daily_sharpe)]), "daily Sharpe, rf = 0"),
            cc.Check("max_drawdown_multiplicative", np.array([_multiplicative_drawdown(f.close)]), ffn_drawdown, "peak-to-trough on prices"),
            cc.Check("max_drawdown_additive_vs_ffn", np.array([metrics.max_drawdown(r)]), ffn_drawdown,
                     "core drawdown is additive on returns, ffn's is multiplicative: different definitions", definitions_match=False),
        ]

    out = cc.run_instruments(ctx, build)
    out["stats"] = stats_by_instrument
    out["version"] = version_of("ffn")
    return out


def _multiplicative_drawdown(close: np.ndarray) -> float:
    peak = np.maximum.accumulate(close)
    return float(np.max(1.0 - close / peak))
