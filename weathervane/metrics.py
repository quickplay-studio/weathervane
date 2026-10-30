"""Performance statistics on daily account returns (fractions of the initial balance) and on trades in R."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: np.ndarray) -> np.ndarray:
    """Additive equity, starting at 1.0 (risk is a fraction of the initial balance)."""
    return 1.0 + np.cumsum(returns)


def max_drawdown(returns: np.ndarray) -> float:
    """Largest peak-to-trough fall of the additive equity curve, as a fraction of the initial balance."""
    if returns.size == 0:
        return 0.0
    eq = np.r_[1.0, equity_curve(returns)]
    return float((np.maximum.accumulate(eq) - eq).max())


def sharpe(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    sd = returns.std(ddof=1) if returns.size > 1 else 0.0
    return float(returns.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def sortino(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)) if returns.size else 0.0
    return float(returns.mean() / downside * np.sqrt(periods)) if downside > 0 else 0.0


def calmar(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    dd = max_drawdown(returns)
    return float(returns.mean() * periods / dd) if dd > 0 else 0.0


def cvar(returns: np.ndarray, level: float = 0.95) -> float:
    """Mean of the worst (1 - level) share of days, a negative number for a losing tail."""
    if returns.size == 0:
        return 0.0
    cut = np.quantile(returns, 1.0 - level)
    return float(returns[returns <= cut].mean())


def trade_stats(trades: pd.DataFrame) -> dict:
    """Expectancy (mean R), win rate, profit factor and average MAE in R."""
    if trades.empty:
        return {"trades": 0, "expectancy_r": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "avg_mae_r": 0.0}
    r = trades["r"].to_numpy(float)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": int(r.size), "expectancy_r": float(r.mean()), "win_rate": float((r > 0).mean()),
            "profit_factor": float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0,
            "avg_mae_r": float(trades["mae_r"].mean())}


def summary(returns: np.ndarray, trades: pd.DataFrame) -> dict:
    out = {"sharpe": sharpe(returns), "sortino": sortino(returns), "calmar": calmar(returns),
           "max_dd": max_drawdown(returns), "cvar95": cvar(returns), "total_return": float(returns.sum()),
           "days": int(returns.size)}
    out.update(trade_stats(trades))
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in out.items()}
