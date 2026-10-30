"""Scores and gates for an out-of-sample series: exam pass probability, deflated Sharpe, regimes, drawdown."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import exam, metrics, regime
from .config import Rules

EULER_GAMMA = 0.5772156649015329


def deflated_sharpe(returns: np.ndarray, n_trials: int, sharpe_variance: float | None = None) -> float:
    """Probability that the true Sharpe ratio is positive after deflating for the number of trials tried.

    Bailey and Lopez de Prado (2014). Sharpe ratios are per period (daily). sharpe_variance is the variance of the
    trials' Sharpe ratios; when unknown it defaults to the sampling variance 1 / T of one estimate.
    """
    t = returns.size
    sd = returns.std(ddof=1) if t > 2 else 0.0
    if t < 3 or sd <= 0:
        return 0.0
    sr = float(returns.mean() / sd)
    skew, kurt = float(stats.skew(returns)), float(stats.kurtosis(returns, fisher=False))
    var = sharpe_variance if sharpe_variance and sharpe_variance > 0 else 1.0 / t
    n = max(int(n_trials), 1)
    if n > 1:
        sr0 = np.sqrt(var) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n) + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e)))
    else:
        sr0 = 0.0
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))


def trial_sharpe_variance(prior_sharpes: list[float]) -> float | None:
    """Variance of earlier trials' per-day Sharpe ratios, or None when there are too few to estimate it."""
    vals = np.array([s for s in prior_sharpes if s is not None and np.isfinite(s)])
    return float(vals.var(ddof=1)) if vals.size >= 5 else None


def regime_table(trades: pd.DataFrame, labels: pd.Series | None) -> dict[str, dict]:
    """Trade count and expectancy in R per regime at entry."""
    if labels is None or trades.empty:
        return {}
    tagged = regime.label_trades(trades, labels)
    out = {}
    for name in regime.LABELS:
        r = trades.loc[tagged == name, "r"]
        out[name] = {"trades": int(r.size), "expectancy_r": float(r.mean()) if r.size else 0.0}
    return out


def regimes_ok(table: dict[str, dict], min_trades: int, min_expectancy: float) -> bool:
    return all(not (v["trades"] >= min_trades and v["expectancy_r"] < min_expectancy) for v in table.values())


@dataclass
class Score:
    oos_pass: float
    risk: float
    oos_trades: int
    dsr: float
    regime_ok: bool
    maxdd: float
    sharpe_day: float
    exam: dict
    table: list[dict]
    metrics: dict
    regimes: dict
    gates: dict[str, bool] = field(default_factory=dict)

    @property
    def passes_gates(self) -> bool:
        return all(self.gates.values())

    def as_dict(self) -> dict:
        return {"oos_pass": self.oos_pass, "risk": self.risk, "oos_trades": self.oos_trades, "dsr": self.dsr,
                "regime_ok": self.regime_ok, "maxdd": self.maxdd, "sharpe_day": self.sharpe_day, "exam": self.exam,
                "table": self.table, "metrics": self.metrics, "regimes": self.regimes, "gates": self.gates,
                "passes_gates": self.passes_gates}


def score_oos(oos_daily: pd.DataFrame, oos_trades: pd.DataFrame, labels: pd.Series | None, rules: Rules, cfg: dict,
              n_trials: int, prior_sharpes: list[float]) -> Score:
    """Score a stitched OOS daily series and its trades against the rules and the configured gates."""
    wf, gates_cfg = cfg["walkforward"], cfg["gates"]
    best, table = exam.optimise_risk(oos_daily, rules, exam.DEFAULT_GRID, wf["oos_sims"], wf["block_len"], wf["seed"])
    rets = best.risk * oos_daily["r_close"].to_numpy(float)
    dsr = deflated_sharpe(rets, n_trials, trial_sharpe_variance(prior_sharpes))
    reg = regime_table(oos_trades, labels)
    maxdd = metrics.max_drawdown(rets)
    gates = {
        "trades": len(oos_trades) >= gates_cfg["min_oos_trades"],
        "dsr": dsr >= gates_cfg["min_dsr"],
        "regime": regimes_ok(reg, gates_cfg["regime_min_trades"], gates_cfg["regime_min_expectancy"]),
        "maxdd": maxdd < gates_cfg["maxdd_frac_of_max_loss"] * rules.max_loss,
    }
    per_day_sharpe = float(rets.mean() / rets.std(ddof=1)) if rets.size > 2 and rets.std(ddof=1) > 0 else 0.0
    return Score(best.pass_combined, best.risk, len(oos_trades), dsr, gates["regime"], maxdd, per_day_sharpe,
                 best.as_dict(), [t.as_dict() for t in table], metrics.summary(rets, oos_trades), reg, gates)


def beats(candidate: Score, champion_pass: float, min_improvement: float) -> bool:
    """All gates pass and the OOS pass probability is higher than the champion's by at least min_improvement."""
    return candidate.passes_gates and candidate.oos_pass >= champion_pass + min_improvement - 1e-12
