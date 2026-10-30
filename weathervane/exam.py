"""Prop-firm exam Monte Carlo on a portfolio's daily R series.

Whole days are resampled with a stationary bootstrap (keeps cross-asset correlation inside a day). Each simulated
path plays the phases in order against the rules. Everything is vectorised over simulations and risk levels.

Approximations, stated once:
- equity equals balance at every close, so the daily-loss reference is the previous close equity;
- intraday_trailing uses equity + risk * max(0, r_close) as the day's best equity, which is a proxy;
- time_limit_days counts calendar days, so it is converted to 5/7 as many bootstrapped trading days.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Rules

DEFAULT_GRID = tuple(round(0.0025 * k, 4) for k in range(1, 9))
CAP_NO_LIMIT = 500
EPS = 1e-9
FAIL_NAMES = {1: "daily", 2: "max_loss", 3: "time", 4: "timeout"}


@dataclass
class ExamResult:
    risk: float
    pass_phase: list[float]          # P(pass phase k | reached phase k)
    pass_combined: float             # P(pass every phase)
    days_median: float               # trading days to pass everything, over passing paths
    days_p90: float
    fail_reasons: dict[str, float]   # share of all simulations ending in each failure
    n_sims: int

    @property
    def pass_p1(self) -> float:
        return self.pass_phase[0]

    @property
    def pass_p2(self) -> float | None:
        return self.pass_phase[1] if len(self.pass_phase) > 1 else None

    def as_dict(self) -> dict:
        return {"risk": self.risk, "pass_p1": self.pass_p1, "pass_p2": self.pass_p2, "pass_combined": self.pass_combined,
                "days_median": self.days_median, "days_p90": self.days_p90, "fail_reasons": self.fail_reasons,
                "n_sims": self.n_sims}


@dataclass
class State:
    """Per-path exam state for the current phase."""

    eq: np.ndarray
    hwm: np.ndarray
    days: np.ndarray
    best: np.ndarray

    @classmethod
    def fresh(cls, n: int) -> "State":
        return cls(np.ones(n), np.ones(n), np.zeros(n, np.int32), np.zeros(n))

    def take(self, keep: np.ndarray) -> "State":
        return State(self.eq[keep], self.hwm[keep], self.days[keep], self.best[keep])


def step(st: State, x: np.ndarray, worst: np.ndarray, active: np.ndarray, rules: Rules, target: float,
         min_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Advance every path one day. x and worst are the day's close and worst P&L as account fractions.

    Returns (fail_code, win): fail_code 0 none, 1 daily loss, 2 max loss. Breach checks come before the pass check.
    """
    low = st.eq + worst
    daily = low <= st.eq - rules.daily_loss
    if rules.max_loss_type == "static":
        floor = 1.0 - rules.max_loss
    else:
        top = st.hwm if rules.max_loss_type == "eod_trailing" else np.maximum(st.hwm, st.eq + np.maximum(x, 0.0))
        floor = top - rules.max_loss
        if rules.lock_at_start:
            floor = np.minimum(floor, 1.0)
    loss = low <= floor
    eq_new = st.eq + x
    st.days = st.days + active
    st.best = np.maximum(st.best, x)
    profit = eq_new - 1.0
    ok = (profit >= target - EPS) & (st.days >= min_days)
    if rules.consistency is not None:
        ok = ok & (st.best <= rules.consistency * profit + EPS)
    st.hwm = np.maximum(st.hwm, eq_new)
    if rules.max_loss_type == "intraday_trailing":
        st.hwm = np.maximum(st.hwm, st.eq + np.maximum(x, 0.0))
    st.eq = eq_new
    fail = np.where(daily, 1, np.where(loss, 2, 0)).astype(np.int8)
    return fail, ok & (fail == 0)


def phase_cap(rules: Rules) -> int:
    if rules.time_limit_days:
        return max(1, int(np.ceil(rules.time_limit_days * 5 / 7)))
    return CAP_NO_LIMIT


def stationary_bootstrap(n_days: int, n_sims: int, length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Index matrix (n_sims, length) of a stationary bootstrap with mean block length `block`."""
    new = rng.random((n_sims, length)) < 1.0 / block
    new[:, 0] = True
    t = np.arange(length, dtype=np.int32)
    last_new = np.maximum.accumulate(np.where(new, t, np.int32(0)), axis=1)
    starts = rng.integers(0, n_days, size=(n_sims, length), dtype=np.int32)
    base = np.take_along_axis(starts, last_new, axis=1)
    return ((base + (t - last_new)) % n_days).astype(np.int32)


@dataclass
class _Paths:
    rc: np.ndarray
    rw: np.ndarray
    ac: np.ndarray


@dataclass
class _PhaseOut:
    passed: np.ndarray
    steps: np.ndarray
    reason: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))


def _run_phase(paths: _Paths, sim: np.ndarray, risk: np.ndarray, off: np.ndarray, rules: Rules, phase_i: int,
               cap: int) -> _PhaseOut:
    n = sim.shape[0]
    ph = rules.phases[phase_i]
    passed, steps = np.zeros(n, bool), np.full(n, cap, np.int32)
    reason = np.zeros(n, np.int8)
    ids, done = np.arange(n), np.zeros(n, bool)
    st = State.fresh(n)
    for t in range(cap):
        cols = off + t
        x = risk * paths.rc[sim, cols]
        worst = risk * paths.rw[sim, cols]
        fail, win = step(st, x, worst, paths.ac[sim, cols], rules, ph.target, ph.min_days)
        fail = np.where(done, 0, fail)
        win = win & ~done
        fin = (fail > 0) | win
        if fin.any():
            idx = ids[fin]
            passed[idx], steps[idx], reason[idx] = win[fin], t + 1, fail[fin]
            done |= fin
            if done.sum() * 8 >= done.size:
                keep = ~done
                if not keep.any():
                    return _PhaseOut(passed, steps, reason)
                sim, risk, off, ids, st, done = sim[keep], risk[keep], off[keep], ids[keep], st.take(keep), done[keep]
    reason[ids[~done]] = 3 if rules.time_limit_days else 4
    return _PhaseOut(passed, steps, reason)


def _summarise(risks: np.ndarray, grid: list[float], n_sims: int, passes: list[np.ndarray], reached: list[np.ndarray],
               days: np.ndarray, fails: np.ndarray) -> list[ExamResult]:
    out = []
    for k, r in enumerate(grid):
        sl = slice(k * n_sims, (k + 1) * n_sims)
        p_phase = [float(p[sl].sum() / max(1, rc[sl].sum())) for p, rc in zip(passes, reached)]
        all_pass = passes[-1][sl]
        d = days[sl][all_pass]
        out.append(ExamResult(
            risk=float(r), pass_phase=p_phase, pass_combined=float(all_pass.mean()),
            days_median=float(np.median(d)) if d.size else float("nan"),
            days_p90=float(np.percentile(d, 90)) if d.size else float("nan"),
            fail_reasons={name: float((fails[sl] == code).mean()) for code, name in FAIL_NAMES.items()},
            n_sims=n_sims))
    return out


def _arrays(daily: pd.DataFrame | tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(daily, pd.DataFrame):
        return (daily["r_close"].to_numpy(float), daily["r_worst"].to_numpy(float), daily["active"].to_numpy(float))
    return tuple(np.asarray(a, float) for a in daily)  # type: ignore[return-value]


def run_exam(daily: pd.DataFrame | tuple, rules: Rules, risks: list[float] | tuple[float, ...] = DEFAULT_GRID,
             n_sims: int = 5000, block: int = 10, seed: int = 0) -> list[ExamResult]:
    """Monte Carlo the exam at every risk fraction in `risks` on shared bootstrap paths."""
    rc, rw, ac = _arrays(daily)
    grid = list(risks)
    n_ph = len(rules.phases)
    if rc.size == 0 or not np.any(rc) or not np.any(ac):
        empty = ExamResult(0.0, [0.0] * n_ph, 0.0, float("nan"), float("nan"), {n: 0.0 for n in FAIL_NAMES.values()}, n_sims)
        return [ExamResult(**{**empty.__dict__, "risk": float(r)}) for r in grid]
    cap = phase_cap(rules)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap(rc.size, n_sims, cap * n_ph, block, rng)
    paths = _Paths(rc[idx], rw[idx], ac[idx])
    total = len(grid) * n_sims
    sim = np.tile(np.arange(n_sims), len(grid))
    risk = np.repeat(np.asarray(grid, float), n_sims)
    alive, off = np.ones(total, bool), np.zeros(total, np.int64)
    days, fails = np.zeros(total, np.int64), np.zeros(total, np.int8)
    passes, reached = [], []
    for p in range(n_ph):
        sel = np.flatnonzero(alive)
        res = _run_phase(paths, sim[sel], risk[sel], off[sel], rules, p, cap)
        reached_mask = np.zeros(total, bool)
        reached_mask[sel] = True
        pass_mask = np.zeros(total, bool)
        pass_mask[sel] = res.passed
        passes.append(pass_mask)
        reached.append(reached_mask)
        failed = sel[~res.passed]
        fails[failed] = res.reason[~res.passed]
        days[sel] += res.steps
        off[sel] += res.steps
        alive = pass_mask
    return _summarise(risk, grid, n_sims, passes, reached, days, fails)


def allowed_grid(rules: Rules, grid: tuple[float, ...] | list[float]) -> list[float]:
    """Risk grid capped by the rules' per-trade risk limit."""
    if rules.max_risk is None:
        return list(grid)
    return sorted({min(g, rules.max_risk) for g in grid})


def optimise_risk(daily: pd.DataFrame | tuple, rules: Rules, grid: tuple[float, ...] | list[float] = DEFAULT_GRID,
                  n_sims: int = 5000, block: int = 10, seed: int = 0) -> tuple[ExamResult, list[ExamResult]]:
    """Risk fraction with the highest combined pass probability (ties go to the lower risk) and the full table."""
    table = run_exam(daily, rules, allowed_grid(rules, grid), n_sims, block, seed)
    best = max(table, key=lambda e: (round(e.pass_combined, 12), -e.risk))
    return best, table


class ExamTracker:
    """Plays real days, one at a time, through the exam rules. Used by the paper ledger.

    A breach ends the attempt and a new one starts the next day. Passing the last phase also starts a new attempt.
    """

    def __init__(self, rules: Rules):
        self.rules = rules
        self.phase = 0
        self.attempt = 1
        self.events: list[dict] = []
        self.st = State.fresh(1)

    def day(self, date: str, x: float, worst: float, active: int) -> dict:
        """x and worst are the day's close and worst P&L as account fractions."""
        ph = self.rules.phases[self.phase]
        fail, win = step(self.st, np.array([x]), np.array([worst]), np.array([active]), self.rules, ph.target, ph.min_days)
        out = {"phase": self.phase + 1, "attempt": self.attempt, "equity": float(self.st.eq[0]),
               "trading_days": int(self.st.days[0]), "event": ""}
        if fail[0]:
            out["event"] = f"breach_{FAIL_NAMES[int(fail[0])]}"
            self._record(date, out)
            self._restart(0, new_attempt=True)
        elif win[0]:
            out["event"] = "passed_all" if self.phase + 1 == len(self.rules.phases) else f"passed_phase_{self.phase + 1}"
            self._record(date, out)
            last = self.phase + 1 == len(self.rules.phases)
            self._restart(0 if last else self.phase + 1, new_attempt=last)
        return out

    def _record(self, date: str, out: dict) -> None:
        self.events.append({"date": date, "event": out["event"], "phase": out["phase"], "attempt": out["attempt"],
                            "equity": out["equity"]})

    def _restart(self, phase: int, new_attempt: bool) -> None:
        self.phase = phase
        self.attempt += int(new_attempt)
        self.st = State.fresh(1)

    def snapshot(self) -> dict:
        return {"phase": self.phase + 1, "attempt": self.attempt, "equity": float(self.st.eq[0]),
                "trading_days": int(self.st.days[0]), "events": self.events}
