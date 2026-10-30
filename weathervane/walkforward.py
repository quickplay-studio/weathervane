"""Walk-forward evaluation of a candidate portfolio: per-fold optimisation, stitched out-of-sample series."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import exam, search, spec
from .config import Instrument, Rules
from .signals_core import Bars
from .simulator import TRADE_COLUMNS, Sim, daily_frame, master_grid, simulate

MIN_TRADES_FOR_FULL_SCORE = 30
SHARPE_TIEBREAK = 0.01
INVALID = -1.0


@dataclass
class SleeveRun:
    """One sleeve (or a sum of sleeves) on the master grid, without the per-instrument detail."""

    rc: np.ndarray
    rw: np.ndarray
    act: np.ndarray
    entry_ns: np.ndarray

    def entries_between(self, t0: np.datetime64, t1: np.datetime64) -> int:
        return int(np.count_nonzero((self.entry_ns >= t0) & (self.entry_ns < t1)))


def sum_runs(runs: list[SleeveRun]) -> SleeveRun:
    return SleeveRun(sum(r.rc for r in runs), sum(r.rw for r in runs), np.maximum.reduce([r.act for r in runs]),
                     np.sort(np.concatenate([r.entry_ns for r in runs])))


class Book:
    """Bars for a set of instruments on one master date grid, with an LRU cache of sleeve simulations."""

    def __init__(self, data: dict[str, pd.DataFrame], universe: dict[str, Instrument], cache_size: int = 256):
        self.bars = {i: Bars(df) for i, df in data.items()}
        self.half = {i: universe[i].half_cost for i in data}
        self.grid = master_grid([b.index for b in self.bars.values()])
        self.pos = {i: np.searchsorted(self.grid, b.index.to_numpy("datetime64[ns]")) for i, b in self.bars.items()}
        self._cache: OrderedDict[str, SleeveRun] = OrderedDict()
        self._size = cache_size

    def sims(self, sleeve: dict) -> list[Sim]:
        return [simulate(sleeve, self.bars[i], self.half[i], i) for i in sleeve["instruments"] if i in self.bars]

    def run_sleeve(self, sleeve: dict) -> SleeveRun:
        key = spec.sleeve_hash(sleeve)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        n = self.grid.shape[0]
        rc, rw, act, entries = np.zeros(n), np.zeros(n), np.zeros(n, np.int8), []
        for sim in self.sims(sleeve):
            p = self.pos[sim.instrument]
            rc[p] += sim.r_close
            rw[p] += sim.r_worst
            act[p] = np.maximum(act[p], sim.active)
            entries.append(sim.entry_days)
        run = SleeveRun(rc, rw, act, np.sort(np.concatenate(entries)) if entries else np.zeros(0, "datetime64[ns]"))
        self._cache[key] = run
        if len(self._cache) > self._size:
            self._cache.popitem(last=False)
        return run

    def run(self, sleeves: list[dict]) -> SleeveRun:
        return sum_runs([self.run_sleeve(s) for s in sleeves])

    def trades(self, sleeves: list[dict], t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
        """Closed trades that entered inside [t0, t1)."""
        frames = [sim.trades_frame() for s in sleeves for sim in self.sims(s)]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return empty_trades()
        out = pd.concat(frames, ignore_index=True)
        return out[(out["entry_date"] >= t0) & (out["entry_date"] < t1)].reset_index(drop=True)

    def daily(self, run: SleeveRun, lo: int = 0, hi: int | None = None) -> pd.DataFrame:
        sl = slice(lo, hi)
        return daily_frame(self.grid[sl], run.rc[sl], run.rw[sl], run.act[sl])

    def window(self, t0: pd.Timestamp, t1: pd.Timestamp) -> tuple[int, int]:
        return (int(np.searchsorted(self.grid, np.datetime64(t0), "left")),
                int(np.searchsorted(self.grid, np.datetime64(t1), "left")))


def empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_COLUMNS)


def split_vault(data: dict[str, pd.DataFrame], vault_days: int) -> tuple[dict[str, pd.DataFrame], pd.Timestamp | None]:
    """Remove the last vault_days bars (by the union calendar) from every instrument."""
    grid = master_grid([df.index for df in data.values()])
    if vault_days <= 0 or grid.shape[0] <= vault_days:
        return data, None
    start = pd.Timestamp(grid[-vault_days])
    return {i: df[df.index < start] for i, df in data.items() if (df.index < start).any()}, start


@dataclass(frozen=True)
class Fold:
    year: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(grid: np.ndarray, wf: dict) -> list[Fold]:
    """Rolling folds: train_years of history, then test_years, stepping one year, from first_test_year on."""
    last = pd.Timestamp(grid[-1])
    folds = []
    year = wf["first_test_year"]
    while pd.Timestamp(year=year, month=1, day=1) <= last:
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = min(pd.Timestamp(year=year + wf["test_years"], month=1, day=1), last + pd.Timedelta(days=1))
        train_start = pd.Timestamp(year=year - wf["train_years"], month=1, day=1)
        n_train = int(np.count_nonzero((grid >= np.datetime64(train_start)) & (grid < np.datetime64(test_start))))
        n_test = int(np.count_nonzero((grid >= np.datetime64(test_start)) & (grid < np.datetime64(test_end))))
        if n_train >= 0.75 * wf["train_years"] * 252 and n_test >= wf["min_test_bars"]:
            folds.append(Fold(year, train_start, test_start, test_start, test_end))
        year += 1
    return folds


def train_objective(book: Book, sleeves: list[dict], window: tuple[int, int], span: tuple[pd.Timestamp, pd.Timestamp],
                    rules: Rules, cfg: dict) -> float:
    """Exam pass probability on the train window at its best risk, scaled down below 30 trades.

    A small Sharpe term breaks ties between parameter sets that all have a zero pass probability.
    """
    wf = cfg["walkforward"]
    run = book.run(sleeves)
    lo, hi = window
    daily = (run.rc[lo:hi], run.rw[lo:hi], run.act[lo:hi])
    n_trades = run.entries_between(np.datetime64(span[0]), np.datetime64(span[1]))
    grid = exam.allowed_grid(rules, wf["train_risk_grid"])
    best, _ = exam.optimise_risk(daily, rules, grid, wf["train_sims"], wf["block_len"], wf["seed"])
    rc = daily[0]
    sd = rc.std()
    sharpe = float(rc.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    score = best.pass_combined * min(1.0, n_trades / MIN_TRADES_FOR_FULL_SCORE)
    return score + SHARPE_TIEBREAK * float(np.clip(sharpe / 3.0, -1.0, 1.0))


def optimise_window(book: Book, sleeves: list[dict], span: tuple[pd.Timestamp, pd.Timestamp], rules: Rules, cfg: dict,
                    seed: int, deadline: float | None) -> search.SearchResult:
    """Optuna over the param_spaces, scored on the bars in [span[0], span[1])."""
    window = book.window(*span)

    def objective(params: dict[str, dict]) -> float:
        applied = [spec.apply_params(s, params[s["id"]]) for s in sleeves]
        try:
            for s in applied:
                spec.validate_sleeve(s)
        except spec.SpecError:
            return INVALID
        return train_objective(book, applied, window, span, rules, cfg)

    return search.optimise(sleeves, objective, cfg["walkforward"]["n_trials"], seed, deadline)


@dataclass
class Evaluation:
    oos_daily: pd.DataFrame
    oos_trades: pd.DataFrame
    folds: list[dict]
    live_params: dict[str, dict]
    runtime_s: float


def evaluate(pf: dict, book: Book, rules: Rules, cfg: dict, deadline: float | None = None) -> Evaluation:
    """Walk-forward evaluation: optimise on each train window, trade the next test window out of sample."""
    t_start = time.monotonic()
    wf = cfg["walkforward"]
    sleeves = [spec.normalise(s) for s in pf["sleeves"]]
    folds, daily_parts, trade_parts, info = make_folds(book.grid, wf), [], [], []
    for fold in folds:
        res = optimise_window(book, sleeves, (fold.train_start, fold.train_end), rules, cfg, wf["seed"] + fold.year, deadline)
        applied = [spec.apply_params(s, res.params[s["id"]]) for s in sleeves]
        lo, hi = book.window(fold.test_start, fold.test_end)
        daily_parts.append(book.daily(book.run(applied), lo, hi))
        trade_parts.append(book.trades(applied, fold.test_start, fold.test_end))
        info.append({"year": fold.year, "train_score": round(res.value, 4), "params": res.params})
    live = live_params(sleeves, book, rules, cfg, deadline)
    trades = [t for t in trade_parts if not t.empty]
    return Evaluation(
        oos_daily=pd.concat(daily_parts) if daily_parts else daily_frame(book.grid[:0], np.zeros(0), np.zeros(0), np.zeros(0, np.int8)),
        oos_trades=pd.concat(trades, ignore_index=True) if trades else empty_trades(),
        folds=info, live_params=live, runtime_s=time.monotonic() - t_start)


def live_params(sleeves: list[dict], book: Book, rules: Rules, cfg: dict, deadline: float | None) -> dict[str, dict]:
    """Parameters for live use: one more optimisation on the last train_years before the vault."""
    wf = cfg["walkforward"]
    end = pd.Timestamp(book.grid[-1]) + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=wf["train_years"])
    return optimise_window(book, sleeves, (start, end), rules, cfg, wf["seed"] + 9999, deadline).params
