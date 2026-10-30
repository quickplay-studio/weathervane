"""Bar-loop simulator that reproduces the TradingView broker emulator (process_orders_on_close=false).

Signals are decided on bar close t. Market orders fill at the open of t+1 with half the round-trip cost against
the trader. Stops and limits become active on the fill bar and are tested along the intrabar path
open -> high -> low -> close when |open-high| < |open-low|, else open -> low -> high -> close. A gap through
a level fills at the open. The same half cost is charged on every fill, as the Pine commission is.

R accounting: 1R is the nominal risk of the trade, stop_atr * ATR on the signal bar, which equals
|signal close - initial stop| and is what both the live sizing and the Pine quantity use.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import jit
from .signals_core import Bars, Signals, build_signals

REASONS = {0: "open", 1: "stop", 2: "target", 3: "time_stop", 4: "exit_signal", 5: "reversal",
           6: "window_end", 7: "trail"}
R_STOP, R_TARGET, R_TIME, R_SIGNAL, R_REVERSAL, R_WINDOW, R_TRAIL = 1, 2, 3, 4, 5, 6, 7

# indices into Sim.state
S_POS, S_ENTRY_I, S_ENTRY_PX, S_R0, S_STOP0, S_STOP, S_STOP_PREV, S_TARGET, S_HELD = range(9)
S_PEND_ENTRY, S_PEND_EXIT, S_PEND_STOP, S_PEND_TARGET, S_PEND_R0, S_EXTREME = range(9, 15)
STATE_LEN = 15


@jit
def _adverse(side, a, b):
    return b if side * b < side * a else a


@jit
def _intrabar(side, o, h, l, stop, target):
    """First stop/target touch along Pine's intrabar path. Returns (hit, level, worst price)."""
    has_target = not np.isnan(target)
    if side > 0:
        if o <= stop:
            return 1, o, o
        if has_target and o >= target:
            return 2, o, o
    else:
        if o >= stop:
            return 1, o, o
        if has_target and o <= target:
            return 2, o, o
    if abs(o - h) < abs(o - l):
        e1, e2 = h, l
    else:
        e1, e2 = l, h
    worst = o
    prev = o
    for leg in range(2):
        nxt = e1 if leg == 0 else e2
        lo = min(prev, nxt)
        hi = max(prev, nxt)
        if lo <= stop <= hi:
            return 1, stop, _adverse(side, worst, stop)
        if has_target and lo <= target <= hi:
            return 2, target, _adverse(side, worst, target)
        worst = _adverse(side, worst, nxt)
        prev = nxt
    return 0, np.nan, worst


@jit
def _run(o, h, l, c, atr, sig_long, sig_short, ex_long, ex_short,
         stop_mult, target_mult, trail_mult, time_stop, reverse, half, start, stop_idx):
    n = o.shape[0]
    r_close = np.zeros(n)
    r_worst = np.zeros(n)
    active = np.zeros(n, np.int8)
    tr_i = np.zeros((n + 1, 2), np.int64)
    tr_f = np.zeros((n + 1, 6))
    tr_r = np.zeros(n + 1, np.int64)
    nt = 0
    pos = 0
    entry_i = -1
    entry_px = 0.0
    r0 = 1.0
    stop = np.nan
    stop0 = np.nan
    stop_prev = np.nan
    target = np.nan
    extreme = 0.0
    mae = 0.0
    held = 0
    pend_entry = 0
    pend_exit = 0
    pend_stop = np.nan
    pend_target = np.nan
    pend_r0 = 1.0
    realized = 0.0
    e_prev = 0.0
    for i in range(n):
        carried = pos != 0
        filled = False
        worst_total = 0.0
        # market orders decided on the previous close fill at this open
        if pend_exit != 0 and pos != 0:
            px = o[i] * (1.0 - pos * half)
            mae = max(mae, pos * (entry_px - o[i]) / r0)
            pnl = pos * (px - entry_px) / r0
            realized += pnl
            tr_i[nt, 0] = entry_i
            tr_i[nt, 1] = i
            tr_f[nt, 0] = pos
            tr_f[nt, 1] = entry_px
            tr_f[nt, 2] = px
            tr_f[nt, 3] = stop0
            tr_f[nt, 4] = pnl
            tr_f[nt, 5] = mae
            tr_r[nt] = pend_exit
            nt += 1
            pos = 0
        if pend_entry != 0 and pos == 0:
            pos = pend_entry
            entry_px = o[i] * (1.0 + pos * half)
            entry_i = i
            r0 = pend_r0
            stop = pend_stop
            stop0 = pend_stop
            target = pend_target
            extreme = o[i]
            mae = 0.0
            filled = True
        pend_entry = 0
        pend_exit = 0
        # stop and target along the intrabar path
        if pos != 0:
            hit, level, worst_px = _intrabar(pos, o[i], h[i], l[i], stop, target)
            mae = max(mae, pos * (entry_px - worst_px) / r0)
            pnl_worst = pos * (worst_px - entry_px) / r0
            if hit != 0:
                px = level * (1.0 - pos * half)
                pnl = pos * (px - entry_px) / r0
                worst_total = realized + min(pnl_worst, pnl)
                realized += pnl
                reason = R_TARGET
                if hit == 1:
                    reason = R_TRAIL if stop != stop0 else R_STOP
                tr_i[nt, 0] = entry_i
                tr_i[nt, 1] = i
                tr_f[nt, 0] = pos
                tr_f[nt, 1] = entry_px
                tr_f[nt, 2] = px
                tr_f[nt, 3] = stop0
                tr_f[nt, 4] = pnl
                tr_f[nt, 5] = mae
                tr_r[nt] = reason
                nt += 1
                pos = 0
            else:
                worst_total = realized + pnl_worst
        else:
            worst_total = realized
        # close of bar: trailing stop, mark to market
        if pos != 0:
            stop_prev = stop
            extreme = max(extreme, h[i]) if pos > 0 else min(extreme, l[i])
            if trail_mult == trail_mult and atr[i] == atr[i]:
                cand = extreme - pos * trail_mult * atr[i]
                stop = max(stop, cand) if pos > 0 else min(stop, cand)
            held = i - entry_i
            e_close = realized + pos * (c[i] - entry_px) / r0
        else:
            e_close = realized
        r_close[i] = e_close - e_prev
        r_worst[i] = min(worst_total - e_prev, r_close[i])
        e_prev = e_close
        if carried or filled:
            active[i] = 1
        # signals on this close become orders for the next open
        sig = 0
        if i >= start and i < stop_idx - 1:
            if sig_long[i]:
                sig = 1
            elif sig_short[i]:
                sig = -1
        if pos != 0:
            reason = 0
            if i == stop_idx - 1:
                reason = R_WINDOW
            elif time_stop > 0 and held >= time_stop:
                reason = R_TIME
            elif (pos > 0 and ex_long[i]) or (pos < 0 and ex_short[i]):
                reason = R_SIGNAL
            if sig == -pos and reverse:
                reason = R_REVERSAL
                pend_entry = sig
            pend_exit = reason
        elif sig != 0:
            pend_entry = sig
        if pend_entry != 0:
            pend_r0 = stop_mult * atr[i]
            pend_stop = c[i] - pend_entry * pend_r0
            pend_target = c[i] + pend_entry * target_mult * atr[i]
    state = np.zeros(STATE_LEN)
    state[S_POS] = pos
    state[S_ENTRY_I] = entry_i if pos != 0 else -1
    state[S_ENTRY_PX] = entry_px
    state[S_R0] = r0
    state[S_STOP0] = stop0
    state[S_STOP] = stop
    state[S_STOP_PREV] = stop_prev
    state[S_TARGET] = target
    state[S_HELD] = held
    state[S_PEND_ENTRY] = pend_entry
    state[S_PEND_EXIT] = pend_exit
    state[S_PEND_STOP] = pend_stop
    state[S_PEND_TARGET] = pend_target
    state[S_PEND_R0] = pend_r0
    state[S_EXTREME] = extreme
    return r_close, r_worst, active, tr_i[:nt], tr_f[:nt], tr_r[:nt], state


@dataclass
class Sim:
    """Result of one sleeve on one instrument."""

    instrument: str
    sleeve: str
    dates: np.ndarray        # datetime64[ns]
    r_close: np.ndarray
    r_worst: np.ndarray
    active: np.ndarray
    trade_i: np.ndarray      # (n, 2) entry and exit bar index
    trade_f: np.ndarray      # (n, 6) side, entry px, exit px, initial stop, R multiple, MAE in R
    trade_reason: np.ndarray
    state: np.ndarray

    @property
    def entry_days(self) -> np.ndarray:
        return self.dates[self.trade_i[:, 0]] if len(self.trade_i) else self.dates[:0]

    def trades_frame(self) -> pd.DataFrame:
        """One row per closed trade. Prices include the half cost charged on each fill."""
        if not len(self.trade_i):
            return pd.DataFrame(columns=TRADE_COLUMNS)
        side = self.trade_f[:, 0]
        return pd.DataFrame({
            "instrument": self.instrument, "sleeve": self.sleeve,
            "side": np.where(side > 0, "long", "short"),
            "entry_date": pd.to_datetime(self.dates[self.trade_i[:, 0]]),
            "exit_date": pd.to_datetime(self.dates[self.trade_i[:, 1]]),
            "entry_price": self.trade_f[:, 1], "exit_price": self.trade_f[:, 2], "stop": self.trade_f[:, 3],
            "r": self.trade_f[:, 4], "bars": self.trade_i[:, 1] - self.trade_i[:, 0], "mae_r": self.trade_f[:, 5],
            "reason": [REASONS[int(r)] for r in self.trade_reason],
        })


TRADE_COLUMNS = ["instrument", "sleeve", "side", "entry_date", "exit_date", "entry_price", "exit_price", "stop",
                 "r", "bars", "mae_r", "reason"]


def _flag(v: float | None) -> float:
    return np.nan if v is None else float(v)


def simulate(sleeve: dict, bars: Bars, half_cost: float, instrument: str = "", signals: Signals | None = None,
             start: int = 0, stop_idx: int | None = None) -> Sim:
    """Run one sleeve (parameters already applied) on one instrument.

    start: signals before this bar index are ignored (account starts flat there).
    stop_idx: the sleeve stops opening positions at this bar and flattens at its open (None = never).
    """
    sig = signals if signals is not None else build_signals(sleeve, bars)
    ex = sleeve["exit"]
    out = _run(bars.open, bars.high, bars.low, bars.close, sig.atr,
               sig.long, sig.short, sig.exit_long, sig.exit_short,
               float(ex["stop_atr"]), _flag(ex.get("target_atr")), _flag(ex.get("trail_atr")),
               int(ex.get("time_stop_bars") or 0), bool(ex.get("reverse_on_opposite", True)),
               float(half_cost), int(start), len(bars) + 1 if stop_idx is None else int(stop_idx))
    r_close, r_worst, active, ti, tf, trs, state = out
    return Sim(instrument, sleeve["id"], bars.index.to_numpy("datetime64[ns]"), r_close, r_worst, active,
               ti, tf, trs, state)


def master_grid(index_list: list[pd.DatetimeIndex]) -> np.ndarray:
    """Sorted union of all instrument dates as datetime64[ns]."""
    union = index_list[0]
    for idx in index_list[1:]:
        union = union.union(idx)
    return union.to_numpy("datetime64[ns]")


def aggregate(sims: list[Sim], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum r_close and r_worst over sims on the master grid (missing days are 0); active is the union."""
    rc, rw = np.zeros(grid.shape[0]), np.zeros(grid.shape[0])
    act = np.zeros(grid.shape[0], np.int8)
    for s in sims:
        pos = np.searchsorted(grid, s.dates)
        rc[pos] += s.r_close
        rw[pos] += s.r_worst
        act[pos] = np.maximum(act[pos], s.active)
    return rc, rw, act


def daily_frame(grid: np.ndarray, rc: np.ndarray, rw: np.ndarray, act: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"r_close": rc, "r_worst": rw, "active": act}, index=pd.DatetimeIndex(grid, name="date"))
