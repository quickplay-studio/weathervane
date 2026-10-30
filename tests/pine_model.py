"""A Python transcription of the order logic in the generated Pine script, run against a small TradingView broker emulator.

It exists because Pine cannot be executed here. The per-bar block in `script_at_close` mirrors the text that
weathervane/pine.py emits (state vars stopLvl, tgtLvl, extreme; trailing; exitNow; isReverse; isEntry; strategy.entry,
strategy.exit, strategy.close_all) and the broker rules are the documented ones for process_orders_on_close=false:
market orders fill at the next open, exit orders belong to an entry id and are tested along the intrabar path.
The tests compare its trades with the simulator's.
"""
from __future__ import annotations

import numpy as np

from weathervane.signals_core import Bars, Signals
from weathervane.simulator import _intrabar


def run(sleeve: dict, bars: Bars, sig: Signals, half: float) -> list[dict]:
    o, h, l, c, atr = bars.open, bars.high, bars.low, bars.close, sig.atr
    ex = sleeve["exit"]
    stop_atr, target_atr, trail = ex["stop_atr"], ex.get("target_atr"), ex.get("trail_atr")
    tstop, reverse = ex.get("time_stop_bars"), ex.get("reverse_on_opposite", True)
    n = len(c)
    pos, entry_px, entry_bar, r0 = 0, 0.0, -1, 1.0
    stop_lvl, tgt_lvl, extreme = np.nan, np.nan, np.nan
    pending: list[tuple] = []
    exit_orders: dict[str, tuple[float, float]] = {}
    trades: list[dict] = []

    def close_at(i: int, price: float, how: str) -> None:
        nonlocal pos
        px = price * (1.0 - pos * half)
        trades.append({"side": pos, "entry_bar": entry_bar, "exit_bar": i, "entry": entry_px, "exit": px,
                       "r": pos * (px - entry_px) / r0, "how": how})
        exit_orders.pop("L" if pos > 0 else "S", None)
        pos = 0

    for i in range(n):
        # broker: market orders from the previous close fill at this open
        for order in pending:
            if order[0] == "close_all" and pos != 0:
                close_at(i, o[i], "close_all")
            elif order[0] == "entry":
                side = order[1]
                if pos != 0 and pos != side:
                    close_at(i, o[i], "reversal")
                if pos == 0:
                    pos, entry_px, entry_bar, r0 = side, o[i] * (1.0 + side * half), i, order[2]
        pending = []
        # broker: exit orders of the open position along the intrabar path
        if pos != 0:
            key = "L" if pos > 0 else "S"
            if key in exit_orders:
                stop, limit = exit_orders[key]
                hit, level, _ = _intrabar(pos, o[i], h[i], l[i], stop, limit)
                if hit:
                    close_at(i, level, "stop" if hit == 1 else "limit")
        # the script runs on the close of the bar
        in_pos = pos != 0
        side = 1 if pos > 0 else -1 if pos < 0 else 0
        bars_held = i - entry_bar if in_pos else 0
        if in_pos:
            extreme = (h[i] if side > 0 else l[i]) if bars_held == 0 else (max(extreme, h[i]) if side > 0 else min(extreme, l[i]))
            if trail is not None and not np.isnan(atr[i]):
                cand = extreme - side * trail * atr[i]
                stop_lvl = max(stop_lvl, cand) if side > 0 else min(stop_lvl, cand)
        s = 1 if sig.long[i] else -1 if sig.short[i] else 0
        stop_dist = stop_atr * atr[i]
        x_now = in_pos and ((tstop is not None and bars_held >= tstop) or (sig.exit_long[i] if side > 0 else sig.exit_short[i]))
        is_reverse = in_pos and reverse and s == -side
        is_entry = is_reverse or (not in_pos and s != 0)
        if is_entry:
            stop_lvl = c[i] - s * stop_dist
            tgt_lvl = c[i] + s * target_atr * atr[i] if target_atr is not None else np.nan
            pending.append(("entry", s, stop_dist))
            exit_orders["L" if s > 0 else "S"] = (stop_lvl, tgt_lvl)
        elif x_now:
            pending.append(("close_all",))
        elif in_pos:
            exit_orders["L" if side > 0 else "S"] = (stop_lvl, tgt_lvl)
    return trades
