"""Live signals: run each champion sleeve on full history, read the state at the last completed bar, size and guard."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, simulator as sm, spec
from .config import Instrument, Rules
from .signals_core import Bars, Signals, build_signals

QUIET_ACTIONS = ("HOLD", "FLAT")
HISTORY_COLUMNS = ["data_date", "portfolio", "sleeve", "instrument", "action", "side", "entry", "stop", "target", "units",
                   "risk_cash", "reason"]
EXIT_TEXT = {sm.R_TIME: "time stop reached", sm.R_SIGNAL: "exit signal fired", sm.R_WINDOW: "champion changed",
             sm.R_REVERSAL: "opposite signal"}


@dataclass
class LiveRun:
    """One sleeve on one instrument, simulated through the last completed bar."""

    sleeve: dict
    instrument: Instrument
    bars: Bars
    signals: Signals
    sim: sm.Sim


def live_sleeves(champion: dict) -> list[dict]:
    """The champion's sleeves with their live parameters written in."""
    params = champion.get("live_params") or {}
    return [spec.apply_params(spec.normalise(s), params.get(s["id"], {})) for s in champion["sleeves"]]


def run_live(champion: dict, data: dict[str, pd.DataFrame], universe: dict[str, Instrument]) -> list[LiveRun]:
    bars = {i: Bars(df) for i, df in data.items()}
    runs = []
    for sleeve in live_sleeves(champion):
        for inst_id in sleeve["instruments"]:
            if inst_id not in bars:
                continue
            sig = build_signals(sleeve, bars[inst_id])
            sim = sm.simulate(sleeve, bars[inst_id], universe[inst_id].half_cost, inst_id, sig)
            runs.append(LiveRun(sleeve, universe[inst_id], bars[inst_id], sig, sim))
    return runs


def action_from_state(state: np.ndarray) -> tuple[str, str]:
    """(action for the next open, reason text) from the simulator's end state."""
    pos, pend_entry, pend_exit = int(state[sm.S_POS]), int(state[sm.S_PEND_ENTRY]), int(state[sm.S_PEND_EXIT])
    if pos == 0:
        return ("ENTER_LONG" if pend_entry > 0 else "ENTER_SHORT", "entry signal") if pend_entry else ("FLAT", "no position")
    if pend_entry:
        return ("REVERSE_TO_LONG" if pend_entry > 0 else "REVERSE_TO_SHORT", "opposite entry signal")
    if pend_exit:
        return "EXIT", EXIT_TEXT.get(pend_exit, "exit")
    if state[sm.S_STOP] != state[sm.S_STOP_PREV]:
        return "MOVE_STOP", "trailing stop ratcheted"
    return "HOLD", "position open"


def _num(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def describe_entry(sleeve: dict) -> str:
    e = sleeve["entry"]
    args = ", ".join(f"{k}={v}" for k, v in e.items() if k != "type")
    filters = "".join(f" + {f['type']}" for f in sleeve["filters"])
    return f"{e['type']}({args}){filters}"


def row_for(run: LiveRun, portfolio: str, risk_fraction: float, account: float) -> dict:
    """The signal row for one sleeve and instrument at the last bar, sized before the risk guard."""
    st, sim, bars = run.sim.state, run.sim, run.bars
    action, reason = action_from_state(st)
    held = int(st[sm.S_POS])
    opening = action.startswith(("ENTER", "REVERSE"))
    side_n = int(st[sm.S_PEND_ENTRY]) if opening else held
    close = float(bars.close[-1])
    row = {"portfolio": portfolio, "sleeve": run.sleeve["id"], "instrument": run.instrument.id, "tv": run.instrument.tv,
           "action": action, "side": {1: "long", -1: "short", 0: ""}[side_n], "close": close,
           "atr": float(run.signals.atr[-1]), "last_bar": str(bars.index[-1].date()), "reason": reason,
           "entry_type": describe_entry(run.sleeve), "contract_note": run.instrument.contract_note,
           "bars_in_trade": int(st[sm.S_HELD]) if held and not opening else 0,
           "entry_date": None, "entry_price": None, "stop_initial": None, "target": None, "stop": None,
           "stop_distance": None, "risk_fraction": risk_fraction, "guard_scale": 1.0, "guard_scaled": False}
    if opening:
        row.update(entry_price=close, stop=float(st[sm.S_PEND_STOP]), stop_initial=float(st[sm.S_PEND_STOP]),
                   target=_num(st[sm.S_PEND_TARGET]), stop_distance=float(st[sm.S_PEND_R0]))
    elif held:
        row.update(entry_date=str(bars.index[int(st[sm.S_ENTRY_I])].date()), entry_price=float(st[sm.S_ENTRY_PX]),
                   stop=float(st[sm.S_STOP]), stop_initial=float(st[sm.S_STOP0]), target=_num(st[sm.S_TARGET]),
                   stop_distance=float(st[sm.S_R0]))
    return size_row(row, account)


def size_row(row: dict, account: float) -> dict:
    """Fill risk cash and units from the risk fraction and guard scale."""
    if row["stop_distance"]:
        row["risk_cash"] = round(account * row["risk_fraction"] * row["guard_scale"], 2)
        row["units"] = row["risk_cash"] / row["stop_distance"]
    else:
        row["risk_cash"], row["units"] = 0.0, 0.0
    return row


def open_risk(row: dict) -> float:
    """Cash lost if the stop is hit from the last close, for a position that stays open."""
    if row["action"] not in ("HOLD", "MOVE_STOP") or not row["units"]:
        return 0.0
    sign = 1 if row["side"] == "long" else -1
    return row["units"] * max(0.0, sign * (row["close"] - row["stop"]))


def apply_risk_guard(rows: list[dict], rules: Rules, account: float) -> None:
    """Scale new entries down so total open risk stays within risk_guard of the daily loss limit."""
    budget = rules.risk_guard * rules.daily_loss * account
    existing = sum(open_risk(r) for r in rows)
    new = [r for r in rows if r["action"].startswith(("ENTER", "REVERSE"))]
    wanted = sum(r["risk_cash"] for r in new)
    if wanted <= 0 or existing + wanted <= budget:
        return
    scale = max(0.0, (budget - existing) / wanted)
    for r in new:
        r["guard_scale"], r["guard_scaled"] = round(scale, 4), True
        r["reason"] += f"; scaled to {scale:.0%} by risk guard"
        size_row(r, account)


def signal_rows(pid: str, champion: dict, data: dict[str, pd.DataFrame], universe: dict[str, Instrument], rules: Rules,
                cfg: dict) -> tuple[list[dict], float]:
    """All rows for one portfolio after sizing and the risk guard, and the risk fraction used."""
    risk = config.live_risk(champion, cfg, rules)
    account = rules.account_size
    rows = [row_for(run, pid, risk, account) for run in run_live(champion, data, universe)]
    apply_risk_guard(rows, rules, account)
    rows.sort(key=lambda r: (r["action"] in QUIET_ACTIONS, r["instrument"], r["sleeve"]))
    return rows, risk


def fmt_price(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.5f}" if abs(x) < 5 else f"{x:.3f}" if abs(x) < 200 else f"{x:.2f}"


def fmt_units(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:.2f}"


def markdown(payload: dict) -> str:
    """signals/latest.md: one table per portfolio."""
    lines = [f"# Weathervane signals", "", f"Data to {payload['data_date']}. Generated {payload['generated']}.",
             "Orders are decided on the last close and fill at the next open. Paper trading only.", ""]
    for pid, p in payload["portfolios"].items():
        lines += [f"## {pid}", "", f"Champion {p['champion_id']}. Risk per trade {p['risk_fraction']:.2%} of {p['account_size']:,.0f}. "
                  f"Open risk guard {p['risk_budget']:,.0f}.", "",
                  "| Instrument | TradingView | Action | Side | Entry | Stop | Target | Units | Risk | Reason |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in p["rows"]:
            entry = r["entry_price"]
            lines.append(f"| {r['instrument']} | {r['tv']} | {r['action']} | {r['side']} | {fmt_price(entry)} | {fmt_price(r['stop'])} "
                         f"| {fmt_price(r['target'])} | {fmt_units(r['units']) if r['units'] else ''} | "
                         f"{r['risk_cash']:,.0f} | {r['reason']} |")
        lines.append("")
    return "\n".join(lines)


def append_history(rows: list[dict], data_date: str) -> int:
    """Append actionable rows not yet in signals/history.csv. Returns how many were added."""
    path = config.path("signals", "history.csv")
    seen = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            seen = {(r["data_date"], r["portfolio"], r["sleeve"], r["instrument"], r["action"]) for r in csv.DictReader(fh)}
    fresh = [r for r in rows if r["action"] not in QUIET_ACTIONS
             and (data_date, r["portfolio"], r["sleeve"], r["instrument"], r["action"]) not in seen]
    if fresh:
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS)
            if new:
                writer.writeheader()
            for r in fresh:
                writer.writerow({"data_date": data_date, "portfolio": r["portfolio"], "sleeve": r["sleeve"],
                                 "instrument": r["instrument"], "action": r["action"], "side": r["side"],
                                 "entry": r["entry_price"], "stop": r["stop"], "target": r["target"],
                                 "units": round(r["units"], 4), "risk_cash": r["risk_cash"], "reason": r["reason"]})
    return len(fresh)


def generate(portfolio_ids: list[str], cfg: dict, synthetic: bool = False) -> dict:
    """Compute signals for the portfolios, write signals/latest.json, latest.md and history.csv. Returns the payload."""
    from . import data as datamod
    universe = config.load_universe()
    portfolios = config.load_portfolios()
    payload: dict = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "data_date": None, "portfolios": {}}
    for pid in portfolio_ids:
        pf = portfolios[pid]
        rules = config.load_rules(pf.rules)
        champion = config.load_champion(pid)
        data = datamod.load(list(pf.instruments), cfg, synthetic=synthetic)
        if not data:
            continue
        rows, risk = signal_rows(pid, champion, data, universe, rules, cfg)
        date = str(datamod.last_date(data).date())
        payload["data_date"] = max(payload["data_date"] or date, date)
        payload["portfolios"][pid] = {"champion_id": champion["id"], "risk_fraction": risk, "account_size": rules.account_size,
                                      "risk_budget": rules.risk_guard * rules.daily_loss * rules.account_size, "rows": rows}
    if not payload["portfolios"]:
        raise SystemExit("no signals: no data for any portfolio (run `fetch` first, or use --synthetic)")
    config.write_json(config.path("signals", "latest.json"), payload)
    config.path("signals", "latest.md").write_text(markdown(payload), encoding="utf-8")
    all_rows = [r for p in payload["portfolios"].values() for r in p["rows"]]
    payload["history_added"] = append_history(all_rows, payload["data_date"])
    return payload
