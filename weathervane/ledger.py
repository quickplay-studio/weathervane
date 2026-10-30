"""Internal paper ledger: replays the champion's live signals forward from the paper start date.

Each run re-simulates every champion segment from its start date with the live parameters, on the real bars, so fills
are the actual next opens and stops and targets follow the same intrabar rules as the backtest. The day-by-day account
changes are played through the exam rules (phase, trading days, breaches) with `ExamTracker`.

When the champion changes, the old segment is closed at the data date of the run that noticed it: its positions are
flattened at the next open and the new champion starts flat from that date. The ledger keeps going and records the
champion id on every trade. The risk guard that scales live entries is not applied here (stated approximation).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, simulator as sm
from .config import Instrument, Rules
from .exam import ExamTracker
from .live import live_sleeves
from .signals_core import Bars, build_signals

log = logging.getLogger("weathervane.ledger")

LEDGER_COLUMNS = ["portfolio", "champion", "sleeve", "instrument", "side", "entry_date", "exit_date", "entry_price",
                  "exit_price", "stop", "r", "pnl_cash", "bars", "mae_r", "reason", "status"]
EQUITY_COLUMNS = ["portfolio", "date", "champion", "risk", "ret", "worst", "active", "equity", "exam_phase", "exam_attempt",
                  "exam_equity", "exam_trading_days", "exam_event"]


def ledger_path():
    return config.path("results", "paper", "ledger.csv")


def equity_path():
    return config.path("results", "paper", "equity.csv")


def state_path():
    return config.path("results", "paper", "state.json")


def load_state() -> dict:
    return config.read_json(state_path(), {})


def segment_for(champion: dict, risk: float, start: str) -> dict:
    """A frozen copy of what the champion trades from `start` on."""
    return {"champion_id": champion["id"], "start": start, "risk": float(risk), "sleeves": champion["sleeves"],
            "live_params": champion.get("live_params") or {}}


def update_segments(segments: list[dict], champion: dict, risk: float, data_date: str, first_start: str) -> list[dict]:
    """Keep the list of champion segments current. A new champion id opens a segment at the run's data date."""
    if not segments:
        return [segment_for(champion, risk, first_start)]
    out = [dict(s) for s in segments]
    last = out[-1]
    if last["champion_id"] == champion["id"]:
        out[-1] = segment_for(champion, risk, last["start"])      # current champion: follow its file (risk, parameters)
        return out
    if data_date <= last["start"]:
        out[-1] = segment_for(champion, risk, last["start"])      # changed on the segment's own first day: replace it
    else:
        out.append(segment_for(champion, risk, data_date))
    return out


def _as_champion(seg: dict) -> dict:
    return {"id": seg["champion_id"], "sleeves": seg["sleeves"], "live_params": seg["live_params"]}


def replay(segments: list[dict], data: dict[str, pd.DataFrame], universe: dict[str, Instrument], rules: Rules,
           pid: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """(daily frame, trades frame, open positions) for all segments from the first segment's start."""
    bars = {i: Bars(df) for i, df in data.items()}
    used = sorted({i for seg in segments for s in seg["sleeves"] for i in s["instruments"] if i in bars})
    if not used:
        return pd.DataFrame(), pd.DataFrame(columns=LEDGER_COLUMNS), []
    grid = sm.master_grid([bars[i].index for i in used])
    ret, worst, act = np.zeros(grid.shape[0]), np.zeros(grid.shape[0]), np.zeros(grid.shape[0], np.int8)
    frames, open_positions = [], []
    starts = [pd.Timestamp(s["start"]) for s in segments]
    for k, seg in enumerate(segments):
        end = starts[k + 1] if k + 1 < len(segments) else None
        for sleeve in live_sleeves(_as_champion(seg)):
            for inst_id in sleeve["instruments"]:
                if inst_id not in bars:
                    continue
                b = bars[inst_id]
                lo = int(b.index.searchsorted(starts[k]))
                hi = int(b.index.searchsorted(end)) if end is not None else None
                sim = sm.simulate(sleeve, b, universe[inst_id].half_cost, inst_id, build_signals(sleeve, b), start=lo,
                                  stop_idx=hi)
                pos = np.searchsorted(grid, sim.dates)
                ret[pos] += seg["risk"] * sim.r_close
                worst[pos] += seg["risk"] * sim.r_worst
                act[pos] = np.maximum(act[pos], sim.active)
                t = sim.trades_frame()
                if not t.empty:
                    t["champion"], t["risk"] = seg["champion_id"], seg["risk"]
                    frames.append(t)
                held = open_position(sim, b, seg, pid)
                if held:
                    open_positions.append(held)
    first = np.datetime64(starts[0])
    keep = grid >= first
    daily = pd.DataFrame({"date": pd.DatetimeIndex(grid[keep]), "ret": ret[keep], "worst": worst[keep],
                          "active": act[keep]})
    seg_idx = np.searchsorted(np.array([np.datetime64(s) for s in starts]), grid[keep], side="right") - 1
    daily["champion"] = [segments[max(0, int(j))]["champion_id"] for j in seg_idx]
    daily["risk"] = [segments[max(0, int(j))]["risk"] for j in seg_idx]
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return daily, trades, open_positions


def open_position(sim: sm.Sim, b: Bars, seg: dict, pid: str) -> dict | None:
    """The position still open after the last bar, marked to the last close, or None."""
    st = sim.state
    side = int(st[sm.S_POS])
    if side == 0:
        return None
    entry_px, r0 = float(st[sm.S_ENTRY_PX]), float(st[sm.S_R0])
    close = float(b.close[-1])
    return {"champion": seg["champion_id"], "risk": seg["risk"], "sleeve": sim.sleeve, "instrument": sim.instrument,
            "side": "long" if side > 0 else "short", "entry_date": str(b.index[int(st[sm.S_ENTRY_I])].date()),
            "entry_price": entry_px, "last_close": close, "stop": float(st[sm.S_STOP]), "stop_initial": float(st[sm.S_STOP0]),
            "r": side * (close - entry_px) / r0}


def play_exam(daily: pd.DataFrame, rules: Rules) -> tuple[pd.DataFrame, ExamTracker]:
    """Feed the real days through the exam rules and add the account curve and exam state columns."""
    tracker = ExamTracker(rules)
    rows = []
    for rec in daily.itertuples(index=False):
        out = tracker.day(str(rec.date.date()), float(rec.ret), float(rec.worst), int(rec.active))
        rows.append((out["phase"], out["attempt"], out["equity"], out["trading_days"], out["event"]))
    cols = ["exam_phase", "exam_attempt", "exam_equity", "exam_trading_days", "exam_event"]
    exam_df = pd.DataFrame(rows, columns=cols, index=daily.index)
    out = pd.concat([daily, exam_df], axis=1)
    out["equity"] = rules.account_size * (1.0 + out["ret"].cumsum())
    return out, tracker


def ledger_rows(pid: str, trades: pd.DataFrame, open_positions: list[dict], rules: Rules) -> pd.DataFrame:
    """Closed trades and open positions in ledger.csv format, newest first."""
    acct = rules.account_size
    rows = []
    if not trades.empty:
        for t in trades.itertuples(index=False):
            reason = "champion_changed" if t.reason == "window_end" else t.reason
            rows.append({"portfolio": pid, "champion": t.champion, "sleeve": t.sleeve, "instrument": t.instrument, "side": t.side,
                         "entry_date": t.entry_date.date().isoformat(), "exit_date": t.exit_date.date().isoformat(),
                         "entry_price": t.entry_price, "exit_price": t.exit_price, "stop": t.stop, "r": t.r,
                         "pnl_cash": t.r * t.risk * acct, "bars": t.bars, "mae_r": t.mae_r, "reason": reason, "status": "closed"})
    for p in open_positions:
        rows.append({"portfolio": pid, "champion": p["champion"], "sleeve": p["sleeve"], "instrument": p["instrument"],
                     "side": p["side"], "entry_date": p["entry_date"], "exit_date": "", "entry_price": p["entry_price"],
                     "exit_price": p["last_close"], "stop": p["stop_initial"], "r": p["r"], "pnl_cash": p["r"] * p["risk"] * acct,
                     "bars": "", "mae_r": "", "reason": "open", "status": "open"})
    out = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    return out.sort_values(["entry_date", "instrument"], ascending=[False, True]).reset_index(drop=True)


def portfolio_state(daily: pd.DataFrame, trades: pd.DataFrame, open_positions: list[dict], tracker: ExamTracker,
                    segments: list[dict], rules: Rules, start: str) -> dict:
    ret = daily["ret"].to_numpy(float)
    eq = 1.0 + np.cumsum(ret) if ret.size else np.ones(1)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    r = trades["r"].to_numpy(float) if not trades.empty else np.zeros(0)
    return {"start": start, "data_date": str(daily["date"].iloc[-1].date()) if len(daily) else None,
            "account_size": rules.account_size, "days": int(len(daily)), "equity": float(rules.account_size * eq[-1]),
            "return_pct": float(100.0 * (eq[-1] - 1.0)), "max_dd_pct": float(100.0 * (peak[1:] - eq).max()) if ret.size else 0.0,
            "closed_trades": int(r.size), "total_r": float(r.sum()), "avg_r": float(r.mean()) if r.size else 0.0,
            "win_rate": float((r > 0).mean()) if r.size else 0.0, "open_positions": open_positions,
            "segments": [{k: v for k, v in s.items()} for s in segments], "exam": tracker.snapshot()}


def update(portfolio_ids: list[str], cfg: dict, synthetic: bool = False, start_override: str | None = None) -> dict:
    """Replay every portfolio, rewrite results/paper/ledger.csv, equity.csv and state.json. Returns the state."""
    from . import data as datamod
    universe = config.load_universe()
    portfolios = config.load_portfolios()
    state = load_state()
    explicit = start_override or cfg["signals"].get("paper_start_date")
    if explicit and state.get("paper_start_date") and str(explicit) != state["paper_start_date"]:
        log.info("paper start date changed to %s, the ledger restarts", explicit)
        state = {}
    ledger_parts, equity_parts = [], []
    state.setdefault("portfolios", {})
    for pid in portfolio_ids:
        pf = portfolios[pid]
        rules = config.load_rules(pf.rules)
        champion = config.load_champion(pid)
        data = datamod.load(list(pf.instruments), cfg, synthetic=synthetic)
        if not data:
            log.warning("%s: no data, ledger not updated", pid)
            continue
        data_date = str(datamod.last_date(data).date())
        start = str(explicit) if explicit else state.get("paper_start_date") or data_date
        state.setdefault("paper_start_date", start)
        pstate = state["portfolios"].get(pid, {})
        segments = update_segments(pstate.get("segments", []), champion, config.live_risk(champion, cfg, rules), data_date,
                                   pstate.get("start") or start)
        daily, trades, held = replay(segments, data, universe, rules, pid)
        if daily.empty:
            continue
        daily, tracker = play_exam(daily, rules)
        new_state = portfolio_state(daily, trades, held, tracker, segments, rules, segments[0]["start"])
        state["portfolios"][pid] = new_state
        ledger_parts.append(ledger_rows(pid, trades, held, rules))
        daily.insert(0, "portfolio", pid)
        daily["date"] = daily["date"].dt.date.astype(str)
        equity_parts.append(daily[EQUITY_COLUMNS])
        log.info("%s ledger: %d days, %d closed trades, equity %.0f, exam phase %d attempt %d", pid, new_state["days"],
                 new_state["closed_trades"], new_state["equity"], new_state["exam"]["phase"], new_state["exam"]["attempt"])
    if not equity_parts:
        raise SystemExit("ledger: no data for any portfolio (run `fetch` first, or use --synthetic)")
    state["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ledger_path().parent.mkdir(parents=True, exist_ok=True)
    done = {p for p in portfolio_ids if p in state["portfolios"]}
    keep_other(ledger_path(), done, ledger_parts)
    keep_other(equity_path(), done, equity_parts)
    pd.concat(ledger_parts, ignore_index=True).to_csv(ledger_path(), index=False, float_format="%.6g")
    pd.concat(equity_parts, ignore_index=True).to_csv(equity_path(), index=False, float_format="%.8g")
    config.write_json(state_path(), state)
    return state


def keep_other(path, updated: set[str], parts: list[pd.DataFrame]) -> None:
    """Add the rows of portfolios that this run did not touch, so updating one portfolio keeps the others."""
    if path.exists() and path.stat().st_size:
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        rest = old[~old["portfolio"].isin(updated)]
        if len(rest):
            parts.append(rest)
