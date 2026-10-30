"""PNG charts with mplfinance: one candle chart per champion instrument, plus OOS and paper-ledger equity curves.

Charts go to out/charts/. They are never committed to main: the workflow force-pushes them to the orphan `charts` branch.
Regime shading is descriptive only (the HMM is never part of a trading rule).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, live, regime, simulator as sm
from .config import Instrument

log = logging.getLogger("weathervane.charts")

RECENT_BARS = 30          # a trade opened or closed this recently makes an instrument "recent"
REGIME_COLOURS = {"bull": "#2e9e5b", "bear": "#d64545", "chop": "#8a8a8a"}
SHADE_ALPHA = 0.07
UP, DOWN = "#1b9e4b", "#d62728"
DPI = 100


def charts_dir():
    return config.path("out", "charts")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def clear_old(directory) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.png"):
        old.unlink()


def ohlc_frame(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    tail = df.tail(bars)
    return tail.rename(columns=str.capitalize)[["Open", "High", "Low", "Close", "Volume"]]


def trade_marks(run: live.LiveRun, base: int, n: int) -> dict[str, list[tuple[int, float]]]:
    """Entry and exit points of one sleeve on one instrument, as (chart x position, price)."""
    sim = run.sim
    marks: dict[str, list[tuple[int, float]]] = {"long": [], "short": [], "exit": []}
    for (i0, i1), f in zip(sim.trade_i, sim.trade_f):
        side = "long" if f[0] > 0 else "short"
        if 0 <= i0 - base < n:
            marks[side].append((int(i0 - base), float(f[1])))
        if 0 <= i1 - base < n:
            marks["exit"].append((int(i1 - base), float(f[2])))
    st = sim.state
    if int(st[sm.S_POS]) != 0:
        i0 = int(st[sm.S_ENTRY_I])
        if 0 <= i0 - base < n:
            marks["long" if st[sm.S_POS] > 0 else "short"].append((i0 - base, float(st[sm.S_ENTRY_PX])))
    return marks


def was_recent(run: live.LiveRun, row: dict) -> bool:
    n = len(run.bars)
    if row["action"] != "FLAT":
        return True
    ti = run.sim.trade_i
    return bool(len(ti) and (ti[:, 1] >= n - RECENT_BARS).any())


def _price_text(x: float) -> str:
    return live.fmt_price(x)


def chart_instrument(inst: Instrument, df: pd.DataFrame, items: list[tuple[live.LiveRun, dict]],
                     labels: pd.Series | None, out_path, bars: int = 300, portfolio: str = "") -> None:
    """Candles of the last `bars` bars with regime shading, entries, exits and the open position's stop and target."""
    import mplfinance as mpf
    plt = _plt()
    ohlc = ohlc_frame(df, bars)
    n, base = len(ohlc), len(df) - len(ohlc)
    style = mpf.make_mpf_style(base_mpf_style="yahoo", gridcolor="#e8e8e8", gridstyle=":", facecolor="white",
                               marketcolors=mpf.make_marketcolors(up=UP, down=DOWN, edge="inherit", wick="inherit"))
    fig, axes = mpf.plot(ohlc, type="candle", style=style, volume=False, returnfig=True, figsize=(12, 6.5),
                         xrotation=0, datetime_format="%b %Y", warn_too_much_data=10 ** 6)
    ax = axes[0]
    if labels is not None and len(labels):
        lab = labels.reindex(ohlc.index, method="ffill").fillna("chop")
        pos_of = {d: k for k, d in enumerate(ohlc.index)}
        for first, last, name in regime.spans(lab):
            ax.axvspan(pos_of[first] - 0.5, pos_of[last] + 0.5, color=REGIME_COLOURS[name], alpha=SHADE_ALPHA, lw=0, zorder=0)
    span = float(ohlc["High"].max() - ohlc["Low"].min()) or 1.0
    off = 0.025 * span
    lows, highs = ohlc["Low"].to_numpy(), ohlc["High"].to_numpy()
    levels = [float(ohlc["Low"].min()), float(ohlc["High"].max())]
    for run, row in items:
        marks = trade_marks(run, base, n)
        if marks["long"]:
            xs, _ = zip(*marks["long"])
            ax.scatter(xs, [lows[x] - off for x in xs], marker="^", s=75, color=UP, edgecolor="black", linewidth=0.5, zorder=6)
        if marks["short"]:
            xs, _ = zip(*marks["short"])
            ax.scatter(xs, [highs[x] + off for x in xs], marker="v", s=75, color=DOWN, edgecolor="black", linewidth=0.5, zorder=6)
        if marks["exit"]:
            xs, ys = zip(*marks["exit"])
            ax.scatter(xs, ys, marker="x", s=45, color="black", linewidth=1.4, zorder=6)
        if row["action"] in ("ENTER_LONG", "ENTER_SHORT", "REVERSE_TO_LONG", "REVERSE_TO_SHORT", "HOLD", "MOVE_STOP"):
            for key, colour in (("stop", DOWN), ("target", UP)):
                level = row.get(key)
                if level is not None:
                    ax.axhline(level, color=colour, linestyle="--", linewidth=1.1, zorder=4)
                    ax.text(n - 0.5, level, f" {key} {_price_text(level)}", color=colour, va="bottom", ha="right", fontsize=8)
                    levels.append(float(level))
    pad = 0.04 * (max(levels) - min(levels))
    ax.set_ylim(min(levels) - pad - off, max(levels) + pad + off)
    ax.set_xlim(-1, n + 1)
    actions = "; ".join(f"{row['sleeve']}: {row['action']}" + (f" {row['side']}" if row["side"] and row["action"] != "EXIT" else "")
                        for _, row in items)
    ax.set_title(f"{inst.id} ({inst.tv})  [{portfolio}]  {actions}  |  last bar {ohlc.index[-1].date()}", fontsize=11, loc="left")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def chart_equity(series: pd.Series, title: str, out_path, ylabel: str, markers: list[tuple[pd.Timestamp, str]] | None = None) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(series.index, series.to_numpy(), color="#1f5fbf", linewidth=1.6)
    ax.axhline(series.iloc[0], color="#999999", linewidth=0.8, linestyle=":")
    for when, text in markers or []:
        ax.axvline(when, color="#555555", linewidth=0.7, linestyle="--")
        ax.text(when, ax.get_ylim()[1], f" {text}", fontsize=7, va="top", rotation=90)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e8e8e8", linestyle=":")
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def equity_charts(pid: str, directory) -> list[str]:
    """OOS curve (written by the researcher) and the paper ledger curve for one portfolio."""
    made = []
    oos = config.path("results", f"oos_curve_{pid}.csv")
    if oos.exists():
        s = pd.read_csv(oos, parse_dates=["date"]).set_index("date")["equity"]
        if len(s) >= 2:
            name = f"{pid}_oos_equity.png"
            chart_equity(s, f"{pid}: walk-forward out-of-sample equity at the champion's risk (weekly, start = 1.0)",
                         directory / name, "equity")
            made.append(name)
    eq = config.path("results", "paper", "equity.csv")
    if eq.exists():
        df = pd.read_csv(eq, parse_dates=["date"])
        df = df[df["portfolio"] == pid]
        if len(df) >= 2:
            name = f"{pid}_paper_equity.png"
            events = [(pd.Timestamp(r.date), r.exam_event) for r in df.itertuples() if isinstance(r.exam_event, str) and r.exam_event]
            chart_equity(df.set_index("date")["equity"], f"{pid}: internal paper ledger equity (cash)", directory / name,
                         "account value", events[:12])
            made.append(name)
    return made


def index_markdown(files: dict[str, list[str]], generated: str) -> str:
    lines = ["# Weathervane charts", "", f"Generated {generated}. Paper trading only.", ""]
    for pid, names in files.items():
        lines += [f"## {pid}", ""] + [f"![{n}]({n})" for n in names] + [""]
    return "\n".join(lines)


def generate(portfolio_ids: list[str], cfg: dict, synthetic: bool = False) -> list[str]:
    """Write all charts to out/charts/. Returns the file names."""
    from datetime import datetime, timezone
    from . import data as datamod
    universe, portfolios = config.load_universe(), config.load_portfolios()
    out = charts_dir()
    clear_old(out)
    budget = int(cfg["charts"]["max_instruments"])
    bars_n = int(cfg["charts"]["bars"])
    per_pf: dict[str, list[str]] = {}
    candidates: list[tuple[bool, str, str, str, list]] = []
    contexts: dict[str, tuple] = {}
    for pid in portfolio_ids:
        pf = portfolios[pid]
        rules = config.load_rules(pf.rules)
        champion = config.load_champion(pid)
        data = datamod.load(list(pf.instruments), cfg, synthetic=synthetic)
        if not data:
            log.warning("%s: no data, no instrument charts", pid)
            continue
        risk = config.live_risk(champion, cfg, rules)
        runs = live.run_live(champion, data, universe)
        rows = [live.row_for(r, pid, risk, rules.account_size) for r in runs]
        live.apply_risk_guard(rows, rules, rules.account_size)
        by_inst: dict[str, list] = {}
        for run, row in zip(runs, rows):
            by_inst.setdefault(run.instrument.id, []).append((run, row))
        ref = data.get(pf.regime_reference)
        labels = None
        if ref is not None:
            cut = ref.index[-cfg["walkforward"]["vault_days"]] if len(ref) > cfg["walkforward"]["vault_days"] else None
            labels = regime.fit(ref, cut).label(ref)
        contexts[pid] = (data, labels)
        for inst_id, items in by_inst.items():
            recent = any(was_recent(r, w) for r, w in items)
            candidates.append((recent, pid, inst_id, universe[inst_id].tv, items))
    candidates.sort(key=lambda c: (not c[0], c[1], c[2]))
    for recent, pid, inst_id, _, items in candidates[:budget]:
        data, labels = contexts[pid]
        name = f"{pid}_{inst_id}.png"
        try:
            chart_instrument(universe[inst_id], data[inst_id], items, labels, out / name, bars_n, pid)
            per_pf.setdefault(pid, []).append(name)
        except Exception as exc:  # one broken chart must not stop the rest
            log.warning("chart %s failed: %s: %s", name, type(exc).__name__, str(exc)[:160])
    for pid in portfolio_ids:
        per_pf[pid] = sorted(per_pf.get(pid, [])) + equity_charts(pid, out)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (out / "index.md").write_text(index_markdown(per_pf, stamp), encoding="utf-8")
    names = [n for v in per_pf.values() for n in v]
    log.info("wrote %d charts to %s", len(names), out)
    return names
