"""Weekly report: reports/weekly/<date>.md with champion, out-of-sample metrics, exam odds, regimes, paper vs backtest,
the week's best experiments, LLM usage and an Actions minutes estimate. Plain prose, no em dashes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config, experiments, ledger

log = logging.getLogger("weathervane.reports")

WEEK_DAYS = 7
TOP_EXPERIMENTS = 5


def reports_dir():
    return config.path("reports", "weekly")


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None or x != x else f"{100.0 * x:.{digits}f}%"


def num(x: float | None, digits: int = 2) -> str:
    if x is None or x != x:
        return "n/a"
    return "inf" if x == float("inf") else f"{x:.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def ensure_scores(pid: str, cfg: dict, synthetic: bool, champion: dict) -> dict:
    """The champion's cached scores, computing them first when the champion has never been scored."""
    if champion.get("scores") and champion["scores"].get("metrics"):
        return champion["scores"]
    from . import researcher
    log.info("%s: champion has no scores yet, scoring it for the report", pid)
    ws = researcher.open_workspace(pid, cfg, synthetic)
    researcher.ensure_champion_scored(ws, champion, None)
    return champion["scores"]


def champion_section(pid: str, champion: dict, scores: dict, rules: config.Rules) -> list[str]:
    lines = [f"## {pid}", "", f"Champion `{champion['id']}`, promoted {champion.get('promoted') or 'never (seed)'}. "
             f"Scored on data to {champion.get('scores_data_date') or 'n/a'}. Risk per trade {pct(champion.get('risk'), 2)} of the account.", ""]
    if champion.get("change_description"):
        lines += [f"Last change: {champion['change_description']} (source {champion.get('source', 'n/a')}).", ""]
    rows = []
    for s in champion["sleeves"]:
        live = (champion.get("live_params") or {}).get(s["id"], {})
        params = ", ".join(f"{k}={v}" for k, v in live.items()) or "default parameters"
        e = s["entry"]
        rows.append([s["id"], f"{e['type']}", s["direction"], ", ".join(s["instruments"]), params])
    lines += table(["Sleeve", "Entry", "Direction", "Instruments", "Live parameters"], rows)
    m = scores.get("metrics") or {}
    lines += ["### Out-of-sample metrics (walk-forward, at the champion's risk)", ""]
    lines += table(["Metric", "Value"], [
        ["Sharpe", num(m.get("sharpe"))], ["Sortino", num(m.get("sortino"))], ["Calmar", num(m.get("calmar"))],
        ["Max drawdown", pct(m.get("max_dd"))], ["CVaR 95 (daily)", pct(m.get("cvar95"), 2)],
        ["Expectancy (R per trade)", num(m.get("expectancy_r"), 3)], ["Win rate", pct(m.get("win_rate"))],
        ["Profit factor", num(m.get("profit_factor"))], ["Average MAE (R)", num(m.get("avg_mae_r"))],
        ["Trades", str(m.get("trades", "n/a"))], ["Days", str(m.get("days", "n/a"))],
        ["Total return", pct(m.get("total_return"))]])
    ex = scores.get("exam") or {}
    lines += [f"### Exam odds ({rules.firm} {rules.program}, out-of-sample, bootstrapped)", ""]
    fails = ex.get("fail_reasons") or {}
    lines += table(["Measure", "Value"], [
        ["Pass phase 1", pct(ex.get("pass_p1"))], ["Pass phase 2", pct(ex.get("pass_p2"))],
        ["Pass both", pct(ex.get("pass_combined"))], ["Median days to pass", num(ex.get("days_median"), 0)],
        ["90th percentile days", num(ex.get("days_p90"), 0)],
        ["Fail: daily loss", pct(fails.get("daily"))], ["Fail: max loss", pct(fails.get("max_loss"))],
        ["Fail: time or timeout", pct((fails.get("time") or 0) + (fails.get("timeout") or 0))]])
    grid = scores.get("table") or []
    if grid:
        lines += ["Risk per trade against pass probability:", ""]
        lines += table(["Risk", "Pass both", "Median days"], [[pct(g["risk"], 2), pct(g["pass_combined"]), num(g.get("days_median"), 0)] for g in grid])
    reg = scores.get("regimes") or {}
    if reg:
        lines += ["### Regimes (HMM on the reference series, trade count and expectancy)", ""]
        lines += table(["Regime", "Trades", "Expectancy (R)"], [[k, v["trades"], num(v["expectancy_r"], 3)] for k, v in reg.items()])
    gates = scores.get("gates") or {}
    if gates:
        lines += ["Gates: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in gates.items()) + f". DSR {num(scores.get('dsr'), 3)}.", ""]
    return lines


def paper_section(pid: str, scores: dict) -> list[str]:
    st = ledger.load_state().get("portfolios", {}).get(pid)
    lines = ["### Paper ledger against the backtest", ""]
    if not st:
        return lines + ["The internal paper ledger has not started yet.", ""]
    m = scores.get("metrics") or {}
    days = max(1, m.get("days") or 1)
    bt_annual = (m.get("total_return") or 0.0) / days * 252
    paper_annual = (st["return_pct"] / 100.0) / max(1, st["days"]) * 252
    ex = st["exam"]
    lines += table(["Measure", "Paper ledger", "Backtest (OOS)"], [
        ["Period", f"{st['start']} to {st['data_date']} ({st['days']} days)", f"{m.get('days', 'n/a')} days"],
        ["Return, annualised", pct(paper_annual), pct(bt_annual)],
        ["Max drawdown", f"{st['max_dd_pct']:.1f}%", pct(m.get("max_dd"))],
        ["Closed trades", st["closed_trades"], m.get("trades", "n/a")],
        ["Win rate", pct(st["win_rate"]), pct(m.get("win_rate"))],
        ["Expectancy (R)", num(st["avg_r"], 3), num(m.get("expectancy_r"), 3)]])
    lines += [f"Exam tracker: phase {ex['phase']}, attempt {ex['attempt']}, equity {ex['equity']:.3f}, {ex['trading_days']} trading days "
              f"in the current phase, {len(ex['events'])} exam events so far.", ""]
    if st["open_positions"]:
        lines += ["Open paper positions: " + "; ".join(f"{p['instrument']} {p['side']} ({p['r']:+.2f} R)" for p in st["open_positions"]) + ".", ""]
    return lines


def experiments_section(pid: str, now: datetime) -> list[str]:
    df = experiments.read_experiments(pid)
    lines = ["### Top experiments of the week", ""]
    if df.empty:
        return lines + ["No experiments yet.", ""]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    week = df[df["ts"] >= pd.Timestamp(now - timedelta(days=WEEK_DAYS))]
    lines.append(f"{len(week)} experiments this week, {int((week['decision'] == 'leaderboard').sum())} reached the leaderboard, "
                 f"{len(df)} in total. Evaluation time this week: {week['runtime_s'].sum() / 60:.0f} minutes.")
    lines.append("")
    if week.empty:
        return lines
    top = week.sort_values(["oos_pass", "dsr"], ascending=False).head(TOP_EXPERIMENTS)
    lines += table(["Experiment", "Change", "OOS pass", "DSR", "Trades", "Max DD", "Decision"],
                   [[r.exp_id, str(r.change_description)[:80], pct(r.oos_pass), num(r.dsr), int(r.oos_trades), pct(r.maxdd), r.decision]
                    for r in top.itertuples()])
    board = experiments.board_for(experiments.load_leaderboard(), pid)
    if board["entries"]:
        e = board["entries"][0]
        lines += [f"Best challenger waiting for the weekly vault check: `{e['exp_id']}` with OOS pass {pct(e['oos_pass'])}.", ""]
    if board.get("vault_failed"):
        lines += [f"Vault failures so far: {len(board['vault_failed'])}.", ""]
    return lines


def llm_section(now: datetime) -> list[str]:
    usage = config.read_json(config.path("results", "llm_usage.json"), {})
    cutoff = (now - timedelta(days=WEEK_DAYS)).strftime("%Y-%m-%d")
    totals: dict[str, list[int]] = {}
    for day, providers in usage.items():
        if day < cutoff:
            continue
        for label, v in providers.items():
            t = totals.setdefault(label, [0, 0])
            t[0] += v.get("requests", 0)
            t[1] += v.get("errors", 0)
    lines = ["## LLM usage, last 7 days", ""]
    if not totals:
        return lines + ["No LLM requests were made (no keys, or random mutation only).", ""]
    return lines + table(["Provider and model", "Requests", "Errors"], [[k, v[0], v[1]] for k, v in sorted(totals.items())])


def minutes_section(cfg: dict) -> list[str]:
    est = cfg["actions"]["estimate_minutes_per_month"]
    total, limit = sum(est.values()), cfg["actions"]["minutes_per_month_limit"]
    lines = ["## Actions minutes estimate (per month)", ""]
    lines += table(["Workflow", "Minutes"], [[k, v] for k, v in est.items()] + [["Total", f"{total} of {limit} ({100 * total / limit:.0f}%)"]])
    lines += ["These are budgets from the workflow timeouts, not measurements. Check Settings, Billing for the real figure.", ""]
    return lines


def promotions_section(now: datetime) -> list[str]:
    hist = sorted(config.path("champion", "history").glob("*.json"))
    cutoff = now - timedelta(days=WEEK_DAYS)
    recent = [p.stem for p in hist if datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) >= cutoff]
    return ["## Promotions this week", "", ("Champions replaced: " + ", ".join(recent) + ".") if recent else "No champion was replaced.", ""]


def tearsheet(pid: str, champion: dict, cfg: dict, synthetic: bool) -> str | None:
    """Optional quantstats HTML for the paper ledger returns, written next to the charts. Skipped without the package."""
    try:
        import quantstats as qs
    except Exception:
        return None
    eq = config.path("results", "paper", "equity.csv")
    if not eq.exists():
        return None
    df = pd.read_csv(eq, parse_dates=["date"])
    df = df[df["portfolio"] == pid]
    if len(df) < 30:
        return None
    from .charts import charts_dir
    out = charts_dir() / f"tearsheet_{pid}.html"
    try:
        charts_dir().mkdir(parents=True, exist_ok=True)
        qs.reports.html(df.set_index("date")["ret"], title=f"Weathervane {pid} paper ledger", output=str(out), download_filename=str(out))
    except Exception as exc:  # the tearsheet is a bonus: never fail the report for it
        log.warning("quantstats tearsheet failed: %s: %s", type(exc).__name__, str(exc)[:120])
        return None
    return out.name


def build(portfolio_ids: list[str], cfg: dict, now: datetime | None = None, synthetic: bool = False) -> str:
    now = now or datetime.now(timezone.utc)
    portfolios = config.load_portfolios()
    lines = [f"# Weathervane weekly report {now.strftime('%Y-%m-%d')}", "",
             "Out-of-sample figures come from walk-forward optimisation on years the optimiser never saw. "
             "Paper trading only: nothing here is advice and no real orders are placed.", ""]
    for pid in portfolio_ids:
        champion = config.load_champion(pid)
        rules = config.load_rules(portfolios[pid].rules)
        try:
            scores = ensure_scores(pid, cfg, synthetic, champion)
        except SystemExit as exc:
            lines += [f"## {pid}", "", f"Skipped: {exc}", ""]
            continue
        lines += champion_section(pid, champion, scores, rules)
        lines += paper_section(pid, scores)
        lines += experiments_section(pid, now)
        sheet = tearsheet(pid, champion, cfg, synthetic)
        if sheet:
            lines += [f"Tearsheet: `{sheet}` on the charts branch.", ""]
    lines += promotions_section(now) + llm_section(now) + minutes_section(cfg)
    return "\n".join(lines)


def write(portfolio_ids: list[str], cfg: dict, synthetic: bool = False, now: datetime | None = None):
    now = now or datetime.now(timezone.utc)
    text = build(portfolio_ids, cfg, now, synthetic)
    assert "—" not in text, "reports must not contain em dashes"
    path = reports_dir() / f"{now.strftime('%Y-%m-%d')}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("wrote %s", path)
    return path
