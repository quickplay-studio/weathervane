"""AutoResearch loop: propose one change to the champion, evaluate it walk-forward, keep it only if it clearly wins."""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from . import config, data as datamod, exam, experiments, mutate, regime, scoring, spec, walkforward
from .config import Portfolio, Rules
from .llm import LLMError, Router, parse_json_reply
from .search import BudgetExceeded

log = logging.getLogger("weathervane.research")

CHAMPION_SHARE_KEYS = ("oos_pass", "risk", "oos_trades", "dsr", "maxdd", "regimes")
MAX_DUPLICATES_IN_A_ROW = 15


@dataclass
class Workspace:
    """Everything one portfolio's research or promotion run needs."""

    pid: str
    portfolio: Portfolio
    rules: Rules
    cfg: dict
    universe: dict
    book: walkforward.Book
    labels: pd.Series | None
    data_date: str
    vault_start: pd.Timestamp | None
    full_data: dict


def open_workspace(pid: str, cfg: dict, synthetic: bool = False) -> Workspace:
    portfolio = config.load_portfolios()[pid]
    universe = config.load_universe()
    full = datamod.load(list(portfolio.instruments), cfg, synthetic=synthetic)
    if not full:
        raise SystemExit(f"no data for portfolio {pid}: run `fetch` first (or use --synthetic)")
    research, vault_start = walkforward.split_vault(full, cfg["walkforward"]["vault_days"])
    book = walkforward.Book(research, universe)
    return Workspace(pid, portfolio, config.load_rules(portfolio.rules), cfg, universe, book,
                     regime_labels(research, portfolio, cfg, book), str(datamod.last_date(full).date()), vault_start, full)


def regime_labels(research: dict, portfolio: Portfolio, cfg: dict, book: walkforward.Book) -> pd.Series | None:
    ref = research.get(portfolio.regime_reference)
    if ref is None:
        log.warning("regime reference %s has no data, regime gate is skipped", portfolio.regime_reference)
        return None
    folds = walkforward.make_folds(book.grid, cfg["walkforward"])
    train_end = folds[0].test_start if folds else None
    return regime.fit(ref, train_end).label(ref)


def evaluate_and_score(ws: Workspace, pf: dict, deadline: float | None) -> tuple[walkforward.Evaluation, scoring.Score]:
    ev = walkforward.evaluate(pf, ws.book, ws.rules, ws.cfg, deadline)
    n_prior = len(experiments.read_experiments(ws.pid))
    score = scoring.score_oos(ev.oos_daily, ev.oos_trades, ws.labels, ws.rules, ws.cfg, n_prior + 1,
                              experiments.prior_sharpes(ws.pid))
    return ev, score


def save_oos_curve(pid: str, ev: walkforward.Evaluation, risk: float, data_date: str) -> None:
    """Weekly equity of the OOS series at the champion's risk, refreshed at most once a week."""
    path = config.path("results", f"oos_curve_{pid}.csv")
    if path.exists():
        last = pd.read_csv(path)["date"].iloc[-1]
        if (pd.Timestamp(data_date) - pd.Timestamp(last)).days < 7:
            return
    equity = 1.0 + (risk * ev.oos_daily["r_close"]).cumsum()
    weekly = equity.resample("W").last().dropna().round(5)
    path.parent.mkdir(parents=True, exist_ok=True)
    weekly.rename("equity").to_csv(path, index_label="date")


def ensure_champion_scored(ws: Workspace, champion: dict, deadline: float | None) -> scoring.Score:
    """Score the champion on today's data once; the result is cached in the champion file by data date."""
    cached = champion.get("scores")
    if cached and champion.get("scores_data_date") == ws.data_date:
        return _score_from_cache(cached)
    log.info("scoring champion %s on data to %s", champion["id"], ws.data_date)
    ev, score = evaluate_and_score(ws, {"sleeves": champion["sleeves"]}, deadline)
    champion["scores"] = score.as_dict()
    champion["scores_data_date"] = ws.data_date
    if not champion.get("risk"):
        champion["risk"] = score.risk
    config.save_champion(champion)
    experiments.save_asset_scan(ws.pid, ws.data_date, experiments.asset_scan(ev.oos_trades))
    save_oos_curve(ws.pid, ev, champion["risk"], ws.data_date)
    return score


def _score_from_cache(d: dict) -> scoring.Score:
    keys = scoring.Score.__dataclass_fields__.keys()
    return scoring.Score(**{k: d.get(k, [] if k == "table" else None) for k in keys})


def build_prompt(ws: Workspace, champion: dict, champ_score: scoring.Score) -> tuple[str, str]:
    """(system, user) messages: program.md plus grammar, then the champion's state."""
    program = config.path("program.md").read_text(encoding="utf-8")
    system = (f"{program}\n\n## Grammar\n{spec.describe_grammar()}\n\n## Allowed instruments for this portfolio\n"
              f"{', '.join(ws.portfolio.instruments)}\n")
    state = {
        "portfolio": ws.pid,
        "rules": {"daily_loss": ws.rules.daily_loss, "max_loss": ws.rules.max_loss, "phases": [p.target for p in ws.rules.phases]},
        "champion": {"sleeves": champion["sleeves"], "live_params": champion.get("live_params")},
        "champion_scores": {k: getattr(champ_score, k) for k in CHAMPION_SHARE_KEYS},
        "last_experiments": experiments.recent_rows(ws.pid, 25),
        "instrument_oos_contribution": experiments.load_asset_scan(ws.pid).get("instruments", {}),
    }
    return system, "Current state:\n" + json.dumps(state, default=float) + "\n\nPropose one change. JSON only."


def parse_proposal(text: str, ws: Workspace) -> tuple[dict, str]:
    obj = parse_json_reply(text)
    pf = obj.get("portfolio")
    if not isinstance(pf, dict) or not isinstance(pf.get("sleeves"), list):
        raise ValueError("reply needs portfolio.sleeves")
    pf = {"sleeves": [spec.normalise(s) for s in pf["sleeves"] if isinstance(s, dict)]}
    r_cfg = ws.cfg["research"]
    try:
        spec.validate_portfolio(pf, ws.universe, ws.portfolio.instruments, r_cfg["max_sleeves"], r_cfg["max_filters"])
    except spec.SpecError as exc:
        raise ValueError(f"invalid portfolio: {exc}") from exc
    return pf, str(obj.get("change_description", "")).strip()[:200] or "llm change"


def llm_proposal(ws: Workspace, champion: dict, champ_score: scoring.Score, router: Router) -> tuple[dict, str, str]:
    """Ask the LLM for one change; on an unusable reply ask once more with the error. Raises on failure."""
    system, user = build_prompt(ws, champion, champ_score)
    text, label = router.complete(system, user)
    try:
        pf, desc = parse_proposal(text, ws)
    except (ValueError, json.JSONDecodeError) as exc:
        log.info("LLM reply rejected (%s), retrying once", exc)
        text, label = router.complete(system, f"{user}\n\nYour previous reply was rejected: {exc}. Reply again with corrected JSON only.")
        pf, desc = parse_proposal(text, ws)
    if spec.portfolio_hash(pf) == spec.portfolio_hash({"sleeves": champion["sleeves"]}):
        raise ValueError("LLM proposed the champion unchanged")
    return pf, desc, f"llm:{label}"


def propose(ws: Workspace, champion: dict, champ_score: scoring.Score, router: Router | None, llm_left: int,
            rng: random.Random) -> tuple[dict, str, str, bool]:
    """One candidate: from the LLM while it has budget, otherwise a random mutation. Last item: whether the LLM was tried."""
    tried = False
    if router is not None and llm_left > 0 and router.available():
        tried = True
        try:
            return (*llm_proposal(ws, champion, champ_score, router), tried)
        except (LLMError, ValueError, spec.SpecError) as exc:
            log.warning("LLM proposal failed (%s), using a random mutation", str(exc)[:160])
    weak = [i for i, v in experiments.load_asset_scan(ws.pid).get("instruments", {}).items() if v["expectancy_r"] < 0]
    ctx = mutate.Context(list(ws.portfolio.instruments), ws.cfg["research"]["max_sleeves"], ws.cfg["research"]["max_filters"], weak)
    child, desc = mutate.mutate({"sleeves": champion["sleeves"]}, ctx, rng, list(ws.universe),
                                ws.cfg["research"]["proposal_retries"] * 3)
    return child, desc, "random", tried


def experiment_row(ws: Workspace, exp_id: str, parent_hash: str, source: str, desc: str, pf: dict, score: scoring.Score,
                   decision: str, runtime: float) -> dict:
    return {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "portfolio": ws.pid, "exp_id": exp_id,
            "parent_hash": parent_hash, "source": source, "change_description": desc, "oos_pass": round(score.oos_pass, 4),
            "oos_trades": score.oos_trades, "dsr": round(score.dsr, 4), "regime_ok": score.regime_ok,
            "maxdd": round(score.maxdd, 4), "risk": score.risk, "decision": decision, "runtime_s": round(runtime, 1),
            "spec_hash": spec.portfolio_hash(pf), "oos_sharpe": round(score.sharpe_day, 5)}


def leaderboard_entry(row: dict, pf: dict, ev: walkforward.Evaluation, score: scoring.Score, champion: dict) -> dict:
    return {"exp_id": row["exp_id"], "ts": row["ts"], "spec_hash": row["spec_hash"], "source": row["source"],
            "change_description": row["change_description"], "parent": champion["id"], "oos_pass": score.oos_pass,
            "dsr": score.dsr, "risk": score.risk, "portfolio": pf, "live_params": ev.live_params,
            "score": {k: v for k, v in score.as_dict().items() if k != "table"}}


def research(pid: str, minutes: float, cfg: dict, synthetic: bool = False, seed: int | None = None,
             use_llm: bool = True) -> dict:
    """Run experiments on one portfolio until the time budget is used. Returns a summary."""
    started = time.monotonic()
    deadline = started + minutes * 60
    ws = open_workspace(pid, cfg, synthetic)
    champion = config.load_champion(pid)
    summary = {"portfolio": pid, "experiments": 0, "leaderboard": 0, "llm": 0, "random": 0, "duplicates": 0, "stopped": "time"}
    try:
        champ_score = ensure_champion_scored(ws, champion, deadline)
    except BudgetExceeded:
        summary["stopped"] = "budget used while scoring the champion"
        return summary
    summary["champion_oos_pass"] = round(champ_score.oos_pass, 4)
    router = Router(cfg) if use_llm else None
    llm_left = cfg["research"]["llm_proposals_per_run"]
    rng = random.Random(seed if seed is not None else int(datetime.now(timezone.utc).strftime("%Y%m%d")) + sum(map(ord, pid)))
    seen = experiments.known_hashes(pid)
    parent_hash = spec.portfolio_hash({"sleeves": champion["sleeves"]})
    last_runtime, dup_streak = 0.0, 0
    while deadline - time.monotonic() > max(5.0, 0.8 * last_runtime) and dup_streak < MAX_DUPLICATES_IN_A_ROW:
        try:
            pf, desc, source, tried = propose(ws, champion, champ_score, router, llm_left, rng)
        except RuntimeError as exc:
            summary["stopped"] = str(exc)
            break
        llm_left -= int(tried)
        h = spec.portfolio_hash(pf)
        if h in seen:
            dup_streak += 1
            summary["duplicates"] += 1
            continue
        dup_streak = 0
        seen.add(h)
        try:
            ev, score = evaluate_and_score(ws, pf, deadline)
        except BudgetExceeded:
            summary["stopped"] = "budget used mid-candidate"
            break
        last_runtime = ev.runtime_s
        better = scoring.beats(score, champ_score.oos_pass, cfg["gates"]["min_improvement"])
        row = experiment_row(ws, f"{pid}-{len(experiments.read_experiments(pid)) + 1:05d}", parent_hash, source, desc, pf,
                             score, "leaderboard" if better else "rejected", ev.runtime_s)
        experiments.append_experiment(row)
        summary["experiments"] += 1
        summary["llm" if source.startswith("llm") else "random"] += 1
        log.info("%s %s oos_pass=%.3f dsr=%.2f -> %s", row["exp_id"], desc, score.oos_pass, score.dsr, row["decision"])
        if better:
            summary["leaderboard"] += 1
            experiments.add_to_leaderboard(pid, leaderboard_entry(row, pf, ev, score, champion), cfg["research"]["leaderboard_size"])
    summary["elapsed_s"] = round(time.monotonic() - started, 1)
    return summary


# promotion ---------------------------------------------------------------------------------------------------------

def vault_check(ws: Workspace, entry: dict) -> dict:
    """Trade the entry's live parameters through the hidden vault bars and bootstrap an exam from them."""
    sleeves = [spec.apply_params(spec.normalise(s), entry["live_params"].get(s["id"], {})) for s in entry["portfolio"]["sleeves"]]
    book = walkforward.Book(ws.full_data, ws.universe)
    lo = int(book.grid.searchsorted(ws.vault_start.to_datetime64())) if ws.vault_start is not None else 0
    run = book.run(sleeves)
    daily = (run.rc[lo:], run.rw[lo:], run.act[lo:])
    wf = ws.cfg["walkforward"]
    res = exam.run_exam(daily, ws.rules, [entry["risk"]], ws.cfg["promotion"]["vault_sims"], wf["block_len"], wf["seed"])[0]
    need = ws.cfg["promotion"]["vault_pass_ratio"] * entry["oos_pass"]
    total_r = float(run.rc[lo:].sum())
    return {"vault_pass": res.pass_combined, "vault_total_r": total_r, "needed_pass": need,
            "passed": res.pass_combined >= need and total_r > 0}


def promote(pid: str, cfg: dict, synthetic: bool = False, force: bool = False, today: datetime | None = None,
            skip_vault: bool = False) -> dict:
    """Weekly: vault-check the best challenger and make it champion if it holds up. At most one check a week.

    force lifts the one-check-a-week limit. skip_vault passes the challenger without the vault test and exists only for
    the selftest, which has to exercise the promotion path on synthetic data."""
    today = today or datetime.now(timezone.utc)
    week = today.strftime("%G-W%V")
    board = experiments.load_leaderboard()
    b = experiments.board_for(board, pid)
    if not b["entries"]:
        return {"portfolio": pid, "result": "no challenger on the leaderboard"}
    if b["vault_checks"].get(week, 0) >= 1 and not force:
        return {"portfolio": pid, "result": f"vault already checked in {week}"}
    ws = open_workspace(pid, cfg, synthetic)
    entry = b["entries"][0]
    check = ({"vault_pass": entry["oos_pass"], "vault_total_r": 0.0, "needed_pass": 0.0, "passed": True, "vault_skipped": True}
             if skip_vault else vault_check(ws, entry))
    b["vault_checks"][week] = b["vault_checks"].get(week, 0) + 1
    result = {"portfolio": pid, "exp_id": entry["exp_id"], **check}
    if check["passed"]:
        old = config.load_champion(pid)
        config.archive_champion(old, today.strftime("%Y%m%dT%H%M%S"))
        config.save_champion(new_champion(pid, entry, today))
        b["entries"] = []
        result.update(result="promoted", previous=old["id"], champion=entry["exp_id"])
    else:
        b["entries"] = b["entries"][1:]
        b.setdefault("vault_failed", []).append({"exp_id": entry["exp_id"], "ts": today.isoformat(timespec="seconds"),
                                                 "vault_pass": check["vault_pass"], "vault_total_r": check["vault_total_r"]})
        b["vault_failed"] = b["vault_failed"][-20:]
        result["result"] = "vault_failed"
    config.write_json(experiments.leaderboard_path(), board)
    return result


def new_champion(pid: str, entry: dict, today: datetime) -> dict:
    return {"id": entry["exp_id"], "portfolio": pid, "promoted": today.date().isoformat(), "risk": entry["risk"],
            "sleeves": entry["portfolio"]["sleeves"], "live_params": entry["live_params"],
            "scores": entry["score"], "scores_data_date": None, "source": entry["source"],
            "change_description": entry["change_description"]}
