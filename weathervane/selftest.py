"""End-to-end self test on synthetic data, in a temporary workspace (WV_ROOT). Never touches the real repo files or the network.

Steps: research (random mutations only), promote, signals, ledger, paper (dry run), charts, pine, report, notify (dry run),
companies (skipped without keys). Secrets are removed from the environment for the duration so nothing can call out.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

from . import config
from .companies import SECRET_ENV

log = logging.getLogger("weathervane.selftest")

PAPER_DAYS = 150
FAST = {"walkforward": {"n_trials": 6, "train_sims": 300, "oos_sims": 1500}, "promotion": {"vault_sims": 500},
        "research": {"llm_proposals_per_run": 0}}


def _copy_workspace(dest: Path) -> None:
    src = config.root()
    shutil.copytree(src / "config", dest / "config")
    shutil.copytree(src / "champion", dest / "champion", ignore=shutil.ignore_patterns("history"))
    (dest / "champion" / "history").mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "program.md", dest / "program.md")


def _inject_challenger(cfg: dict) -> None:
    """Put the champion's own structure on the leaderboard so the promotion path can be exercised."""
    from . import experiments, researcher, spec
    ws = researcher.open_workspace("ftmo", cfg, True)
    champion = config.load_champion("ftmo")
    pf = {"sleeves": champion["sleeves"]}
    ev, score = researcher.evaluate_and_score(ws, pf, None)
    row = researcher.experiment_row(ws, "ftmo-selftest", spec.portfolio_hash(pf), "selftest", "selftest challenger", pf, score,
                                    "leaderboard", ev.runtime_s)
    experiments.add_to_leaderboard("ftmo", researcher.leaderboard_entry(row, pf, ev, score, champion), cfg["research"]["leaderboard_size"])


def _steps(minutes: float, problems: list[str]) -> None:
    from . import charts, experiments, ledger, live, notify, pine, reports, researcher
    from .companies import run as run_companies
    from .paper import sync
    from .synthetic import SYNTHETIC_END
    start = str((pd.Timestamp(SYNTHETIC_END) - pd.offsets.BDay(PAPER_DAYS)).date())
    cfg = config.load_engine({**FAST, "signals": {"paper_start_date": start}})
    ids = [p.id for p in config.select_portfolios(None)]
    timings: dict[str, float] = {}

    def step(name: str, fn):
        t = time.monotonic()
        try:
            out = fn()
        except BaseException as exc:                      # a failing step is reported, the rest still run
            problems.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
            log.error("selftest step %s failed: %s", name, exc)
            out = None
        timings[name] = time.monotonic() - t
        log.info("selftest %-10s %.1fs", name, timings[name])
        return out

    def check(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    summary = step("research", lambda: researcher.research("ftmo", minutes, cfg, synthetic=True, seed=1, use_llm=False))
    check(bool(summary) and summary.get("experiments", 0) >= 1, "research ran no experiments")
    check(experiments.experiments_path().exists(), "results/experiments.csv missing")

    def promote():
        board = experiments.board_for(experiments.load_leaderboard(), "ftmo")
        injected = not board["entries"]
        if injected:
            _inject_challenger(cfg)
        res = researcher.promote("ftmo", cfg, synthetic=True, force=True, skip_vault=injected)
        if res.get("result") == "promoted":
            pine.generate(["ftmo"], cfg)
        return res

    res = step("promote", promote)
    check(bool(res) and res.get("result") in ("promoted", "vault_failed"), f"promote gave {res}")

    payload = step("signals", lambda: live.generate(ids, cfg, synthetic=True))
    check(bool(payload) and config.path("signals", "latest.md").exists(), "signals/latest.md missing")
    state = step("ledger", lambda: ledger.update(ids, cfg, synthetic=True))
    check(bool(state) and ledger.ledger_path().exists() and ledger.equity_path().exists(), "ledger files missing")
    check(bool(state) and all(p["days"] > 100 for p in state["portfolios"].values()), "ledger has too few days")
    sync_res = step("paper", lambda: sync.run(cfg, dry_run=True, synthetic=True))
    check(bool(sync_res) and sync_res["dry_run"] and config.path("results", "paper", "broker_sync.json").exists(), "broker_sync.json missing")
    names = step("charts", lambda: charts.generate(ids, cfg, synthetic=True))
    check(bool(names) and any(n.endswith(".png") for n in names), "no charts written")
    written = step("pine", lambda: pine.generate(ids, cfg))
    check(bool(written) and all(not pine.lint(p.read_text(encoding="utf-8")) for p in written), "Pine lint failed")
    report = step("report", lambda: reports.write(ids, cfg, synthetic=True))
    check(bool(report) and Path(report).exists(), "report missing")
    step("notify", lambda: notify.notify_signals(cfg, dry_run=True))
    rows = step("companies", lambda: run_companies(cfg, ["AAPL"], minutes=1))
    check(bool(rows) and rows[0]["status"].startswith("skipped"), "companies should skip without keys")
    log.info("selftest timings: %s", {k: round(v, 1) for k, v in timings.items()})


def run(minutes: float = 2.0, keep: bool = False) -> int:
    started = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix="wv_selftest_"))
    saved = {k: os.environ.pop(k) for k in (*SECRET_ENV, "SEC_USER_AGENT", "GITHUB_REPOSITORY") if k in os.environ}
    old_root = os.environ.get("WV_ROOT")
    problems: list[str] = []
    try:
        _copy_workspace(tmp)
        os.environ["WV_ROOT"] = str(tmp)
        _steps(minutes, problems)
        latest = config.path("signals", "latest.md")
        if latest.exists():
            print("\n".join(latest.read_text(encoding="utf-8").splitlines()[:16]))
        for p in sorted(config.path("pine").glob("*.pine"))[:1]:
            print("\n".join(p.read_text(encoding="utf-8").splitlines()[:6]))
    finally:
        os.environ.update(saved)
        if old_root is None:
            os.environ.pop("WV_ROOT", None)
        else:
            os.environ["WV_ROOT"] = old_root
        if keep:
            print(f"workspace kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    elapsed = time.monotonic() - started
    if problems:
        print(f"selftest FAILED in {elapsed:.0f}s:\n  " + "\n  ".join(problems))
        return 1
    print(f"selftest passed in {elapsed:.0f}s")
    return 0
