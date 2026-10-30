"""Command line: python -m weathervane <command>. Every command that reads market data accepts --synthetic."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from . import config

COMMANDS = ("fetch", "research", "promote", "signals", "ledger", "paper", "charts", "pine", "report", "companies", "notify", "selftest")


def portfolio_ids(arg: str | None) -> list[str]:
    return [p.id for p in config.select_portfolios(None if arg in (None, "all") else arg)]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m weathervane", description="Weathervane: swing-trading research engine (paper trading only).")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str, portfolio: bool = True, synthetic: bool = True):
        p = sub.add_parser(name, help=help_)
        if portfolio:
            p.add_argument("--portfolio", default=None, help="portfolio id (default: all)")
        if synthetic:
            p.add_argument("--synthetic", action="store_true", help="use deterministic synthetic data instead of the data cache")
        return p

    add("fetch", "update the data cache (yfinance or OANDA practice)")
    p = add("research", "run the nightly experiment loop on a time budget")
    p.add_argument("--minutes", type=float, default=30.0, help="time budget in minutes (default 30)")
    p.set_defaults(portfolio="auto")        # 'auto' alternates by date, 'all' splits the budget
    p.add_argument("--no-llm", action="store_true", help="random mutations only")
    p.add_argument("--seed", type=int, default=None)
    p = add("promote", "vault-check the best challenger and promote it (weekly)")
    p.add_argument("--force", action="store_true", help="ignore the one vault check per week limit")
    p.add_argument("--skip-vault", action="store_true", help="selftest only: pass the challenger without the vault test")
    add("signals", "compute live signals from the champions and write signals/")
    p = add("ledger", "update the internal paper ledger")
    p.add_argument("--paper-start", default=None, help="paper start date YYYY-MM-DD (default: config or first run)")
    p = add("paper", "sync signals to OANDA practice and Alpaca paper", portfolio=False)
    p.add_argument("--dry-run", action="store_true", help="print the intended orders, send nothing")
    add("charts", "draw PNG charts into out/charts/")
    add("pine", "generate Pine v6 strategies into pine/", synthetic=False)
    add("report", "write the weekly report")
    p = add("companies", "TradingAgents company reports (research only)", portfolio=False, synthetic=False)
    p.add_argument("--tickers", default=None, help="comma separated tickers (default: today's rotation)")
    p.add_argument("--minutes", type=float, default=18.0, help="time budget in minutes")
    p = add("notify", "post the latest actionable signals to the GitHub issue", portfolio=False, synthetic=False)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="post even if these signals were already posted")
    p = add("selftest", "end-to-end run on synthetic data in a temporary workspace", portfolio=False)
    p.add_argument("--minutes", type=float, default=2.0, help="research budget inside the selftest")
    p.add_argument("--keep", action="store_true", help="keep the temporary workspace and print its path")
    return ap


def cmd_fetch(args, cfg) -> int:
    from . import data
    ids = sorted({i for pid in portfolio_ids(args.portfolio) for i in config.load_portfolios()[pid].instruments})
    if args.synthetic:
        print("fetch: --synthetic uses generated data, nothing to fetch")
        return 0
    for inst, status in data.fetch(ids, cfg).items():
        print(f"{inst:<8} {status}")
    return 0


def cmd_research(args, cfg) -> int:
    from . import researcher
    ids = portfolio_ids(None if args.portfolio in ("auto", "all") else args.portfolio)
    if args.portfolio == "auto":
        ids = [ids[datetime.now(timezone.utc).date().toordinal() % len(ids)]]
    share = args.minutes / len(ids)
    for pid in ids:
        summary = researcher.research(pid, share, cfg, synthetic=args.synthetic, seed=args.seed, use_llm=not args.no_llm)
        print(summary)
    return 0


def cmd_promote(args, cfg) -> int:
    from . import notify, pine, researcher
    results = []
    for pid in portfolio_ids(args.portfolio):
        res = researcher.promote(pid, cfg, synthetic=args.synthetic, force=args.force, skip_vault=args.skip_vault)
        print(res)
        results.append(res)
    promoted = [r["portfolio"] for r in results if r.get("result") == "promoted"]
    if promoted:
        pine.generate(promoted, cfg)
        notify.notify_promotion(results, cfg)
    return 0


def cmd_signals(args, cfg) -> int:
    from . import live
    payload = live.generate(portfolio_ids(args.portfolio), cfg, synthetic=args.synthetic)
    n_act = sum(r["action"] not in live.QUIET_ACTIONS for p in payload["portfolios"].values() for r in p["rows"])
    print(f"signals for {payload['data_date']}: {n_act} actionable, {payload['history_added']} added to history")
    return 0


def cmd_ledger(args, cfg) -> int:
    from . import ledger
    state = ledger.update(portfolio_ids(args.portfolio), cfg, synthetic=args.synthetic, start_override=args.paper_start)
    for pid, p in state["portfolios"].items():
        print(f"{pid}: {p['days']} days, equity {p['equity']:,.0f}, exam phase {p['exam']['phase']} attempt {p['exam']['attempt']}")
    return 0


def cmd_paper(args, cfg) -> int:
    from .paper import sync
    summary = sync.run(cfg, dry_run=args.dry_run, synthetic=args.synthetic)
    for broker, res in summary["brokers"].items():
        print(f"{broker}: {res['status']} ({len(res['orders'])} orders)")
    return 0


def cmd_charts(args, cfg) -> int:
    from . import charts
    names = charts.generate(portfolio_ids(args.portfolio), cfg, synthetic=args.synthetic)
    print(f"{len(names)} charts in {charts.charts_dir()}")
    return 0


def cmd_pine(args, cfg) -> int:
    from . import pine
    for path in pine.generate(portfolio_ids(args.portfolio), cfg):
        print(path)
    return 0


def cmd_report(args, cfg) -> int:
    from . import reports
    print(reports.write(portfolio_ids(args.portfolio), cfg, synthetic=args.synthetic))
    return 0


def cmd_companies(args, cfg) -> int:
    from . import companies
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    for row in companies.run(cfg, tickers, args.minutes):
        print(f"{row['ticker']}: {row['status']} -> {row['path']}")
    return 0


def cmd_notify(args, cfg) -> int:
    from . import notify
    print(notify.notify_signals(cfg, dry_run=args.dry_run, force=args.force))
    return 0


def cmd_selftest(args, cfg) -> int:
    from . import selftest
    return selftest.run(minutes=args.minutes, keep=args.keep)


HANDLERS = {name: globals()[f"cmd_{name}"] for name in COMMANDS}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("urllib3", "matplotlib", "numba", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return HANDLERS[args.command](args, config.load_engine())


if __name__ == "__main__":
    sys.exit(main())
