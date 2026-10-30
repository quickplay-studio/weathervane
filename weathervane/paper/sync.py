"""Sync the champion's latest signals to the paper brokers (OANDA practice, Alpaca paper). Paper accounts only.

Each portfolio is routed to the brokers listed in config/portfolios.yaml. A broker without credentials is skipped with a
log line. Dry-run (and any --synthetic run) never sends an order: it plans against the real account when credentials exist
and read-only calls work, otherwise against an assumed empty account, and prints what it would do.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests

from .. import config
from . import Order, alpaca, oanda

log = logging.getLogger("weathervane.paper")


def sync_path():
    return config.path("results", "paper", "broker_sync.json")


def format_order(o: Order) -> str:
    qty = f"{abs(o.units):g}" if o.units else ""
    return (f"{o.broker:<6} {o.kind:<7} {o.symbol:<10} {o.side:<5} {qty:>10}  stop {o.stop if o.stop is not None else '-':<10} "
            f"target {o.target if o.target is not None else '-':<10} [{o.status}] {o.note} {o.detail}".rstrip())


def credentials(env: dict, broker: str) -> tuple[str, ...] | None:
    keys = {"oanda": ("OANDA_TOKEN", "OANDA_ACCOUNT_ID"), "alpaca": ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}[broker]
    vals = tuple(env.get(k, "") for k in keys)
    return vals if all(vals) else None


def rows_for(payload: dict, portfolios: dict, broker: str) -> list[dict]:
    return [r for pid, p in payload["portfolios"].items() if broker in portfolios[pid].brokers for r in p["rows"]]


def sync_oanda(rows: list[dict], universe: dict, env: dict, dry_run: bool, assumed_nav: float) -> dict:
    creds = credentials(env, "oanda")
    if not rows:
        return {"status": "skipped: no portfolio routed to oanda", "orders": []}
    if creds is None and not dry_run:
        log.info("oanda: OANDA_TOKEN or OANDA_ACCOUNT_ID missing, skipped")
        return {"status": "skipped: no credentials", "orders": []}
    client, assumed = None, creds is None
    symbols = sorted({universe[r["instrument"]].oanda for r in rows if universe[r["instrument"]].oanda})
    nav, trades, info, conv = assumed_nav, [], {}, {}
    if creds is not None:
        try:
            client = oanda.OandaClient(*creds)
            nav, trades, info, conv = oanda.gather_state(client, symbols)
        except requests.RequestException as exc:
            log.warning("oanda: could not read the account (%s), skipped", type(exc).__name__)
            return {"status": f"error: {type(exc).__name__} reading the account", "orders": []}
    orders = oanda.plan(rows, universe, trades, nav, info, conv)
    oanda.execute(orders, client, dry_run, info)
    return {"status": "ok" if not assumed else "dry run on an assumed empty account", "nav": nav, "orders": [o.as_dict() for o in orders],
            "_objs": orders}


def sync_alpaca(rows: list[dict], universe: dict, env: dict, dry_run: bool, assumed_nav: float, stamp: str) -> dict:
    creds = credentials(env, "alpaca")
    if not rows:
        return {"status": "skipped: no portfolio routed to alpaca", "orders": []}
    if creds is None and not dry_run:
        log.info("alpaca: ALPACA_API_KEY or ALPACA_SECRET_KEY missing, skipped")
        return {"status": "skipped: no credentials", "orders": []}
    client, assumed = None, creds is None
    symbols = sorted({universe[r["instrument"]].alpaca_proxy for r in rows if universe[r["instrument"]].alpaca_proxy})
    state = {"nav": assumed_nav, "buying_power": None, "shorting_enabled": True, "positions": [], "orders": [], "assets": {}, "prices": {}}
    if creds is not None:
        try:
            client = alpaca.AlpacaClient(*creds)
            state = alpaca.gather_state(client, symbols)
        except requests.RequestException as exc:
            log.warning("alpaca: could not read the account (%s), skipped", type(exc).__name__)
            return {"status": f"error: {type(exc).__name__} reading the account", "orders": []}
    orders = alpaca.plan(rows, universe, state["positions"], state["orders"], state["nav"], state["buying_power"], state["assets"],
                         state["prices"], state["shorting_enabled"])
    alpaca.execute(orders, client, dry_run, stamp, state["orders"])
    return {"status": "ok" if not assumed else "dry run on an assumed empty account", "nav": state["nav"],
            "orders": [o.as_dict() for o in orders], "_objs": orders}


def run(cfg: dict, dry_run: bool = False, synthetic: bool = False, env: dict | None = None) -> dict:
    """Plan and send paper orders for signals/latest.json. Returns the summary that is also written to broker_sync.json."""
    env = dict(os.environ) if env is None else env
    payload = config.read_json(config.path("signals", "latest.json"))
    if not payload:
        raise SystemExit("paper: no signals/latest.json, run `signals` first")
    if synthetic and not dry_run:
        log.info("--synthetic never sends orders: running as a dry run")
    dry_run = dry_run or synthetic
    universe, portfolios = config.load_universe(), config.load_portfolios()
    stamp = str(payload["data_date"]).replace("-", "")
    nav = next(iter(payload["portfolios"].values()))["account_size"]
    summary = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dry_run": dry_run, "data_date": payload["data_date"],
               "brokers": {}}
    for broker, fn in (("oanda", lambda rows: sync_oanda(rows, universe, env, dry_run, nav)),
                       ("alpaca", lambda rows: sync_alpaca(rows, universe, env, dry_run, nav, stamp))):
        try:
            res = fn(rows_for(payload, portfolios, broker))
        except Exception as exc:  # a broker problem must never stop the workflow
            log.warning("%s sync failed: %s: %s", broker, type(exc).__name__, str(exc)[:160])
            res = {"status": f"error: {type(exc).__name__}", "orders": []}
        for o in res.pop("_objs", []):
            print(format_order(o))
        summary["brokers"][broker] = res
        log.info("%s: %s, %d orders", broker, res["status"], len(res["orders"]))
    config.write_json(sync_path(), summary)
    return summary
