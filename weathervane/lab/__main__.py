"""python -m weathervane.lab run --group modern|legacy|video|all | report | chains [--synthetic]

`run` executes the adapters of a group (each guarded: it returns a record and never raises) and writes
out/lab/<group>.json. `report` merges the group files into reports/lab/<date>.md and results/context.json. `chains`
is the first workflow job: it fetches the rate and option chains and writes the shared inputs. The exit code is 0
whenever the command itself ran; a failing library is data, not an error.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from . import registry
from .adapters import base
from .context import build_context, load_config, prepare_inputs
from .util import clip, out_dir, write_json

log = logging.getLogger("weathervane.lab")


def run_group(group: str, only: list[str] | None = None) -> dict:
    ctx = build_context()
    entries = registry.group_entries(group)
    if only:
        entries = [e for e in entries if e.key in only]
    payload = {"group": group, "asof": ctx.today.isoformat(), "python": sys.version.split()[0], "adapters": {}}
    for e in entries:
        t0 = time.time()
        rec = registry.run_entry(e, ctx)
        payload["adapters"][e.key] = rec
        print(f"[lab:{group}] {e.key:<22} {rec['status']:<8} {rec['seconds']:>7.1f}s  {clip(rec.get('reason', ''), 90)}", flush=True)
    write_json(out_dir(f"{group}.json"), payload)
    return payload


def cmd_run(args) -> int:
    groups = list(registry.GROUPS) if args.group == "all" else [args.group]
    for g in groups:
        run_group(g, args.only)
    return 0


def cmd_report(args) -> int:
    from . import report
    info = report.write(build_context())
    print(json.dumps(info))
    return 0


def cmd_chains(args) -> int:
    meta = prepare_inputs(load_config(), synthetic_chains_flag=args.synthetic)
    print(f"[lab:chains] source={meta.get('source')} rows={meta.get('rows')} rate={meta.get('rate')} ({meta.get('rate_source')})")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m weathervane.lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the adapters of a group")
    r.add_argument("--group", choices=[*registry.GROUPS, "all"], required=True)
    r.add_argument("--only", nargs="*", help="adapter keys (debugging)")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("report", help="merge group results into reports/lab/<date>.md").set_defaults(fn=cmd_report)
    c = sub.add_parser("chains", help="fetch the rate and option chains")
    c.add_argument("--synthetic", action="store_true", help="Black-Scholes chains from a flat smile (offline testing)")
    c.set_defaults(fn=cmd_chains)
    args = ap.parse_args(argv)
    try:
        rc = args.fn(args)
    finally:
        sys.stdout.flush()
    if base.STUCK_THREADS:      # a timed-out adapter left a daemon thread running; do not wait for it
        print(f"[lab] threads still running after timeout: {base.STUCK_THREADS}", flush=True)
        os._exit(rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
