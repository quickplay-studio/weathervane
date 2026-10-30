"""OpenBB Open Data Platform: CFTC Commitments of Traders positioning through the `cftc` provider.

Only the provider layer is used (`openbb-core` + `openbb-cftc`, the `CftcCotFetcher`); the `openbb` meta package pulls
in the whole platform and is not needed. Managed money exists only in the disaggregated reports (gold, crude). For the
financial futures (S&P 500 e-mini, euro, yen) the equivalent speculative group in the TFF report is leveraged funds, and
the result says which group each market uses. The optional CFTC_APP_TOKEN raises the Socrata rate limit.

OpenBB keeps a settings and cache folder under the user's home; HOME is pointed at out/lab/ while it runs so nothing is
written elsewhere.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..context import LabContext
from ..util import clip
from ._common import version_of

GROUPS = {
    "disaggregated": ("managed money", ["managed_money_positions_long_all", "m_money_positions_long_all"],
                      ["managed_money_positions_short_all", "m_money_positions_short_all"]),
    "financial": ("leveraged funds", ["leveraged_funds_positions_long", "lev_money_positions_long"],
                  ["leveraged_funds_positions_short", "lev_money_positions_short"]),
}


@contextlib.contextmanager
def isolated_home(base: Path):
    home = base / "openbb_home"
    home.mkdir(parents=True, exist_ok=True)
    saved = {k: os.environ.get(k) for k in ("HOME", "USERPROFILE")}
    os.environ["HOME"] = os.environ["USERPROFILE"] = str(home)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def percentile_of_last(values: np.ndarray) -> float:
    """Percent of the window at or below the latest value (ties count half)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    last = v[-1]
    return 100.0 * (float((v < last).mean()) + 0.5 * float((v == last).mean()))


def positioning(df: pd.DataFrame, report: str, years: int) -> dict:
    """Net speculative position and its percentile over `years` from a COT frame (date, long and short columns)."""
    label, long_cols, short_cols = GROUPS[report]
    long_col = next((c for c in long_cols if c in df.columns), None)
    short_col = next((c for c in short_cols if c in df.columns), None)
    if long_col is None or short_col is None:
        raise KeyError(f"no {label} columns in the response (have {sorted(df.columns)[:6]}...)")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date")
    d["net"] = pd.to_numeric(d[long_col], errors="coerce") - pd.to_numeric(d[short_col], errors="coerce")
    d = d.dropna(subset=["net"])
    if d.empty:
        raise ValueError("no rows with long and short positions")
    window = d[d["date"] >= d["date"].iloc[-1] - pd.Timedelta(days=365 * years + 7)]
    last = window.iloc[-1]
    oi = pd.to_numeric(last.get("open_interest_all"), errors="coerce") if "open_interest_all" in window.columns else float("nan")
    return {"group": label, "report": report, "date": last["date"].date().isoformat(), "net": float(last["net"]),
            "net_pct_oi": float(100.0 * last["net"] / oi) if math.isfinite(oi) and oi else None,
            "change_1w": float(window["net"].iloc[-1] - window["net"].iloc[-2]) if len(window) > 1 else None,
            "percentile": percentile_of_last(window["net"].to_numpy()), "weeks": int(len(window)),
            "min": float(window["net"].min()), "max": float(window["net"].max())}


def _fetch(fetcher, code: str, report: str, start: dt.date, token: str | None) -> pd.DataFrame:
    params = {"code": code, "report_type": report, "start_date": start, "measure": "positions"}
    creds = {"cftc_app_token": token} if token else None
    rows = asyncio.run(fetcher.fetch_data(params, creds))
    return pd.DataFrame([r.model_dump() for r in rows])


def run(ctx: LabContext) -> dict:
    cot = ctx.cfg["cot"]
    years = int(cot["lookback_years"])
    start = ctx.today - dt.timedelta(days=365 * years + 21)
    token = os.environ.get("CFTC_APP_TOKEN") or None
    markets: dict[str, dict] = {}
    with isolated_home(ctx.out):
        from openbb_cftc.models.cot import CftcCotFetcher
        for key, m in cot["markets"].items():
            entry = {"label": m["label"], "code": m["code"], "report": m["report"]}
            try:
                entry.update(status="ok", **positioning(_fetch(CftcCotFetcher, m["code"], m["report"], start, token), m["report"], years))
            except Exception as exc:  # network, empty response, schema change: record and move on
                entry.update(status="failed", error=clip(f"{type(exc).__name__}: {exc}", 160))
            markets[key] = entry
    ok = [k for k, v in markets.items() if v["status"] == "ok"]
    out = {"markets": markets, "token_used": bool(token), "version": version_of("openbb-core"), "provider": version_of("openbb-cftc")}
    if not ok:
        first = next(iter(markets.values()))
        raise RuntimeError(f"no COT market could be fetched ({first.get('error', 'unknown error')})")
    return out
