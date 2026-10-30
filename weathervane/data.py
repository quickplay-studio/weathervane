"""Market data: OANDA practice candles or yfinance, parquet cache, incremental updates, synthetic swap."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from . import config
from .config import Instrument
from .paper.oanda import BASE_URL as OANDA_BASE
from .synthetic import load_synthetic

log = logging.getLogger("weathervane.data")

COLUMNS = ["open", "high", "low", "close", "volume"]
US_CLOSE_HOUR_UTC = 21.5     # indices and ETFs are final once the US session is over
OANDA_PARAMS = {"granularity": "D", "price": "M", "dailyAlignment": "17", "alignmentTimezone": "America/New_York"}
FULL_REFRESH_DAYS = 7        # adjusted yfinance history shifts after dividends, so refetch in full weekly


def data_dir():
    return config.path("data")


def cache_path(inst_id: str):
    return data_dir() / f"{inst_id}.parquet"


def meta_path():
    return data_dir() / "meta.json"


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, tz-naive date index, no NaN or non-positive OHLC, consistent high and low."""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"))
    out = df.rename(columns=str.lower)[COLUMNS].copy()
    idx = pd.DatetimeIndex(out.index)
    out.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    return out.astype(float)


def drop_incomplete(df: pd.DataFrame, asset_class: str, now: datetime) -> pd.DataFrame:
    """Drop today's bar unless the instrument's session is over (US-hours markets after the close)."""
    today = pd.Timestamp(now.date())
    hour = now.hour + now.minute / 60.0
    if asset_class in ("index", "etf") and hour >= US_CLOSE_HOUR_UTC:
        return df[df.index <= today]
    return df[df.index < today]


def merge(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """New rows replace old rows on the same date."""
    if old is None or old.empty:
        return new
    return normalise(pd.concat([old[~old.index.isin(new.index)], new]))


def with_retries(fn, tries: int, what: str):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # network and parsing errors alike are retried then reported
            wait = 2 ** attempt
            log.warning("%s failed (%s: %s); attempt %d/%d", what, type(exc).__name__, str(exc)[:120], attempt + 1, tries)
            if attempt + 1 < tries:
                time.sleep(wait)
    return None


def fetch_yfinance(inst: Instrument, full: bool, days: int, now: datetime) -> pd.DataFrame | None:
    import yfinance as yf  # imported late: heavy, and absent in some test environments
    ticker = yf.Ticker(inst.yf)
    if full:
        raw = ticker.history(period="max", interval="1d", auto_adjust=True)
    else:
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        raw = ticker.history(start=start, interval="1d", auto_adjust=True)
    return None if raw is None or raw.empty else drop_incomplete(normalise(raw), inst.asset_class, now)


def parse_oanda_candles(payload: dict) -> pd.DataFrame:
    """Completed OANDA mid candles to a frame labelled with the trading date (bar start + 7h in New York)."""
    rows = []
    for c in payload.get("candles", []):
        if not c.get("complete"):
            continue
        start = pd.Timestamp(c["time"]).tz_convert("America/New_York")
        mid = c["mid"]
        rows.append((start + pd.Timedelta(hours=7), float(mid["o"]), float(mid["h"]), float(mid["l"]),
                     float(mid["c"]), float(c.get("volume", 0))))
    if not rows:
        return normalise(None)
    frame = pd.DataFrame(rows, columns=["date", *COLUMNS]).set_index("date")
    return normalise(frame)


def fetch_oanda(inst: Instrument, token: str, full: bool) -> pd.DataFrame | None:
    url = f"{OANDA_BASE}/v3/instruments/{inst.oanda}/candles"
    params = {**OANDA_PARAMS, "count": "5000" if full else "40"}
    resp = requests.get(url, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    frame = parse_oanda_candles(resp.json())
    return None if frame.empty else frame


def _full_due(meta: dict, inst_id: str, now: datetime, cached: pd.DataFrame | None) -> bool:
    if cached is None or cached.empty:
        return True
    last = meta.get(inst_id)
    return last is None or (now - datetime.fromisoformat(last)).days >= FULL_REFRESH_DAYS


def fetch(ids: list[str], cfg: dict, now: datetime | None = None) -> dict[str, str]:
    """Update the parquet cache for each instrument. Returns id -> status text. Never raises on a failed source."""
    now = now or datetime.now(timezone.utc)
    universe = config.load_universe()
    d_cfg = cfg["data"]
    token = os.environ.get("OANDA_TOKEN")
    data_dir().mkdir(parents=True, exist_ok=True)
    meta = config.read_json(meta_path(), {})
    deadline = time.monotonic() + d_cfg["fetch_budget_s"]
    status: dict[str, str] = {}
    for inst_id in ids:
        if time.monotonic() > deadline:
            status[inst_id] = "skipped: fetch time budget used"
            log.warning("fetch budget used, skipping %s", inst_id)
            continue
        inst = universe[inst_id]
        cached = pd.read_parquet(cache_path(inst_id)) if cache_path(inst_id).exists() else None
        full = _full_due(meta, inst_id, now, cached)
        use_oanda = bool(token and inst.oanda)
        fresh = with_retries(
            (lambda: fetch_oanda(inst, token, full)) if use_oanda else (lambda: fetch_yfinance(inst, full, d_cfg["incremental_days"], now)),
            d_cfg["retries"], f"fetch {inst_id} via {'oanda' if use_oanda else 'yfinance'}")
        if not use_oanda:
            time.sleep(d_cfg["yf_sleep_s"])
        if fresh is None or fresh.empty:
            status[inst_id] = "failed: kept cache" if cached is not None else "failed: no data"
            log.warning("%s: %s", inst_id, status[inst_id])
            continue
        merged = fresh if full else merge(cached, fresh)
        merged.to_parquet(cache_path(inst_id))
        if full:
            meta[inst_id] = now.isoformat()
        status[inst_id] = f"ok: {len(merged)} bars to {merged.index[-1].date()} ({'full' if full else 'incremental'})"
        log.info("%s: %s", inst_id, status[inst_id])
    config.write_json(meta_path(), meta)
    return status


def load(ids: list[str], cfg: dict, start: str | None = None, synthetic: bool = False) -> dict[str, pd.DataFrame]:
    """id -> OHLCV frame. Instruments with fewer than data.min_bars bars are skipped with a log line."""
    if synthetic:
        frames = load_synthetic(config.load_universe(), ids)
    else:
        frames = {i: pd.read_parquet(cache_path(i)) for i in ids if cache_path(i).exists()}
    out = {}
    for inst_id in ids:
        df = frames.get(inst_id)
        if df is None:
            log.warning("%s: no data, skipped", inst_id)
        elif len(df) < cfg["data"]["min_bars"]:
            log.warning("%s: only %d bars, skipped", inst_id, len(df))
        else:
            out[inst_id] = df if start is None else df[df.index >= pd.Timestamp(start)]
    return out


def last_date(data: dict[str, pd.DataFrame]) -> pd.Timestamp:
    return max(df.index[-1] for df in data.values())
