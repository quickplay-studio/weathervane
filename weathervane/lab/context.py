"""Shared inputs for the adapters: config, the deterministic test option set, option chains, price data, rates.

Chains come from yfinance and may be unavailable (blocked network, no listed options); everything downstream copes
with an empty frame. The 12-contract test set needs no network at all, so every pricing library is always compared
on identical inputs.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .. import config
from . import refprice
from .util import clip, out_dir, read_json, today_utc, write_json

log = logging.getLogger("weathervane.lab")

DEFAULTS: dict[str, Any] = {
    "timeout_s": 120,
    "chains": {
        "tickers": ["SPY", "QQQ", "IWM", "GLD", "USO", "TLT", "FXE"],
        "n_expiries": 4, "min_days": 7, "max_days": 120, "sleep_s": 1.0,
        "dividend_yield": {"SPY": 0.013, "QQQ": 0.006, "IWM": 0.012, "GLD": 0.0, "USO": 0.0, "TLT": 0.035, "FXE": 0.0},
    },
    "rates": {"fred_series": "DGS3MO", "default": 0.04},
    "test_set": {"spot": 100.0, "sigma": 0.20, "r": 0.04, "q": 0.01, "moneyness": [0.95, 1.0, 1.05], "expiry_days": [45, 105]},
    "tolerances": {"agreement_flag_rel": 0.01, "agreement_abs_floor": 0.01, "unit_abs": 0.01},
    "crosscheck": {"instruments": ["US500", "EURUSD", "XAUUSD"], "rel_flag": 1e-6, "tail": 1000},
    "sabr": {"beta": 1.0},
    "cot": {
        "lookback_years": 3,
        "markets": {
            "gold": {"label": "Gold", "code": "CFTC_088691", "report": "disaggregated"},
            "crude": {"label": "WTI crude", "code": "CFTC_067411", "report": "disaggregated"},
            "sp500_emini": {"label": "S&P 500 e-mini", "code": "CFTC_13874+", "report": "financial"},
            "eur": {"label": "Euro FX", "code": "CFTC_099741", "report": "financial"},
            "jpy": {"label": "Japanese yen", "code": "CFTC_097741", "report": "financial"},
        },
    },
    "news": {"enabled": False, "base_url": "http://127.0.0.1:9377", "tickers": ["SPY", "QQQ"], "allowlist": [],
             "blocklist": [], "delay_s": 5.0, "max_per_ticker": 5, "sources": []},
    "video": {"enabled": True, "seconds": 45, "fps": 30, "width": 1920, "height": 1080, "hyperframes_version": "0.8.217"},
    "sources": {
        "willowtree": {"path": "data/lab/src/willowtree", "repo": "https://github.com/federicomariamassari/willowtree",
                       "ref": "1a4c37ed896d3213ed8e919737511bc61d355398"},
        "financial_engineering": {"path": "data/lab/src/financial-engineering",
                                  "repo": "https://github.com/federicomariamassari/financial-engineering",
                                  "ref": "a62c130f6050426b27c9f4b1d020473140e6d51b"},
    },
}


def load_config(overrides: dict | None = None) -> dict:
    """config/lab.yaml merged over the built-in defaults (the file may be missing or partial)."""
    cfg = config.deep_merge(DEFAULTS, {})
    p = config.path("config", "lab.yaml")
    if p.exists():
        with p.open(encoding="utf-8") as fh:
            cfg = config.deep_merge(cfg, yaml.safe_load(fh) or {})
    return config.deep_merge(cfg, overrides) if overrides else cfg


# ----------------------------------------------------------------------------------------------------------------
# deterministic test option set


@dataclass(frozen=True)
class Contract:
    id: str
    cp: str            # 'c' or 'p'
    S: float
    K: float
    T: float           # years, days / 365
    days: int
    r: float
    q: float
    sigma: float
    moneyness: float

    @property
    def flag(self) -> str:
        return self.cp

    def args(self) -> tuple:
        return (self.cp, self.S, self.K, self.T, self.r, self.q, self.sigma)


def build_test_set(cfg: dict | None = None) -> list[Contract]:
    """12 contracts: call and put x 3 moneyness levels x 2 expiries, flat vol 20%, r 4%, q 1% (config)."""
    t = (cfg or load_config())["test_set"]
    out = []
    for days in t["expiry_days"]:
        for m in t["moneyness"]:
            for cp in ("c", "p"):
                k = round(t["spot"] * m, 6)
                out.append(Contract(f"{cp.upper()}{int(round(m * 100))}_{days}d", cp, float(t["spot"]), k, days / 365.0,
                                    int(days), float(t["r"]), float(t["q"]), float(t["sigma"]), float(m)))
    return out



# ----------------------------------------------------------------------------------------------------------------
# rates


def fetch_rate(cfg: dict) -> tuple[float, str]:
    """3-month rate as a decimal. FRED DGS3MO when FRED_API_KEY is set and reachable, else the configured constant."""
    default = float(cfg["rates"]["default"])
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return default, "constant"
    try:
        import requests
        resp = requests.get("https://api.stlouisfed.org/fred/series/observations",
                            params={"series_id": cfg["rates"]["fred_series"], "api_key": key, "file_type": "json",
                                    "sort_order": "desc", "limit": 10}, timeout=15)
        resp.raise_for_status()
        for obs in resp.json().get("observations", []):
            if obs.get("value") not in (None, ".", ""):
                return float(obs["value"]) / 100.0, f"FRED {cfg['rates']['fred_series']} {obs.get('date')}"
    except Exception as exc:  # network, parsing, bad key: fall back and say why (never the key)
        log.warning("FRED rate unavailable (%s), using the constant", type(exc).__name__)
    return default, "constant (FRED unavailable)"


# ----------------------------------------------------------------------------------------------------------------
# option chains

CHAIN_COLUMNS = ["ticker", "expiry", "days", "T", "cp", "strike", "bid", "ask", "mid", "last", "volume", "open_interest",
                 "spot", "q", "r", "yf_iv"]


def _empty_chains() -> pd.DataFrame:
    return pd.DataFrame(columns=CHAIN_COLUMNS)


def _expiry_picks(expiries: list[str], today: dt.date, lo: int, hi: int, n: int) -> list[tuple[str, int]]:
    picks = []
    for e in expiries:
        days = (dt.date.fromisoformat(e) - today).days
        if lo <= days <= hi:
            picks.append((e, days))
    return sorted(picks, key=lambda x: x[1])[:n]


def clean_side(df: pd.DataFrame) -> pd.DataFrame:
    """Keep strikes with open interest > 0 and bid > 0; add the mid price."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ("bid", "ask", "openInterest", "strike"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out = out[(out["openInterest"].fillna(0) > 0) & (out["bid"].fillna(0) > 0) & (out["ask"].fillna(0) > 0)]
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    return out


def fetch_chains(cfg: dict, today: dt.date | None = None, rate: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Option chains for the configured tickers via yfinance. Per-ticker failures are recorded, never raised."""
    today = today or today_utc()
    c = cfg["chains"]
    rate = cfg["rates"]["default"] if rate is None else rate
    frames, meta = [], {"source": "yfinance", "asof": today.isoformat(), "tickers": {}}
    try:
        import yfinance as yf
    except Exception as exc:
        meta["error"] = clip(f"yfinance import failed: {type(exc).__name__}: {exc}")
        return _empty_chains(), meta
    import time
    for ticker in c["tickers"]:
        info: dict[str, Any] = {"status": "failed", "options": 0}
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d", interval="1d", auto_adjust=False)
            if hist is None or hist.empty:
                raise RuntimeError("no spot price")
            spot = float(hist["Close"].dropna().iloc[-1])
            picks = _expiry_picks(list(tk.options or []), today, c["min_days"], c["max_days"], c["n_expiries"])
            if not picks:
                raise RuntimeError("no expiries in range")
            q = float(c["dividend_yield"].get(ticker, 0.0))
            for expiry, days in picks:
                chain = tk.option_chain(expiry)
                for cp, side in (("c", chain.calls), ("p", chain.puts)):
                    side = clean_side(side)
                    if side.empty:
                        continue
                    frames.append(pd.DataFrame({
                        "ticker": ticker, "expiry": expiry, "days": days, "T": days / 365.0, "cp": cp,
                        "strike": side["strike"].to_numpy(float), "bid": side["bid"].to_numpy(float),
                        "ask": side["ask"].to_numpy(float), "mid": side["mid"].to_numpy(float),
                        "last": pd.to_numeric(side.get("lastPrice"), errors="coerce").to_numpy(float),
                        "volume": pd.to_numeric(side.get("volume"), errors="coerce").fillna(0).to_numpy(float),
                        "open_interest": side["openInterest"].to_numpy(float), "spot": spot, "q": q, "r": rate,
                        "yf_iv": pd.to_numeric(side.get("impliedVolatility"), errors="coerce").to_numpy(float)}))
            n = int(sum(len(f) for f in frames if f["ticker"].iloc[0] == ticker))
            info.update(status="ok" if n else "failed", options=n, spot=spot, expiries=[e for e, _ in picks])
            if not n:
                info["error"] = "no quotes with open interest and bid"
        except Exception as exc:
            info["error"] = clip(f"{type(exc).__name__}: {exc}")
            log.warning("chains %s failed: %s", ticker, info["error"])
        meta["tickers"][ticker] = info
        time.sleep(float(c.get("sleep_s", 0)))
    df = pd.concat(frames, ignore_index=True) if frames else _empty_chains()
    meta["rows"] = int(len(df))
    return df, meta


SYNTH_SPOT = {"SPY": 580.0, "QQQ": 500.0, "IWM": 220.0, "GLD": 250.0, "USO": 75.0, "TLT": 90.0, "FXE": 105.0}
SYNTH_ATM = {"SPY": 0.16, "QQQ": 0.21, "IWM": 0.22, "GLD": 0.15, "USO": 0.35, "TLT": 0.17, "FXE": 0.07}


def synthetic_chains(cfg: dict, today: dt.date | None = None, rate: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Deterministic offline chains from a SABR smile (beta 1, rho -0.6, nu 0.7), for tests and offline runs only."""
    today = today or today_utc()
    c = cfg["chains"]
    rate = cfg["rates"]["default"] if rate is None else rate
    rows, meta = [], {"source": "synthetic", "asof": today.isoformat(), "tickers": {}}
    days_list = [14, 35, 63, 98][: c["n_expiries"]]
    for ticker in c["tickers"]:
        S = SYNTH_SPOT.get(ticker, 100.0)
        atm = SYNTH_ATM.get(ticker, 0.2)
        q = float(c["dividend_yield"].get(ticker, 0.0))
        n0 = len(rows)
        for days in days_list:
            T = days / 365.0
            F = S * math.exp((rate - q) * T)
            expiry = (today + dt.timedelta(days=days)).isoformat()
            step = max(round(S * 0.01, 1), 0.5)
            for k in np.arange(math.floor(0.82 * S / step) * step, 1.18 * S, step):
                vol = refprice.sabr_vol_beta1(atm, -0.6, 0.7, F, float(k), T)
                for cp in ("c", "p"):
                    mid = refprice.bs_price(cp, S, float(k), T, rate, q, vol)
                    if mid < 0.05:
                        continue
                    half = max(0.005, 0.01 * mid)
                    rows.append((ticker, expiry, days, T, cp, float(k), mid - half, mid + half, mid, mid, 100.0,
                                 1000.0, S, q, rate, vol))
        meta["tickers"][ticker] = {"status": "ok", "options": len(rows) - n0, "spot": S}
    df = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
    meta["rows"] = int(len(df))
    return df, meta


# ----------------------------------------------------------------------------------------------------------------
# context


@dataclass
class LabContext:
    cfg: dict
    today: dt.date
    out: Path
    rate: float = 0.04
    rate_source: str = "constant"
    contracts: list[Contract] = field(default_factory=list)
    _chains: pd.DataFrame | None = None
    _prices: dict[str, pd.DataFrame] = field(default_factory=dict)
    price_source: dict[str, str] = field(default_factory=dict)

    # -- chains ------------------------------------------------------------------------------------------------
    def chains_path(self) -> Path:
        return self.out / "chains.parquet"

    def chains(self) -> pd.DataFrame:
        """The chains saved by the `chains` job, or an empty frame (never raises)."""
        if self._chains is None:
            try:
                self._chains = pd.read_parquet(self.chains_path()) if self.chains_path().exists() else _empty_chains()
            except Exception as exc:
                log.warning("chains.parquet unreadable: %s", clip(exc))
                self._chains = _empty_chains()
        return self._chains

    def chains_meta(self) -> dict:
        return read_json(self.out / "chains_meta.json", {"source": "none", "rows": 0, "tickers": {}})

    def has_chains(self) -> bool:
        return not self.chains().empty

    # -- price data --------------------------------------------------------------------------------------------
    def prices(self, ids: list[str]) -> dict[str, pd.DataFrame]:
        """Daily OHLC per instrument id: the data cache when present, else deterministic synthetic bars."""
        from .. import data as datamod, synthetic
        universe = config.load_universe()
        for inst_id in ids:
            if inst_id in self._prices:
                continue
            df, src = None, "synthetic"
            try:
                p = datamod.cache_path(inst_id)
                if p.exists():
                    cached = pd.read_parquet(p)
                    if len(cached) >= 300:
                        df, src = cached, "cache"
            except Exception as exc:
                log.warning("price cache for %s unreadable: %s", inst_id, clip(exc))
            if df is None:
                cls = universe[inst_id].asset_class if inst_id in universe else "index"
                df = synthetic.generate(inst_id, cls, 3000)
            self._prices[inst_id] = df
            self.price_source[inst_id] = src
        return {i: self._prices[i] for i in ids}

    def timeout(self) -> float:
        return float(self.cfg.get("timeout_s", 120))


def build_context(cfg: dict | None = None, today: dt.date | None = None) -> LabContext:
    cfg = cfg or load_config()
    ctx = LabContext(cfg=cfg, today=today or today_utc(), out=out_dir(), contracts=build_test_set(cfg))
    stored = read_json(ctx.out / "testset.json")
    if stored:
        ctx.rate, ctx.rate_source = float(stored.get("rate", ctx.rate)), stored.get("rate_source", ctx.rate_source)
    else:
        ctx.rate = float(cfg["rates"]["default"])
    return ctx


def prepare_inputs(cfg: dict | None = None, synthetic_chains_flag: bool = False, today: dt.date | None = None) -> dict:
    """The `chains` job: fetch the rate and chains, write chains.parquet, chains_meta.json and testset.json."""
    cfg = cfg or load_config()
    today = today or today_utc()
    out = out_dir()
    rate, rate_source = fetch_rate(cfg)
    if synthetic_chains_flag:
        df, meta = synthetic_chains(cfg, today, rate)
    else:
        df, meta = fetch_chains(cfg, today, rate)
    if not df.empty:
        df.to_parquet(out / "chains.parquet")
    elif (out / "chains.parquet").exists():
        (out / "chains.parquet").unlink()
    meta["rate"], meta["rate_source"] = rate, rate_source
    write_json(out / "chains_meta.json", meta)
    contracts = build_test_set(cfg)
    write_json(out / "testset.json", {"rate": rate, "rate_source": rate_source, "asof": today.isoformat(),
                                      "contracts": [c.__dict__ for c in contracts], "n": len(contracts)})
    return meta
