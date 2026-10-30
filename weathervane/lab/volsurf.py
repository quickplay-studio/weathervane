"""Implied-volatility analytics on the saved option chains.

Per-row implied vol and delta (computed by the vollib adapter, or by the reference pricer when vollib is missing),
ATM interpolation to a fixed tenor, 25-delta risk reversal, term slope, IV history and the per-ticker dashboard rows.
Everything works on plain DataFrames and tolerates empty or partial chains.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from . import refprice

IV_COLUMNS = ["iv", "delta", "iv_source"]
HISTORY_MIN_OBS = 20
HISTORY_WINDOW = 252


def chain_iv_path(out: Path) -> Path:
    return out / "chain_iv.parquet"


def history_path() -> Path:
    return config.path("data", "lab", "iv_history.parquet")


# ----------------------------------------------------------------------------------------------------------------
# per-row implied vol


def compute_chain_iv(chains: pd.DataFrame, iv_fn: Callable | None = None, source: str = "reference") -> pd.DataFrame:
    """Add iv and delta to every chain row. iv_fn(cp, price, S, K, T, r, q) -> vol; NaN on failure (default: bisection)."""
    if chains.empty:
        return chains.assign(iv=pd.Series(dtype=float), delta=pd.Series(dtype=float), iv_source=pd.Series(dtype=object))
    fn = iv_fn or (lambda cp, p, S, K, T, r, q: refprice.implied_vol(cp, p, S, K, T, r, q))
    ivs, deltas = [], []
    for row in chains.itertuples(index=False):
        try:
            iv = float(fn(row.cp, float(row.mid), float(row.spot), float(row.strike), float(row.T), float(row.r), float(row.q)))
        except Exception:
            iv = float("nan")
        if not math.isfinite(iv) or iv <= 0.005 or iv > 4.0:
            iv = float("nan")
        ivs.append(iv)
        if math.isfinite(iv):
            deltas.append(refprice.bs_greeks(row.cp, float(row.spot), float(row.strike), float(row.T), float(row.r), float(row.q), iv)["delta"])
        else:
            deltas.append(float("nan"))
    out = chains.copy()
    out["iv"], out["delta"], out["iv_source"] = ivs, deltas, source
    return out


def load_chain_iv(out: Path, chains: pd.DataFrame) -> pd.DataFrame:
    """chain_iv.parquet when the vollib adapter wrote it, else computed with the reference pricer."""
    p = chain_iv_path(out)
    if p.exists():
        try:
            df = pd.read_parquet(p)
            if not df.empty and "iv" in df.columns:
                return df
        except Exception:
            pass
    return compute_chain_iv(chains)


# ----------------------------------------------------------------------------------------------------------------
# smile and term structure


def otm_curve(expiry_rows: pd.DataFrame) -> pd.DataFrame:
    """Out-of-the-money quotes only (puts below the forward, calls at or above), sorted by strike, valid IV."""
    if expiry_rows.empty:
        return expiry_rows
    S, T, r, q = (float(expiry_rows[c].iloc[0]) for c in ("spot", "T", "r", "q"))
    fwd = S * math.exp((r - q) * T)
    keep = ((expiry_rows["cp"] == "p") & (expiry_rows["strike"] < fwd)) | ((expiry_rows["cp"] == "c") & (expiry_rows["strike"] >= fwd))
    df = expiry_rows[keep & expiry_rows["iv"].notna()].sort_values("strike")
    return df.drop_duplicates("strike", keep="first")


def forward(df: pd.DataFrame) -> float:
    S, T, r, q = (float(df[c].iloc[0]) for c in ("spot", "T", "r", "q"))
    return S * math.exp((r - q) * T)


def atm_iv(expiry_rows: pd.DataFrame) -> float:
    """IV at the forward strike, interpolated between the nearest OTM quotes on each side."""
    c = otm_curve(expiry_rows)
    if len(c) < 2:
        return float("nan")
    f = forward(expiry_rows)
    k, v = c["strike"].to_numpy(float), c["iv"].to_numpy(float)
    if k.min() > f or k.max() < f:
        return float("nan")
    return float(np.interp(f, k, v))


def term_structure(ticker_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per expiry: days, T, forward, ATM IV (NaN rows dropped)."""
    rows = []
    for (expiry, days), g in ticker_rows.groupby(["expiry", "days"]):
        v = atm_iv(g)
        if math.isfinite(v):
            rows.append({"expiry": expiry, "days": int(days), "T": float(g["T"].iloc[0]), "forward": forward(g), "atm_iv": v})
    return pd.DataFrame(rows, columns=["expiry", "days", "T", "forward", "atm_iv"]).sort_values("days").reset_index(drop=True)


def interp_tenor(term: pd.DataFrame, days: float, col: str = "atm_iv") -> float:
    """Value at a fixed tenor: linear in total variance between bracketing expiries, flat outside the range."""
    if term.empty:
        return float("nan")
    d = term["days"].to_numpy(float)
    v = term[col].to_numpy(float)
    if len(term) == 1 or days <= d[0]:
        return float(v[0])
    if days >= d[-1]:
        return float(v[-1])
    hi = int(np.searchsorted(d, days))
    lo = hi - 1
    if col != "atm_iv":
        w = (days - d[lo]) / (d[hi] - d[lo])
        return float(v[lo] * (1 - w) + v[hi] * w)
    t = d / 365.0
    tv = days / 365.0
    var_lo, var_hi = v[lo] ** 2 * t[lo], v[hi] ** 2 * t[hi]
    w = (tv - t[lo]) / (t[hi] - t[lo])
    return float(math.sqrt(max((var_lo * (1 - w) + var_hi * w) / tv, 0.0)))


def iv_at_delta(expiry_rows: pd.DataFrame, target: float) -> float:
    """IV at a given delta (+0.25 for the call wing, -0.25 for the put wing), interpolated along the OTM quotes."""
    c = otm_curve(expiry_rows)
    side = c[c["cp"] == ("c" if target > 0 else "p")].dropna(subset=["delta"])
    if len(side) < 2:
        return float("nan")
    d, v = side["delta"].to_numpy(float), side["iv"].to_numpy(float)
    order = np.argsort(d)
    d, v = d[order], v[order]
    if target < d.min() or target > d.max():
        return float("nan")
    return float(np.interp(target, d, v))


def risk_reversals(ticker_rows: pd.DataFrame, target: float = 0.25) -> pd.DataFrame:
    """25-delta risk reversal (call IV minus put IV) per expiry."""
    rows = []
    for (expiry, days), g in ticker_rows.groupby(["expiry", "days"]):
        cv, pv = iv_at_delta(g, target), iv_at_delta(g, -target)
        if math.isfinite(cv) and math.isfinite(pv):
            rows.append({"expiry": expiry, "days": int(days), "call_iv": cv, "put_iv": pv, "rr": cv - pv})
    return pd.DataFrame(rows, columns=["expiry", "days", "call_iv", "put_iv", "rr"]).sort_values("days").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------
# history


def read_history() -> pd.DataFrame:
    p = history_path()
    try:
        if p.exists():
            h = pd.read_parquet(p)
            return h[["date", "ticker", "atm_iv_30d"]]
    except Exception:
        pass
    return pd.DataFrame(columns=["date", "ticker", "atm_iv_30d"])


def update_history(today: str, values: dict[str, float], write: bool = True) -> pd.DataFrame:
    """Upsert today's ATM 30d IV per ticker into data/lab/iv_history.parquet and return the full history."""
    hist = read_history()
    new = pd.DataFrame([{"date": today, "ticker": t, "atm_iv_30d": v} for t, v in values.items() if v is not None and math.isfinite(v)])
    if not new.empty:
        hist = pd.concat([hist[~((hist["date"] == today) & hist["ticker"].isin(new["ticker"]))], new], ignore_index=True)
        hist = hist.sort_values(["ticker", "date"]).reset_index(drop=True)
        if write:
            p = history_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            hist.to_parquet(p)
    return hist


def rank_percentile(hist: pd.DataFrame, ticker: str, value: float) -> dict:
    """IV rank (position in the min-max range) and percentile over the trailing history of one ticker."""
    h = hist[hist["ticker"] == ticker]["atm_iv_30d"].dropna().to_numpy(float)[-HISTORY_WINDOW:]
    out = {"history_days": int(len(h)), "iv_rank": None, "iv_percentile": None}
    if len(h) < HISTORY_MIN_OBS or not math.isfinite(value):
        return out
    lo, hi = float(h.min()), float(h.max())
    out["iv_rank"] = round(100.0 * (value - lo) / (hi - lo), 1) if hi > lo else 50.0
    out["iv_percentile"] = round(100.0 * (float((h < value).mean()) + 0.5 * float((h == value).mean())), 1)
    return out


# ----------------------------------------------------------------------------------------------------------------
# dashboard


def pick_sabr(sabr_ticker: dict | None, days: float = 30.0) -> dict:
    """The SABR fit for the expiry nearest to `days`, as {rho, nu, days}; empty when none."""
    if not sabr_ticker:
        return {}
    best = min(sabr_ticker.values(), key=lambda f: abs(float(f.get("days", 0)) - days))
    return {"rho": best.get("rho"), "nu": best.get("nu"), "days": best.get("days"), "alpha": best.get("alpha")}


def dashboard(chain_iv: pd.DataFrame, sabr: dict | None, today: str, history: pd.DataFrame | None = None,
              tenor: float = 30.0, long_tenor: float = 90.0) -> dict:
    """Per-ticker dashboard rows. `sabr` is the pysabr adapter's {ticker: {expiry: fit}} block (may be empty)."""
    rows: dict[str, dict] = {}
    if chain_iv is None or chain_iv.empty:
        return rows
    atm30: dict[str, float] = {}
    for ticker, g in chain_iv.groupby("ticker"):
        term = term_structure(g)
        spot = float(g["spot"].iloc[0])
        if term.empty:
            rows[ticker] = {"spot": spot, "note": "no usable ATM quotes"}
            continue
        iv30 = interp_tenor(term, tenor)
        iv90 = interp_tenor(term, long_tenor)
        rr = risk_reversals(g)
        rr30 = interp_tenor(rr, tenor, "rr") if not rr.empty else float("nan")
        move = spot * iv30 * math.sqrt(7.0 / 365.0)
        atm30[ticker] = iv30
        rows[ticker] = {
            "spot": spot, "atm_iv_30d": iv30, "atm_iv_90d": iv90, "term_slope": iv90 - iv30,
            "rr25_30d": rr30 if math.isfinite(rr30) else None,
            "implied_move_1w": move, "implied_move_1w_pct": 100.0 * move / spot,
            "expiries": int(len(term)), "min_days": int(term["days"].min()), "max_days": int(term["days"].max()),
            "sabr": pick_sabr((sabr or {}).get(ticker), tenor),
        }
    hist = update_history(today, atm30) if history is None else history
    for ticker, row in rows.items():
        if "atm_iv_30d" in row:
            row.update(rank_percentile(hist, ticker, row["atm_iv_30d"]))
    return rows
