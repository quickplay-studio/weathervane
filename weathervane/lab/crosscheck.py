"""Indicator cross-checks: compare the core indicators with an independent library's implementation.

A check pairs a core array with the library's array. Libraries differ in warm-up rows and in conventions (sample versus
population standard deviation, partial windows), so each check states how the core series was converted to the
library's definition. Anything above `rel_flag` after that conversion is flagged. Relative difference is the largest
absolute difference divided by the mean absolute level of the core series.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .. import indicators as ind

SAMPLE_FACTOR = lambda n: math.sqrt(n / (n - 1.0))   # population stdev -> sample stdev   # noqa: E731


@dataclass
class Check:
    name: str
    core: np.ndarray
    other: "np.ndarray | Callable[[], np.ndarray]"     # a callable is run guarded: one failing library call must not hide the rest
    note: str = ""
    skip: int = 0                 # leading rows ignored (recursive indicators that need time to converge)
    definitions_match: bool = True


@dataclass
class Frame:
    """The series every checker needs, as float64 arrays plus the dated close."""
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    index: pd.DatetimeIndex
    instrument: str = ""
    close_series: pd.Series = field(init=False)

    def __post_init__(self) -> None:
        self.close_series = pd.Series(self.close, index=self.index, name="close")


def frame(df: pd.DataFrame, tail: int, instrument: str = "") -> Frame:
    d = df.tail(tail)
    return Frame(d["close"].to_numpy(float), d["high"].to_numpy(float), d["low"].to_numpy(float), pd.DatetimeIndex(d.index),
                 instrument)


def to_array(x) -> np.ndarray:
    """A Series, one-column DataFrame or array as a float64 vector."""
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return np.asarray(x, dtype=float).ravel()


def compare(core: np.ndarray, other: np.ndarray, skip: int = 0) -> dict:
    """Align on the last rows, ignore NaN pairs and the first `skip` rows, return max abs / relative difference."""
    core, other = np.asarray(core, float), np.asarray(other, float)
    m = min(len(core), len(other))
    if m == 0:
        return {"n": 0, "max_abs": None, "max_rel": None}
    a, b = core[-m:], other[-m:]
    ok = np.isfinite(a) & np.isfinite(b)
    ok[:skip] = False
    if not ok.any():
        return {"n": 0, "max_abs": None, "max_rel": None}
    diff = np.abs(a[ok] - b[ok])
    scale = float(np.mean(np.abs(a[ok])))
    max_abs = float(diff.max())
    return {"n": int(ok.sum()), "max_abs": max_abs, "max_rel": max_abs / scale if scale > 0 else (0.0 if max_abs == 0 else float("inf"))}


def evaluate(checks: list[Check], rel_flag: float) -> dict:
    out = {}
    for c in checks:
        try:
            other = c.other() if callable(c.other) else c.other
            res = compare(c.core, other, c.skip)
            res["flagged"] = bool(c.definitions_match and res["max_rel"] is not None and res["max_rel"] > rel_flag)
        except Exception as exc:
            res = {"n": 0, "max_abs": None, "max_rel": None, "flagged": False, "error": f"{type(exc).__name__}: {exc}"[:120]}
        res["definitions_match"] = c.definitions_match
        res["note"] = c.note
        out[c.name] = res
    return out


def run_instruments(ctx, builder: Callable[[Frame], list[Check]]) -> dict:
    """Run `builder` on each configured instrument and gather {instrument: {check: result}} plus a summary."""
    cc = ctx.cfg["crosscheck"]
    ids = list(cc["instruments"])
    data = ctx.prices(ids)
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for inst in ids:
        try:
            results[inst] = evaluate(builder(frame(data[inst], int(cc["tail"]), inst)), float(cc["rel_flag"]))
        except Exception as exc:  # a failing instrument must not hide the others
            errors[inst] = f"{type(exc).__name__}: {exc}"[:150]
    if not results:
        raise RuntimeError("no instrument could be checked: " + "; ".join(errors.values()))
    flagged = [f"{i}:{n}" for i, checks in results.items() for n, r in checks.items() if r["flagged"]]
    rels = [r["max_rel"] for checks in results.values() for r in checks.values() if r["definitions_match"] and r["max_rel"] is not None]
    failed_checks = [f"{i}:{n}" for i, checks in results.items() for n, r in checks.items() if r.get("error")]
    return {"failed_checks": failed_checks, "instruments": results, "flagged": flagged, "max_rel_matching": max(rels) if rels else None, "errors": errors,
            "data_source": dict(ctx.price_source), "rel_flag": float(cc["rel_flag"])}


# ---------------------------------------------------------------------------------------------------------------
# core-side definitions shared by several checkers


def core_sma(f: Frame, n: int = 20) -> np.ndarray:
    return ind.sma(f.close, n)


def core_sample_std(f: Frame, n: int = 20) -> np.ndarray:
    """Core stdev is the population one (Pine); libraries mostly use the sample one."""
    return ind.stdev(f.close, n) * SAMPLE_FACTOR(n)


def simple_returns(f: Frame) -> np.ndarray:
    return ind.roc(f.close, 1) / 100.0


def core_annual_vol(f: Frame, n: int = 20, periods: int = 252) -> np.ndarray:
    """Annualised percent volatility of simple returns with the sample stdev (what gs-quant's volatility() computes)."""
    r = simple_returns(f)
    out = np.full(len(r), np.nan)
    out[1:] = ind.stdev(r[1:], n) * SAMPLE_FACTOR(n) * math.sqrt(periods) * 100.0
    return out
