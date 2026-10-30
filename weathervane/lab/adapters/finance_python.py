"""alpha-miner/Finance-Python (import name `PyFin`): rolling operators checked against the core indicators.

Its operators return values on partial windows at the start and MSTD is the sample standard deviation, so the core
population stdev is converted with sqrt(n / (n - 1)) before comparing. Python 3.10, built from a pinned git commit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import crosscheck as cc
from ...indicators import highest, lowest
from ..context import LabContext
from ._common import version_of

N = 20


def run(ctx: LabContext) -> dict:
    from PyFin.api import MA, MMAX, MMIN, MSTD

    def transform(expr, df: pd.DataFrame) -> np.ndarray:
        return cc.to_array(expr.transform(df.copy()))

    def build(f: cc.Frame) -> list[cc.Check]:
        df = pd.DataFrame({"x": f.close, "hi": f.high, "lo": f.low}, index=range(len(f.close)))
        return [
            cc.Check("sma20", cc.core_sma(f, N), transform(MA(N, "x"), df), "MA(20)"),
            cc.Check("stdev20_sample", cc.core_sample_std(f, N), transform(MSTD(N, "x"), df),
                     "MSTD is the sample stdev; core population stdev x sqrt(n/(n-1))"),
            cc.Check("highest20", highest(f.high, N), transform(MMAX(N, "hi"), df), "MMAX(20) of the high"),
            cc.Check("lowest20", lowest(f.low, N), transform(MMIN(N, "lo"), df), "MMIN(20) of the low"),
        ]

    out = cc.run_instruments(ctx, build)
    out["version"] = version_of("Finance-Python", "finance-python")
    return out
