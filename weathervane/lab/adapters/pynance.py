"""GriffinAustin/pynance: returns, moving averages and rolling volatility checked against the core indicators.

pynance expects a frame with an 'Adj Close' column (its data module scrapes Yahoo, which is not used here). Its
volatility() is the rolling sample standard deviation of the selected column, so the core population stdev is converted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import crosscheck as cc
from ...indicators import ema
from ..context import LabContext
from ._common import version_of

N = 20


def run(ctx: LabContext) -> dict:
    from pynance.tech import movave, simple

    def build(f: cc.Frame) -> list[cc.Check]:
        eq = pd.DataFrame({"Adj Close": f.close}, index=f.index)
        def returns() -> np.ndarray:
            # ret() is growth() - 1 but edits a read-only array under pandas 3; growth() - 1 is the same definition
            growth = cc.to_array(simple.growth(eq, n_sessions=1))
            return np.concatenate([[np.nan], growth - 1.0])       # pynance drops the first row

        return [
            cc.Check("returns1", cc.simple_returns(f), returns, "simple.growth(n_sessions=1) - 1 (= simple.ret)"),
            cc.Check("sma20", cc.core_sma(f, N), lambda: cc.to_array(movave.sma(eq, window=N)), "movave.sma"),
            cc.Check("ema20", ema(f.close, N), lambda: cc.to_array(movave.ema(eq, span=N)),
                     "movave.ema (pandas ewm adjust=True; core seeds with the sma, so compared after a 250-bar warm-up)", skip=250),
            cc.Check("stdev20_sample", cc.core_sample_std(f, N), lambda: cc.to_array(movave.volatility(eq, window=N)),
                     "movave.volatility is the rolling sample stdev of the price"),
        ]

    out = cc.run_instruments(ctx, build)
    out["version"] = version_of("pynance")
    return out
