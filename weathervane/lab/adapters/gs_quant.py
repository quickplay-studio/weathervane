"""goldmansachs/gs-quant: offline `gs_quant.timeseries` functions checked against the core indicators.

Nothing here needs a Marquee session. Only when GS_CLIENT_ID and GS_CLIENT_SECRET are both set does the adapter try to
open one (read-only scope) and record whether it worked; the secrets are never written anywhere.
"""
from __future__ import annotations

import os

import numpy as np

from .. import crosscheck as cc
from ...indicators import bollinger, rsi
from ..context import LabContext
from ._common import version_of

N = 20


def _marquee() -> dict:
    cid, secret = os.environ.get("GS_CLIENT_ID"), os.environ.get("GS_CLIENT_SECRET")
    if not (cid and secret):
        return {"used": False, "note": "GS_CLIENT_ID and GS_CLIENT_SECRET not set; offline timeseries only"}
    try:
        from gs_quant.session import Environment, GsSession
        GsSession.use(Environment.PROD, client_id=cid, client_secret=secret, scopes=("read_product_data",))
        return {"used": True, "session": "ok"}
    except Exception as exc:  # report the failure class only, the message could echo request details
        return {"used": True, "session": f"failed: {type(exc).__name__}"}


def run(ctx: LabContext) -> dict:
    from gs_quant.timeseries import Returns, Window
    from gs_quant.timeseries.technicals import bollinger_bands, moving_average, relative_strength_index
    from gs_quant.timeseries.technicals import returns as gs_returns
    from gs_quant.timeseries import volatility, zscores

    def build(f: cc.Frame) -> list[cc.Check]:
        px = f.close_series
        window = Window(N, 0)
        core_sma = cc.core_sma(f, N)
        core_std = cc.core_sample_std(f, N)
        basis, upper_pop, lower_pop = bollinger(f.close, N, 2.0)
        scale = cc.SAMPLE_FACTOR(N)
        bands = bollinger_bands(px, window, 2)
        values = np.asarray(bands, float)
        checks = [
            cc.Check("sma20", core_sma, cc.to_array(moving_average(px, window)), "moving_average"),
            cc.Check("rsi14", rsi(f.close, 14), cc.to_array(relative_strength_index(px, 14)), "relative_strength_index (Wilder)",
                     skip=250),
            cc.Check("returns1", cc.simple_returns(f), cc.to_array(gs_returns(px, 1, Returns.SIMPLE)), "returns(SIMPLE)"),
            cc.Check("volatility20", cc.core_annual_vol(f, N), cc.to_array(volatility(px, window)),
                     "annualised percent, sample stdev of simple returns"),
            cc.Check("zscore20", (f.close - core_sma) / core_std, cc.to_array(zscores(px, window)), "zscores, sample stdev"),
        ]
        if values.ndim == 2 and values.shape[1] == 2:
            checks.append(cc.Check("bollinger_upper", basis + (upper_pop - basis) * scale, values.max(axis=1),
                                   "bollinger_bands upper; core band width x sqrt(n/(n-1))"))
            checks.append(cc.Check("bollinger_lower", basis - (basis - lower_pop) * scale, values.min(axis=1),
                                   "bollinger_bands lower; core band width x sqrt(n/(n-1))"))
        return checks

    out = cc.run_instruments(ctx, build)
    out["marquee"] = _marquee()
    out["version"] = version_of("gs-quant", "gs_quant")
    return out
