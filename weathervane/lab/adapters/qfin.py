"""RomanMichaelPaolucci/Q-Fin (`qfin`): Black-Scholes classes and a path-simulating Monte Carlo.

qfin has no dividend yield, so the spot is passed as S * exp(-q T), which is the exact Black-Scholes-Merton equivalent.
Monte Carlo prices come from the library's MonteCarloCall/Put with a fixed seed and are flagged as stochastic.
"""
from __future__ import annotations

import math

import numpy as np

from .. import refprice
from ..context import LabContext
from ._common import price_test_set, version_of

MC_PATHS = 4000


def run(ctx: LabContext) -> dict:
    import qfin

    def adjusted(c) -> float:
        return c.S * math.exp(-c.q * c.T)

    def european(c) -> float:
        cls = qfin.BlackScholesCall if c.cp == "c" else qfin.BlackScholesPut
        return float(cls(adjusted(c), c.sigma, c.K, c.T, c.r).price)

    out = price_test_set(ctx, european=european,
                         methods={"european": "analytic Black-Scholes (spot discounted by dividend yield)"})

    def mc(c) -> float:
        np.random.seed(7)
        cls = qfin.MonteCarloCall if c.cp == "c" else qfin.MonteCarloPut
        steps = max(int(round(c.T * 365)), 1)
        return float(cls(c.K, MC_PATHS, c.r, adjusted(c), c.r, c.sigma, c.T / steps, c.T).price)

    mc_out = price_test_set(ctx, european=mc, methods={"european": f"GBM Monte Carlo, {MC_PATHS} paths, seed 7"}, stochastic=True)
    out["prices"]["european:mc"] = mc_out["prices"]["european"]
    out["methods"]["european:mc"] = mc_out["methods"]["european"]
    out["errors"].update({f"mc:{k}": v for k, v in mc_out["errors"].items()})
    out["stochastic_styles"] = ["european:mc"]
    out["version"] = version_of("qfin")
    out["known_issue"] = ("qfin's closed-form Black-Scholes leaves the interest rate out of d1 and d2 (calls drift by about 0.02 at r = 4%), "
                          "and its put price is wrong away from the money even at r = 0 (negative values for in-the-money puts); "
                          "the table shows the library's actual output")
    zero = [c for c in ctx.contracts if abs(c.moneyness - 1.0) < 1e-9]
    gaps = []
    for c in zero:
        cls = qfin.BlackScholesCall if c.cp == "c" else qfin.BlackScholesPut
        gaps.append(abs(float(cls(c.S, c.sigma, c.K, c.T, 0.0).price) - refprice.bs_price(c.cp, c.S, c.K, c.T, 0.0, 0.0, c.sigma)))
    out["zero_rate_max_abs_dev"] = max(gaps) if gaps else None
    return out
