"""bbcho/finoptions-dev (port of R fOptions): GBS closed form, CRR tree for American style, Sobol Monte Carlo.

Cost of carry b = r - q. Exotic pricers (barrier, Asian) are not part of the agreement table; one Asian price is
recorded as an illustration.
"""
from __future__ import annotations

import numpy as np

from ..context import LabContext
from ._common import price_test_set, version_of

TREE_STEPS = 250


def run(ctx: LabContext) -> dict:
    import finoptions as fo
    from finoptions.monte_carlo_options import (ArithmeticAsianPayoff, MonteCarloOption, NormalSobolInnovations,
                                                PlainVanillaPayoff, WienerPath)

    def european(c) -> float:
        o = fo.GBSOption(c.S, c.K, c.T, c.r, c.r - c.q, c.sigma)
        return o.call() if c.cp == "c" else o.put()

    def american(c) -> float:
        o = fo.binomial_tree_options.CRRBinomialTreeOption(c.S, c.K, c.T, c.r, c.r - c.q, c.sigma, type="american", n=TREE_STEPS)
        return o.call() if c.cp == "c" else o.put()

    def mc_option(c, payoff):
        return MonteCarloOption(mc_loops=20, path_length=64, mc_paths=2000, S=c.S, K=c.K, t=c.T, r=c.r, b=c.r - c.q,
                                sigma=c.sigma, Innovation=NormalSobolInnovations, Path=WienerPath, Payoff=payoff,
                                antithetic=True, standardization=True)

    def mc(c) -> float:
        o = mc_option(c, PlainVanillaPayoff)
        return float(np.mean(o.call() if c.cp == "c" else o.put()))

    out = price_test_set(ctx, european=european, american=american,
                         methods={"european": "GBS closed form", "american": f"CRR binomial tree, {TREE_STEPS} steps"})
    mc_out = price_test_set(ctx, european=mc, methods={"european": "Sobol Monte Carlo, antithetic, 20 x 2000 paths"}, stochastic=True)
    out["prices"]["european:mc"] = mc_out["prices"]["european"]
    out["methods"]["european:mc"] = mc_out["methods"]["european"]
    out["errors"].update({f"mc:{k}": v for k, v in mc_out["errors"].items()})
    out["stochastic_styles"] = ["european:mc"]
    try:
        atm = next(c for c in ctx.contracts if c.cp == "c" and abs(c.moneyness - 1.0) < 1e-9)
        out["asian_atm_call"] = float(np.mean(mc_option(atm, ArithmeticAsianPayoff).call()))
    except Exception as exc:
        out["asian_atm_call_error"] = f"{type(exc).__name__}: {exc}"[:120]
    out["version"] = version_of("finoptions")
    return out
