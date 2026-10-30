"""dbrojas/optlib: only the pure-math optlib.gbs module (never optlib.api, which talks to a broker API).

merton() is Black-Scholes with a continuous dividend yield; american() is the Bjerksund-Stensland (2002) closed-form
approximation, so its American column is an approximation and not a tree. Needs scipy.stats.mvn, which newer SciPy
removed: install with the legacy requirements (Python 3.10, SciPy 1.11).
"""
from __future__ import annotations

from ..context import LabContext
from ._common import price_test_set, version_of


def run(ctx: LabContext) -> dict:
    from optlib import gbs

    def european(c) -> float:
        return float(gbs.merton(c.cp, c.S, c.K, c.T, c.r, c.q, c.sigma)[0])

    def american(c) -> float:
        return float(gbs.american(c.cp, c.S, c.K, c.T, c.r, c.q, c.sigma)[0])

    out = price_test_set(ctx, european=european, american=american,
                         methods={"european": "gbs.merton (Black-Scholes-Merton)",
                                  "american": "gbs.american (Bjerksund-Stensland 2002 approximation)"})
    out["version"] = version_of("optlib")
    return out
