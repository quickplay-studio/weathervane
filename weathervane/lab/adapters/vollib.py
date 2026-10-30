"""vollib/vollib (Black-Scholes-Merton, implied volatility via Jaeckel's LetsBeRational, analytical Greeks).

Prices the 12 test contracts and, when chains exist, computes implied vol and delta for every quote. The result is saved
as out/lab/chain_iv.parquet, which the SABR fit and the volatility dashboard read.
"""
from __future__ import annotations

from .. import volsurf
from ..context import LabContext
from ._common import price_test_set, version_of


def run(ctx: LabContext) -> dict:
    from vollib.black_scholes_merton import black_scholes_merton
    from vollib.black_scholes_merton.greeks import analytical as greeks
    from vollib.black_scholes_merton.implied_volatility import implied_volatility

    def european(c) -> float:
        return black_scholes_merton(c.cp, c.S, c.K, c.T, c.r, c.sigma, c.q)

    out = price_test_set(ctx, european=european, methods={"european": "analytic Black-Scholes-Merton"})
    out["greeks"] = {c.id: {"delta": greeks.delta(c.cp, c.S, c.K, c.T, c.r, c.sigma, c.q),
                            "gamma": greeks.gamma(c.cp, c.S, c.K, c.T, c.r, c.sigma, c.q),
                            "vega": greeks.vega(c.cp, c.S, c.K, c.T, c.r, c.sigma, c.q)} for c in ctx.contracts}
    out["version"] = version_of("vollib", "py_vollib")
    # implied vol round trip on the test set: price -> IV must return the input vol
    back = []
    for c in ctx.contracts:
        iv = implied_volatility(european(c), c.S, c.K, c.T, c.r, c.q, c.cp)
        back.append(abs(iv - c.sigma))
    out["iv_roundtrip_max_abs"] = max(back)

    chains = ctx.chains()
    if chains.empty:
        out["chain_iv"] = {"rows": 0, "valid": 0, "note": "no chains available"}
        return out
    df = volsurf.compute_chain_iv(
        chains, lambda cp, price, S, K, T, r, q: implied_volatility(price, S, K, T, r, q, cp), source="vollib")
    df.to_parquet(volsurf.chain_iv_path(ctx.out))
    out["chain_iv"] = {"rows": int(len(df)), "valid": int(df["iv"].notna().sum()), "path": "out/lab/chain_iv.parquet"}
    return out
