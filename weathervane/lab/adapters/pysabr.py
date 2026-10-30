"""ynouri/pysabr (Hagan 2002 lognormal SABR).

Test set: for each expiry the model is fitted to a flat 20% smile (the fit must return alpha 0.2 and nu near 0), and
each contract is priced as a Black call or put at the SABR vol of its strike. Chains: one fit per ticker and expiry on
the out-of-the-money implied vols (beta fixed), giving rho and nu for the dashboard.
"""
from __future__ import annotations

import math

import numpy as np

from .. import volsurf
from ..context import LabContext
from ..util import clip
from ._common import price_test_set, version_of

MIN_POINTS = 6


def _fit(sabr_cls, f: float, t: float, strikes, vols, beta: float, atm: float, guess: list[float]) -> tuple[float, float, float]:
    model = sabr_cls(f=f, shift=0.0, t=t, v_atm_n=atm * f, beta=beta)
    alpha, rho, nu = model.fit(np.asarray(strikes, float), np.asarray(vols, float) * 100.0, initial_guess=guess)
    return float(alpha), float(rho), float(nu)


def run(ctx: LabContext) -> dict:
    import pysabr.models.hagan_2002_lognormal_sabr as hagan
    from pysabr import Hagan2002LognormalSABR

    beta = float(ctx.cfg["sabr"]["beta"])
    fits: dict[float, tuple[float, float, float]] = {}

    def params_for(c) -> tuple[float, float, float, float]:
        f = c.S * math.exp((c.r - c.q) * c.T)
        key = round(c.T, 9)
        if key not in fits:
            ks = f * np.linspace(0.85, 1.15, 11)
            fits[key] = _fit(Hagan2002LognormalSABR, f, c.T, ks, np.full(len(ks), c.sigma), beta, c.sigma, [c.sigma, 0.0, 0.3])
        return (f, *fits[key])

    def european(c) -> float:
        f, alpha, rho, nu = params_for(c)
        vol = hagan.lognormal_vol(c.K, f, c.T, alpha, beta, rho, nu)
        return hagan.black.lognormal_call(c.K, f, c.T, vol, c.r, "call" if c.cp == "c" else "put")

    out = price_test_set(ctx, european=european, methods={"european": "Black price at the fitted SABR vol (flat smile)"})
    out["test_set_fit"] = {f"{k:.5f}": {"alpha": v[0], "rho": v[1], "nu": v[2]} for k, v in fits.items()}
    out["beta"] = beta
    out["version"] = version_of("pysabr")

    chains = ctx.chains()
    sabr: dict[str, dict] = {}
    out["chain_fits"] = {"fitted": 0, "skipped": 0}
    if chains.empty:
        out["sabr"] = sabr
        out["note"] = "no chains: only the test-set fit ran"
        return out
    civ = volsurf.load_chain_iv(ctx.out, chains)
    for ticker, g in civ.groupby("ticker"):
        for (expiry, days), eg in g.groupby(["expiry", "days"]):
            curve = volsurf.otm_curve(eg)
            if len(curve) < MIN_POINTS:
                out["chain_fits"]["skipped"] += 1
                continue
            f, T = volsurf.forward(eg), float(eg["T"].iloc[0])
            k, v = curve["strike"].to_numpy(float), curve["iv"].to_numpy(float)
            atm = float(np.interp(f, k, v)) if k.min() <= f <= k.max() else float(np.median(v))
            try:
                alpha, rho, nu = _fit(Hagan2002LognormalSABR, f, T, k, v, beta, atm, [atm, -0.3, 0.8])
                model = np.array([hagan.lognormal_vol(x, f, T, alpha, beta, rho, nu) for x in k])
                rmse = float(np.sqrt(np.mean((model - v) ** 2)) * 100.0)
            except Exception as exc:  # one bad expiry must not stop the rest
                out["chain_fits"]["skipped"] += 1
                out.setdefault("fit_errors", []).append(clip(f"{ticker} {expiry}: {type(exc).__name__}: {exc}", 100))
                continue
            sabr.setdefault(ticker, {})[expiry] = {"days": int(days), "forward": f, "n": int(len(k)), "alpha": alpha, "rho": rho,
                                                   "nu": nu, "rmse_vol_pts": rmse}
            out["chain_fits"]["fitted"] += 1
    out["sabr"] = sabr
    return out
