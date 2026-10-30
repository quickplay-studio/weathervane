"""federicomariamassari/financial-engineering: Monte Carlo with antithetic and control variates, plus Merton jumps.

The repository holds one importable module, python-modules/jump_diffusion.py (Merton jump-diffusion paths, prints and
plots, returns nothing). It is loaded from a pinned clone (config/lab.yaml: sources.financial_engineering) and run once
so the library's own Monte Carlo mean can be compared with its theoretical mean. The pricing cross-check itself is
reimplemented from the method the README cites (Glasserman, Monte Carlo Methods in Financial Engineering): antithetic
draws and the discounted terminal price as control variate. A second check prices a call under Merton jump diffusion
by simulation and compares it with Merton's (1976) series.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import norm, poisson

from ... import config
from .. import refprice
from ..context import LabContext
from ._common import price_test_set

PATHS = 200_000
SEED = 7
JUMP = {"mu": 0.03, "sigma": 0.20, "Lambda": 0.25, "a": -0.05, "b": 0.10}


def _mc_vanilla(c, n: int = PATHS, seed: int = SEED) -> tuple[float, float]:
    """European price and standard error: antithetic pairs, discounted terminal price as control variate."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n // 2)
    st = np.concatenate([c.S * np.exp((c.r - c.q - 0.5 * c.sigma ** 2) * c.T + c.sigma * math.sqrt(c.T) * s * z) for s in (1.0, -1.0)])
    disc = math.exp(-c.r * c.T)
    sign = 1.0 if c.cp == "c" else -1.0
    pay = disc * np.maximum(sign * (st - c.K), 0.0)
    cv = disc * st
    expect_cv = c.S * math.exp(-c.q * c.T)
    half = n // 2
    pay_pair = 0.5 * (pay[:half] + pay[half:])        # antithetic pair means are the i.i.d. units
    cv_pair = 0.5 * (cv[:half] + cv[half:])
    beta = np.cov(pay_pair, cv_pair)[0, 1] / np.var(cv_pair, ddof=1)
    adjusted = pay_pair - beta * (cv_pair - expect_cv)
    return float(adjusted.mean()), float(adjusted.std(ddof=1) / math.sqrt(half))


def merton_series(cp: str, S: float, K: float, T: float, r: float, q: float, sigma: float, lam: float, a: float, b: float,
                  terms: int = 60) -> float:
    """Merton (1976) jump-diffusion call or put: Poisson-weighted Black-Scholes prices."""
    k = math.exp(a + 0.5 * b * b) - 1.0
    lam2 = lam * (1.0 + k)
    total = 0.0
    for n in range(terms):
        sigma_n = math.sqrt(sigma ** 2 + n * b * b / T)
        r_n = r - lam * k + n * math.log1p(k) / T
        total += poisson.pmf(n, lam2 * T) * refprice.bs_price(cp, S, K, T, r_n, q, sigma_n)
    return float(total)


def _mc_merton(S: float, K: float, T: float, r: float, q: float, p: dict, n: int = PATHS, seed: int = SEED) -> tuple[float, float]:
    """Call price under Merton jumps with the library's jump parametrisation; drift compensated for risk neutrality."""
    rng = np.random.default_rng(seed)
    k = math.exp(p["a"] + 0.5 * p["b"] ** 2) - 1.0
    drift = r - q - p["Lambda"] * k
    half = n // 2
    z1, z2 = rng.standard_normal(half), rng.standard_normal(half)
    jumps = rng.poisson(p["Lambda"] * T, half)
    out = []
    for sign in (1.0, -1.0):
        x = (drift - 0.5 * p["sigma"] ** 2) * T + p["sigma"] * math.sqrt(T) * sign * z1 + p["a"] * jumps + p["b"] * np.sqrt(jumps) * sign * z2
        out.append(math.exp(-r * T) * np.maximum(S * np.exp(x) - K, 0.0))
    pair = 0.5 * (out[0] + out[1])
    return float(pair.mean()), float(pair.std(ddof=1) / math.sqrt(half))


def _library_run(path: Path) -> dict:
    """Execute the repository's jump_diffusion() once and compare its Monte Carlo mean with its theoretical mean."""
    import matplotlib
    matplotlib.use("Agg")
    spec = importlib.util.spec_from_file_location("fe_jump_diffusion", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        module.jump_diffusion(S=100.0, X=100.0, T=0.25, mu=JUMP["mu"], sigma=JUMP["sigma"], Lambda=JUMP["Lambda"], a=JUMP["a"],
                              b=JUMP["b"], Nsteps=63, Nsim=20000, alpha=0.05, seed=SEED)
    text = buf.getvalue()

    def grab(label: str) -> float | None:
        m = re.search(re.escape(label) + r"\s+(-?\d+(?:\.\d+)?)", text)
        return float(m.group(1)) if m else None

    mc_mean, theory = grab("Mean"), grab("Mean (M)")
    lo, hi = grab("Lower bound"), grab("Upper bound")
    return {"mc_mean": mc_mean, "theoretical_mean": theory, "ci_low": lo, "ci_high": hi,
            "theory_inside_ci": bool(lo is not None and hi is not None and theory is not None and lo <= theory <= hi)}


def run(ctx: LabContext) -> dict:
    stderr: dict[str, float] = {}

    def european(c) -> float:
        price, se = _mc_vanilla(c)
        stderr[c.id] = se
        return price

    out: dict = {"prices": {}, "methods": {}, "errors": {}, "stochastic": True}
    mc = price_test_set(ctx, european=european, methods={"european": f"Monte Carlo, {PATHS} paths, antithetic + control variate (reimplemented)"},
                        stochastic=True)
    out["prices"]["european:mc"] = mc["prices"]["european"]
    out["methods"]["european:mc"] = mc["methods"]["european"]
    out["errors"].update(mc["errors"])
    out["stochastic_styles"] = ["european:mc"]
    out["mc_stderr"] = stderr

    atm = next(c for c in ctx.contracts if c.cp == "c" and abs(c.moneyness - 1.0) < 1e-9 and c.days == max(x.days for x in ctx.contracts))
    series = merton_series("c", atm.S, atm.K, atm.T, atm.r, atm.q, JUMP["sigma"], JUMP["Lambda"], JUMP["a"], JUMP["b"])
    sim, se = _mc_merton(atm.S, atm.K, atm.T, atm.r, atm.q, JUMP)
    out["merton_check"] = {"contract": atm.id, "series": series, "monte_carlo": sim, "stderr": se,
                           "z_score": (sim - series) / se if se > 0 else None, "params": JUMP}

    src = config.path(*Path(ctx.cfg["sources"]["financial_engineering"]["path"]).parts) / "python-modules" / "jump_diffusion.py"
    out["source_ref"] = ctx.cfg["sources"]["financial_engineering"].get("ref", "")
    if not src.exists():
        out["_status"] = "failed"
        out["_reason"] = f"import failed: clone of financial-engineering not found at {ctx.cfg['sources']['financial_engineering']['path']}"
        return out
    out["library_run"] = _library_run(src)
    return out
