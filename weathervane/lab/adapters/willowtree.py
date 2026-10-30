"""federicomariamassari/willowtree: Curran-style willow tree (sampling of the normal density plus an LP Markov chain).

The package is not on PyPI under this name (PyPI's `willowtree` is unrelated), so the job clones the repository at a
pinned commit into data/lab/src/willowtree (config/lab.yaml: sources.willowtree). That code was written for NumPy 1.x
and SciPy's old simplex solver, so for the duration of the call `np.int`, `np.float`, `np.bool` and `np.object` are
restored and linprog is routed to HiGHS; both are undone afterwards.

Pricing: node i at step j has log spot (r - q - s^2/2) T t_j + s sqrt(T t_j) z_i, probabilities start at q and move with
the LP transition matrices. European = discounted expectation at the last step; American = backward induction.
"""
from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path

import numpy as np

from ... import config
from ..context import LabContext
from ._common import price_test_set

NODES = 20
STEPS = 12
GAMMA = 0.1
ALGORITHM = "fpm"      # first partial moment: the library recommends it near the money (kurtosis matching is 10-100x worse here)


@contextlib.contextmanager
def _compat():
    """Temporary NumPy/SciPy shims for the old willowtree code; restored on exit."""
    import scipy.optimize as so
    added = []
    for name, value in (("int", int), ("float", float), ("bool", bool), ("object", object)):
        if name not in np.__dict__:
            setattr(np, name, value)
            added.append(name)
    original = so.linprog

    def linprog(c, *args, method="highs", options=None, **kwargs):
        return original(c, *args, method="highs", **kwargs)

    so.linprog = linprog
    try:
        yield
    finally:
        so.linprog = original
        for name in added:
            try:
                delattr(np, name)
            except AttributeError:
                pass


def _import_willowtree(ctx: LabContext):
    src = config.path(*Path(ctx.cfg["sources"]["willowtree"]["path"]).parts)
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import matplotlib
    matplotlib.use("Agg")
    import willowtree as wt
    if not (hasattr(wt, "sampling") and hasattr(wt, "lp")):
        raise ImportError("the installed `willowtree` is not federicomariamassari/willowtree (no sampling/lp); clone it to "
                          + str(ctx.cfg["sources"]["willowtree"]["path"]))
    return wt


def run(ctx: LabContext) -> dict:
    wt = _import_willowtree(ctx)
    with _compat():
        q, z, gamma = wt.sampling(NODES, GAMMA, ALGORITHM)
        P, t = wt.lp(z, q, STEPS, tol=1e-12)
    P = np.asarray(P)
    if P.ndim == 2:
        P = P[np.newaxis]
    t = np.asarray(t, float)
    steps = len(P) + 1                       # time nodes t_1 ... t_steps
    if steps < 2 or len(t) < steps + 1:
        raise RuntimeError(f"Markov chain too short ({len(P)} matrices)")
    q = np.asarray(q, float)
    z = np.asarray(z, float)

    def lattice(c):
        T_j = c.T * t[1: steps + 1]
        drift = (c.r - c.q - 0.5 * c.sigma ** 2) * T_j
        return c.S * np.exp(drift[:, None] + c.sigma * np.sqrt(T_j)[:, None] * z[None, :])   # (steps, nodes)

    def payoff(c, s):
        return np.maximum(s - c.K, 0.0) if c.cp == "c" else np.maximum(c.K - s, 0.0)

    def european(c) -> float:
        s = lattice(c)[-1]
        return float(math.exp(-c.r * c.T) * (q @ payoff(c, s)))

    def american(c) -> float:
        s = lattice(c)
        times = np.concatenate([[0.0], c.T * t[1: steps + 1]])
        value = payoff(c, s[-1])
        for j in range(steps - 2, -1, -1):          # from step j+2 back to step j+1 (0-based row j)
            cont = math.exp(-c.r * (times[j + 2] - times[j + 1])) * (P[j] @ value)
            value = np.maximum(cont, payoff(c, s[j]))
        cont0 = math.exp(-c.r * times[1]) * (q @ value)
        return float(max(cont0, float(payoff(c, np.array([c.S]))[0])))

    out = price_test_set(ctx, european=european, american=american,
                         methods={"european": f"willow tree, {NODES} nodes (expectation at maturity)",
                                  "american": f"willow tree, {NODES} nodes x {steps} steps, backward induction"})
    out["tree"] = {"nodes": NODES, "steps": steps, "gamma": float(gamma), "algorithm": ALGORITHM, "q_sum": float(q.sum()),
                   "moment2": float(q @ z ** 2), "moment4": float(q @ z ** 4), "matrices": int(len(P))}
    out["source_ref"] = ctx.cfg["sources"]["willowtree"].get("ref", "")
    return out
