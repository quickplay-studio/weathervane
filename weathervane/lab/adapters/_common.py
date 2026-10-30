"""Helpers the pricing adapters share: price the test set, build uniform result blocks."""
from __future__ import annotations

import math
from importlib import metadata
from typing import Callable

from ..context import Contract, LabContext
from ..util import clip

PriceFn = Callable[[Contract], float]


def price_test_set(ctx: LabContext, european: PriceFn | None = None, american: PriceFn | None = None,
                   methods: dict[str, str] | None = None, stochastic: bool = False) -> dict:
    """Price every test contract with the given functions. A failing contract is recorded, not raised.

    Returns {"prices": {"european": {id: p}, "american": {id: p}}, "methods": {...}, "errors": {...}, "stochastic": bool}.
    """
    out: dict = {"prices": {}, "methods": methods or {}, "errors": {}, "stochastic": stochastic}
    for style, fn in (("european", european), ("american", american)):
        if fn is None:
            continue
        prices: dict[str, float | None] = {}
        for c in ctx.contracts:
            try:
                p = float(fn(c))
                prices[c.id] = p if math.isfinite(p) else None
            except Exception as exc:  # one bad contract must not hide the others
                prices[c.id] = None
                out["errors"][f"{style}:{c.id}"] = clip(f"{type(exc).__name__}: {exc}", 120)
        out["prices"][style] = prices
    ok = [p for style in out["prices"].values() for p in style.values() if p is not None]
    if not ok:
        raise RuntimeError("no contract could be priced: " + "; ".join(list(out["errors"].values())[:2]))
    return out


def max_gap(a: dict, b: dict) -> float:
    """Largest absolute difference between two {id: price} dicts over their common ids."""
    gaps = [abs(a[k] - b[k]) for k in a if a.get(k) is not None and b.get(k) is not None]
    return max(gaps) if gaps else float("nan")


def version_of(*dists: str) -> str:
    """Installed version of the first distribution that exists, else 'unknown'."""
    for name in dists:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "unknown"
