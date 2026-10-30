"""google/tf-quant-finance: black_scholes.option_price on TensorFlow tensors (legacy job, Python 3.10).

The library is archived and needs TensorFlow; when TensorFlow cannot be imported the adapter reports failed with the
import error. Vectorised: all 12 contracts are priced in one call.
"""
from __future__ import annotations

import numpy as np

from ..context import LabContext
from ._common import price_test_set, version_of

TREE_STEPS = 400


def run(ctx: LabContext) -> dict:
    import tensorflow as tf  # noqa: F401 - fail early with a clear message when TensorFlow is missing
    import tf_quant_finance as tff

    contracts = ctx.contracts
    f64 = np.float64
    price_all = tff.black_scholes.option_price(
        volatilities=np.array([c.sigma for c in contracts], f64),
        strikes=np.array([c.K for c in contracts], f64),
        expiries=np.array([c.T for c in contracts], f64),
        spots=np.array([c.S for c in contracts], f64),
        discount_rates=np.array([c.r for c in contracts], f64),
        dividend_rates=np.array([c.q for c in contracts], f64),
        is_call_options=np.array([c.cp == "c" for c in contracts]),
        dtype=tf.float64)
    values = {c.id: float(v) for c, v in zip(contracts, np.asarray(price_all).ravel())}
    american_values: dict[str, float] = {}
    american_error = ""
    try:
        tree = tff.black_scholes.option_price_binomial(
            volatilities=np.array([c.sigma for c in contracts], f64), strikes=np.array([c.K for c in contracts], f64),
            expiries=np.array([c.T for c in contracts], f64), spots=np.array([c.S for c in contracts], f64),
            discount_rates=np.array([c.r for c in contracts], f64), dividend_rates=np.array([c.q for c in contracts], f64),
            is_call_options=np.array([c.cp == "c" for c in contracts]), is_american=np.array([True] * len(contracts)),
            num_steps=TREE_STEPS, dtype=tf.float64)
        american_values = {c.id: float(v) for c, v in zip(contracts, np.asarray(tree).ravel())}
    except Exception as exc:  # the European column is still useful without the tree
        american_error = f"{type(exc).__name__}: {exc}"[:150]
    out = price_test_set(ctx, european=lambda c: values[c.id],
                         american=(lambda c: american_values[c.id]) if american_values else None,
                         methods={"european": "tff.black_scholes.option_price",
                                  "american": f"tff.black_scholes.option_price_binomial, {TREE_STEPS} steps"})
    if not american_values:
        out["american_error"] = american_error
    out["version"] = version_of("tf-quant-finance", "tf_quant_finance")
    out["tensorflow"] = version_of("tensorflow", "tensorflow-cpu")
    return out
