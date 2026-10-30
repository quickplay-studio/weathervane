"""jsmidt/QuantPy: mean-variance and inverse-volatility weights for the champion sleeves (suggestion only).

The upstream package imports `pandas.io.data`, which pandas removed years ago, so it normally cannot be imported on a
current stack; a stub for that one module is tried before giving up. Whether or not the import works, the weights come
from the formulas of its Portfolio.get_w (reimplemented in lab/sleeves.py, because the class downloads its prices from
Yahoo in __init__). The adapter reports `failed` with the import error and still returns the weights when the package
cannot be imported, so the report is complete and the status table stays honest.
"""
from __future__ import annotations

import sys
import types

from .. import sleeves
from ..context import LabContext
from ._common import version_of


def _import_quantpy() -> str:
    """Import quantpy, installing a stub for the removed pandas.io.data on failure. Returns a short status text."""
    try:
        import quantpy  # noqa: F401
        return "imported"
    except ImportError as first:
        if isinstance(first, ModuleNotFoundError) and getattr(first, "name", "") == "quantpy":
            raise                                   # not installed at all: a stub for one of its imports cannot help
        stub = types.ModuleType("pandas.io.data")

        def _offline(*args, **kwargs):
            raise RuntimeError("DataReader is not available (pandas.io.data was removed)")

        stub.DataReader = _offline
        sys.modules["pandas.io.data"] = stub
        try:
            import pandas.io as pio
            pio.data = stub
            import quantpy  # noqa: F401
            return "imported with a pandas.io.data stub"
        except Exception as second:
            sys.modules.pop("pandas.io.data", None)
            for name in [m for m in sys.modules if m == "quantpy" or m.startswith("quantpy.")]:
                sys.modules.pop(name, None)
            raise ImportError(f"{type(first).__name__}: {first}; with stub: {type(second).__name__}: {second}") from second


def run(ctx: LabContext) -> dict:
    lookback = int(ctx.cfg.get("sleeves", {}).get("lookback_days", 1260))
    out: dict = {"portfolios": sleeves.all_portfolios(lookback), "formula": "Portfolio.get_w('characteristic' / 'sharpe')"}
    try:
        out["import"] = _import_quantpy()
        out["version"] = version_of("quantpy", "QuantPy")
    except ImportError as exc:
        out["import"] = "failed"
        out["_status"] = "failed"
        out["_reason"] = f"import failed: {exc}; weights use the reimplemented formulas"
    return out
