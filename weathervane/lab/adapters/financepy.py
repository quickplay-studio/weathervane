"""domokane/FinancePy: EquityVanillaOption (Black-Scholes) and EquityAmericanOption (CRR tree).

FinancePy measures time to expiry as calendar days over its own year length, so the European column can differ from the
Act/365 reference by a few basis points; the table shows the library's actual output.
"""
from __future__ import annotations

import io
import contextlib

from ..context import LabContext
from ._common import price_test_set, version_of

TREE_STEPS_PER_YEAR = 1500


def run(ctx: LabContext) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):   # the package prints a banner on import
        from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
        from financepy.models.black_scholes import BlackScholes
        from financepy.products.equity.equity_american_option import EquityAmericanOption
        from financepy.products.equity.equity_vanilla_option import EquityVanillaOption
        from financepy.utils.date import Date
        from financepy.utils.global_types import OptionTypes

    today = ctx.today
    valuation = Date(today.day, today.month, today.year)

    def curves(c):
        return DiscountCurveFlat(valuation, c.r), DiscountCurveFlat(valuation, c.q)

    def european(c) -> float:
        rate, div = curves(c)
        kind = OptionTypes.EUROPEAN_CALL if c.cp == "c" else OptionTypes.EUROPEAN_PUT
        opt = EquityVanillaOption(valuation.add_days(c.days), c.K, kind)
        return float(opt.value(valuation, c.S, rate, div, BlackScholes(c.sigma)))

    def american(c) -> float:
        from financepy.utils.global_types import BlackScholesTypes
        rate, div = curves(c)
        kind = OptionTypes.AMERICAN_CALL if c.cp == "c" else OptionTypes.AMERICAN_PUT
        opt = EquityAmericanOption(valuation.add_days(c.days), c.K, kind)
        model = BlackScholes(c.sigma, BlackScholesTypes.CRR_TREE, num_steps_per_year=TREE_STEPS_PER_YEAR)
        return float(opt.value(valuation, c.S, rate, div, model))

    out = price_test_set(ctx, european=european, american=american,
                         methods={"european": "Black-Scholes closed form", "american": f"CRR tree, {TREE_STEPS_PER_YEAR} steps per year"})
    out["version"] = version_of("financepy")
    return out
