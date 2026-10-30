"""quantsbin/Quantsbin (Python 3.12+): EqOption pricing engines and payoff plots.

Binomial pricing is capped at 100 steps by the library, so the American column carries a larger discretisation error
than the other libraries; the payoff plot is saved to out/lab/quantsbin_payoff.png.
"""
from __future__ import annotations

import datetime as dt

from ..context import LabContext
from ._common import price_test_set, version_of


def run(ctx: LabContext) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    from quantsbin.derivativepricing import EqOption, Plotting

    today = ctx.today
    fmt = "%Y%m%d"

    def engine(c, expiry_type: str, model: str):
        expiry = today + dt.timedelta(days=int(c.days))
        opt = EqOption(option_type="Call" if c.cp == "c" else "Put", expiry_type=expiry_type, strike=c.K,
                       expiry_date=expiry.strftime(fmt))
        kw = dict(model=model, spot0=c.S, rf_rate=c.r, yield_div=c.q, volatility=c.sigma, pricing_date=today.strftime(fmt))
        if model == "Binomial":
            kw["no_of_steps"] = 100
        return opt, opt.engine(**kw)

    def european(c) -> float:
        return engine(c, "European", "BSM")[1].valuation()

    def american(c) -> float:
        return engine(c, "American", "Binomial")[1].valuation()

    out = price_test_set(ctx, european=european, american=american,
                         methods={"european": "BSM", "american": "binomial tree, 100 steps (library maximum)"})
    out["version"] = version_of("Quantsbin", "quantsbin")
    try:
        c = next(x for x in ctx.contracts if x.cp == "c" and abs(x.moneyness - 1.0) < 1e-9)
        opt, _ = engine(c, "European", "BSM")
        fig = Plotting(opt, "payoff", x_axis_range=[0.7 * c.S, 1.3 * c.S]).line_plot()
        path = ctx.out / "quantsbin_payoff.png"
        fig.savefig(path)
        out["payoff_plot"] = "out/lab/quantsbin_payoff.png"
    except Exception as exc:
        out["payoff_plot_error"] = f"{type(exc).__name__}: {exc}"[:150]
    return out
