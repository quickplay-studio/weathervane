"""enthought/pyql (Cython wrappers) with a fallback to the QuantLib SWIG wheel.

pyql is built against apt libquantlib0-dev in the legacy job. If that build fails or the wrapper does not behave, the
adapter prices with the `QuantLib` wheel instead and says so in results["backend"]. The QuantLib prices are also the
reference every other library is compared with in the agreement table.
"""
from __future__ import annotations

import datetime as dt

from ..context import Contract, LabContext
from ._common import price_test_set

EU_METHOD = "analytic European (Black-Scholes-Merton)"
AM_METHOD = "Leisen-Reimer binomial tree, 801 steps"


def _quantlib(ctx: LabContext) -> dict:
    import QuantLib as ql

    today = ctx.today
    qdate = ql.Date(today.day, today.month, today.year)
    ql.Settings.instance().evaluationDate = qdate
    dc, cal = ql.Actual365Fixed(), ql.NullCalendar()

    def build(c: Contract):
        spot = ql.QuoteHandle(ql.SimpleQuote(c.S))
        rf = ql.YieldTermStructureHandle(ql.FlatForward(qdate, c.r, dc))
        dv = ql.YieldTermStructureHandle(ql.FlatForward(qdate, c.q, dc))
        vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(qdate, cal, c.sigma, dc))
        proc = ql.BlackScholesMertonProcess(spot, dv, rf, vol)
        expiry = qdate + int(c.days)
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if c.cp == "c" else ql.Option.Put, c.K)
        return proc, expiry, payoff

    def european(c: Contract) -> float:
        proc, expiry, payoff = build(c)
        opt = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))
        opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
        return opt.NPV()

    def american(c: Contract) -> float:
        proc, expiry, payoff = build(c)
        opt = ql.VanillaOption(payoff, ql.AmericanExercise(qdate, expiry))
        opt.setPricingEngine(ql.BinomialVanillaEngine(proc, "lr", 801))
        return opt.NPV()

    out = price_test_set(ctx, european, american, {"european": EU_METHOD, "american": AM_METHOD})
    greeks = {}
    for c in ctx.contracts:
        proc, expiry, payoff = build(c)
        opt = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))
        opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
        greeks[c.id] = {"delta": opt.delta(), "gamma": opt.gamma(), "vega": opt.vega()}
    out.update(backend="QuantLib", version=ql.__version__, greeks=greeks)
    return out


def _pyql(ctx: LabContext) -> dict:
    """enthought/pyql API (module name `quantlib`); only a real pyql has quantlib.instruments."""
    import quantlib
    from quantlib.exercise import AmericanExercise, EuropeanExercise
    from quantlib.handle import HandleBlackVolTermStructure, HandleYieldTermStructure
    from quantlib.instruments.api import OptionType
    from quantlib.instruments.vanillaoption import VanillaOption
    from quantlib.payoffs import PlainVanillaPayoff
    from quantlib.pricingengines.vanilla.fdblackscholesvanillaengine import FdBlackScholesVanillaEngine
    from quantlib.pricingengines.vanilla.vanilla import AnalyticEuropeanEngine
    from quantlib.processes.black_scholes_process import BlackScholesMertonProcess
    from quantlib.quotes import SimpleQuote
    from quantlib.settings import Settings
    from quantlib.termstructures.volatility.api import BlackConstantVol
    from quantlib.termstructures.yields.flat_forward import FlatForward
    from quantlib.time.api import Actual365Fixed, Date, NullCalendar

    today = ctx.today
    qdate = Date.from_datetime(dt.date(today.year, today.month, today.day))
    settings = Settings.instance() if hasattr(Settings, "instance") else Settings()
    settings.evaluation_date = qdate
    dc = Actual365Fixed()

    def build(c: Contract):
        proc = BlackScholesMertonProcess(
            SimpleQuote(c.S),
            HandleYieldTermStructure(FlatForward(reference_date=qdate, forward=c.q, daycounter=dc)),
            HandleYieldTermStructure(FlatForward(reference_date=qdate, forward=c.r, daycounter=dc)),
            HandleBlackVolTermStructure(BlackConstantVol(qdate, NullCalendar(), c.sigma, dc)))
        payoff = PlainVanillaPayoff(OptionType.Call if c.cp == "c" else OptionType.Put, c.K)
        return proc, qdate + int(c.days), payoff

    def european(c: Contract) -> float:
        proc, expiry, payoff = build(c)
        opt = VanillaOption(payoff, EuropeanExercise(expiry))
        opt.set_pricing_engine(AnalyticEuropeanEngine(proc))
        return opt.net_present_value

    def american(c: Contract) -> float:
        proc, expiry, payoff = build(c)
        opt = VanillaOption(payoff, AmericanExercise(qdate, expiry))
        opt.set_pricing_engine(FdBlackScholesVanillaEngine(proc, 801, 800))
        return opt.net_present_value

    out = price_test_set(ctx, european, american, {"european": EU_METHOD, "american": "finite differences 801x800 (pyql)"})
    out.update(backend="pyql", version=getattr(quantlib, "__version__", "unknown"))
    return out


def run(ctx: LabContext) -> dict:
    pyql_error = None
    try:
        import quantlib.instruments  # noqa: F401 - the enthought wrapper has this subpackage, other `quantlib` modules do not
        res = _pyql(ctx)
        res["quantlib_swig_available"] = _has("QuantLib")
        return res
    except Exception as exc:  # not installed, build failed, or the wrapper API differs: fall back
        pyql_error = f"{type(exc).__name__}: {exc}"[:200]
    res = _quantlib(ctx)
    res["pyql_error"] = pyql_error
    res["_reason"] = f"pyql unavailable, used the QuantLib wheel ({pyql_error})"[:200]
    return res


def _has(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None
