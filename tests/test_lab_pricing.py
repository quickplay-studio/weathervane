"""Unit test of each pricing library against the reference Black-Scholes price on one contract, tolerance 1e-2.

Each test skips when the library is not installed (the lab libraries are optional and live in requirements-lab-*.txt).
The adapters run through the same guard the workflow uses, so these tests also cover the adapter wiring.
"""
import importlib
import importlib.util
from pathlib import Path

import pytest

from weathervane.lab import context, refprice, registry
from weathervane.lab.context import LabContext
from weathervane.lab.util import today_utc

TOL = 1e-2
REPO = Path(__file__).resolve().parents[1]


def make_ctx(tmp_path, **test_set) -> LabContext:
    cfg = context.load_config({"test_set": test_set} if test_set else None)
    return LabContext(cfg=cfg, today=today_utc(), out=tmp_path, rate=cfg["test_set"]["r"], contracts=context.build_test_set(cfg))


def need(module: str):
    try:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not installed")
    except (ImportError, ValueError):
        pytest.skip(f"{module} is not importable")


def run_adapter(key: str, ctx: LabContext) -> dict:
    rec = registry.run_entry(registry.BY_KEY[key], ctx)
    if rec["status"] == "failed" and rec["reason"].startswith("import failed"):
        pytest.skip(rec["reason"])
    assert rec["status"] == "ok", rec["reason"]
    return rec["results"]


def check_european(ctx, results, tol=TOL, ids=None):
    prices = results["prices"]["european"]
    assert prices, "no european prices"
    worst = 0.0
    for c in ctx.contracts:
        if ids and c.id not in ids:
            continue
        ref = refprice.bs_price(c.cp, c.S, c.K, c.T, c.r, c.q, c.sigma)
        worst = max(worst, abs(prices[c.id] - ref))
    assert worst < tol, f"max abs deviation {worst}"


def test_reference_price_matches_known_value():
    # S=K=100, T=1, r=5%, q=0, sigma=20%: textbook call 10.4506
    assert refprice.bs_price("c", 100, 100, 1.0, 0.05, 0.0, 0.2) == pytest.approx(10.4506, abs=1e-3)


@pytest.mark.parametrize("key,module", [("vollib", "vollib"), ("pysabr", "pysabr"), ("quantsbin", "quantsbin"),
                                         ("finoptions", "finoptions")])
def test_european_price_matches_reference(tmp_path, key, module):
    need(module)
    ctx = make_ctx(tmp_path)
    check_european(ctx, run_adapter(key, ctx))


def test_qfin_matches_reference_at_zero_rate(tmp_path):
    """qfin's call is exact at r = 0, q = 0 and its put is exact at the money; its put is wrong away from the money
    and its calls ignore r (library defects, reported by the lab), so only the cases that should hold are asserted."""
    need("qfin")
    ctx = make_ctx(tmp_path, r=0.0, q=0.0)
    ids = {c.id for c in ctx.contracts if c.cp == "c" or c.moneyness == 1.0}
    check_european(ctx, run_adapter("qfin", ctx), ids=ids)


@pytest.mark.xfail(reason="qfin closed-form Black-Scholes ignores the interest rate and misprices puts away from the money",
                   strict=False)
def test_qfin_matches_reference_with_a_rate(tmp_path):
    need("qfin")
    ctx = make_ctx(tmp_path)
    check_european(ctx, run_adapter("qfin", ctx))


def test_optlib_matches_reference(tmp_path):
    need("optlib")
    ctx = make_ctx(tmp_path)
    check_european(ctx, run_adapter("optlib", ctx))


def test_willowtree_matches_reference(tmp_path):
    src = REPO / "data" / "lab" / "src" / "willowtree"
    if not src.is_dir():
        pytest.skip("willowtree clone not staged (data/lab/src/willowtree)")
    ctx = make_ctx(tmp_path)
    check_european(ctx, run_adapter("willowtree", ctx))


def test_american_prices_are_not_below_european(tmp_path):
    need("finoptions")
    ctx = make_ctx(tmp_path)
    res = run_adapter("finoptions", ctx)
    am, eu = res["prices"]["american"], res["prices"]["european"]
    for c in ctx.contracts:
        assert am[c.id] >= eu[c.id] - 5e-3
