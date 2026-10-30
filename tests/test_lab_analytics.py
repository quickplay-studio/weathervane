"""Lab analytics that need no third-party library: vol surface, cross-check framework, sleeve weights, COT statistics,
camofox rules, report rendering and the video template."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from weathervane.lab import context, crosscheck, refprice, report, sleeves, volsurf
from weathervane.lab.adapters import openbb_odp
from weathervane.lab.context import LabContext
from weathervane.lab.fetch import camofox
from weathervane.lab.util import today_utc
from weathervane.lab.video import render


def lab_ctx(tmp_path, **overrides) -> LabContext:
    cfg = context.load_config(overrides or None)
    return LabContext(cfg=cfg, today=today_utc(), out=tmp_path, contracts=context.build_test_set(cfg))


# --- volatility ----------------------------------------------------------------------------------------------------


def synthetic_iv(tmp_path, workspace):
    cfg = context.load_config()
    df, _ = context.synthetic_chains(cfg, today_utc(), 0.04)
    return volsurf.compute_chain_iv(df)


def test_implied_vol_round_trip():
    p = refprice.bs_price("c", 100, 105, 0.25, 0.04, 0.01, 0.27)
    assert refprice.implied_vol("c", p, 100, 105, 0.25, 0.04, 0.01) == pytest.approx(0.27, abs=1e-6)


def test_dashboard_on_synthetic_chains(tmp_path, workspace):
    civ = synthetic_iv(tmp_path, workspace)
    rows = volsurf.dashboard(civ, {}, "2026-01-02", history=pd.DataFrame(columns=["date", "ticker", "atm_iv_30d"]))
    assert set(rows) == set(civ["ticker"].unique())
    spy = rows["SPY"]
    assert 0.05 < spy["atm_iv_30d"] < 0.6
    assert spy["rr25_30d"] is not None and spy["rr25_30d"] < 0           # synthetic smile has a put skew
    assert spy["implied_move_1w_pct"] == pytest.approx(100 * spy["atm_iv_30d"] * math.sqrt(7 / 365), rel=1e-6)
    assert spy["iv_rank"] is None and spy["history_days"] == 0           # fewer than 20 observations


def test_rank_and_percentile_need_history():
    hist = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30).strftime("%Y-%m-%d"), "ticker": "SPY",
                         "atm_iv_30d": np.linspace(0.10, 0.40, 30)})
    out = volsurf.rank_percentile(hist, "SPY", 0.25)
    assert out["iv_rank"] == pytest.approx(50.0, abs=1.0) and 40 < out["iv_percentile"] < 60
    assert volsurf.rank_percentile(hist.head(5), "SPY", 0.25)["iv_rank"] is None


def test_history_upsert_is_idempotent(workspace):
    volsurf.update_history("2026-03-02", {"SPY": 0.2})
    h = volsurf.update_history("2026-03-02", {"SPY": 0.25})
    assert len(h) == 1 and h["atm_iv_30d"].iloc[0] == 0.25


def test_interp_tenor_total_variance():
    term = pd.DataFrame({"days": [20, 60], "atm_iv": [0.20, 0.30]})
    v = volsurf.interp_tenor(term, 30)
    assert 0.20 < v < 0.30
    assert volsurf.interp_tenor(term, 5) == 0.20 and volsurf.interp_tenor(term, 90) == 0.30


# --- cross-check framework -----------------------------------------------------------------------------------------


def test_compare_aligns_tail_and_ignores_nan():
    core = np.array([np.nan, 1.0, 2.0, 3.0])
    other = np.array([9.0, 1.0, 2.0, 3.0 + 1e-9, 9.9])[:4]
    r = crosscheck.compare(core, other)
    assert r["n"] == 3 and r["max_abs"] == pytest.approx(1e-9)


def test_evaluate_flags_only_matching_definitions_and_survives_errors():
    a = np.arange(1.0, 50.0)
    checks = [crosscheck.Check("same", a, a.copy()),
              crosscheck.Check("off", a, a * 1.001),
              crosscheck.Check("diff_def", a, a * 1.5, definitions_match=False),
              crosscheck.Check("boom", a, lambda: (_ for _ in ()).throw(RuntimeError("nope")))]
    res = crosscheck.evaluate(checks, 1e-6)
    assert not res["same"]["flagged"] and res["off"]["flagged"] and not res["diff_def"]["flagged"]
    assert "RuntimeError" in res["boom"]["error"]


def test_core_sample_std_matches_pandas():
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(size=300))
    df = pd.DataFrame({"close": close, "high": close + 1, "low": close - 1}, index=pd.date_range("2024-01-01", periods=300))
    f = crosscheck.frame(df, 300)
    ours = crosscheck.core_sample_std(f, 20)
    ref = pd.Series(close).rolling(20).std().to_numpy()
    assert crosscheck.compare(ours, ref)["max_abs"] < 1e-9


# --- sleeve weights ------------------------------------------------------------------------------------------------


def test_sleeve_weights_are_valid_allocations():
    rng = np.random.default_rng(3)
    streams = pd.DataFrame({"a": rng.normal(0.05, 1.0, 500), "b": rng.normal(0.02, 0.5, 500), "c": rng.normal(0.03, 2.0, 500)})
    out = sleeves.suggest_weights(streams)
    for key in ("equal", "inverse_vol", "mvo_long_only", "min_variance"):
        w = out[key]
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6), key
    assert out["inverse_vol"]["b"] > out["inverse_vol"]["c"]             # calmer stream gets more weight
    assert all(v >= -1e-9 for v in out["mvo_long_only"].values())


# --- COT -----------------------------------------------------------------------------------------------------------


def test_positioning_net_and_percentile():
    dates = pd.date_range("2023-01-03", periods=200, freq="7D")
    df = pd.DataFrame({"date": dates, "managed_money_positions_long_all": np.arange(200) + 1000.0,
                       "managed_money_positions_short_all": np.full(200, 500.0), "open_interest_all": np.full(200, 10000.0)})
    out = openbb_odp.positioning(df, "disaggregated", 3)
    assert out["net"] == pytest.approx(199 + 500) and out["percentile"] > 95 and out["change_1w"] == 1.0


def test_percentile_of_last():
    assert openbb_odp.percentile_of_last(np.array([1.0, 2.0, 3.0, 4.0, 4.0])) == pytest.approx(80.0)


# --- camofox -------------------------------------------------------------------------------------------------------


def test_camofox_blocklist_always_wins():
    for url in ("https://ftmo.com/x", "https://www.topstep.com/", "https://api.oanda.com/v3", "https://app.alpaca.markets/"):
        assert camofox.is_blocked(url)
        assert not camofox.is_allowed(url, ["ftmo.com", "topstep.com", "oanda.com", "alpaca.markets"])


def test_camofox_empty_allowlist_allows_nothing():
    assert not camofox.is_allowed("https://example.com/news", [])
    assert camofox.is_allowed("https://news.example.com/a", ["example.com"])
    assert not camofox.is_allowed("https://notexample.com/a", ["example.com"])
    assert not camofox.is_allowed("ftp://example.com/a", ["example.com"])


def test_camofox_rate_limiter_spaces_requests_per_domain():
    now = [0.0]
    slept = []
    lim = camofox.RateLimiter(5.0, clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    assert lim.wait("a.com") == 0.0
    now[0] += 2.0
    assert lim.wait("a.com") == pytest.approx(3.0)
    assert lim.wait("b.com") == 0.0                                      # other domains are independent
    assert slept == [pytest.approx(3.0)]


def test_camofox_guard_reasons():
    g = camofox.Guard(allowlist=["example.com"], limiter=camofox.RateLimiter(0, sleep=lambda s: None))
    assert g.check("https://ftmo.com/a") == (False, "blocked domain")
    assert g.check("https://other.org/a") == (False, "not on the allowlist")
    assert g.check("https://example.com/a") == (True, "")


def test_camofox_skips_when_disabled(tmp_path):
    from weathervane.lab import registry
    rec = registry.run_entry(registry.BY_KEY["camofox"], lab_ctx(tmp_path))
    assert rec["status"] == "skipped"


# --- report and video ----------------------------------------------------------------------------------------------


def test_report_renders_with_no_results(tmp_path, workspace):
    ctx = lab_ctx(tmp_path)
    md, ctxjson = report.build(ctx, groups={}, write_history=False)
    assert md.startswith("# Weathervane lab report") and "## 1. Library status" in md
    assert md.count("\n| ") >= 22 and "not run" in md
    assert ctxjson["volatility"] == {} and ctxjson["positioning"] == {}
    json.dumps(ctxjson)


def test_report_pricing_flags_a_bad_library(tmp_path, workspace):
    ctx = lab_ctx(tmp_path)
    ref = {c.id: refprice.bs_price(c.cp, c.S, c.K, c.T, c.r, c.q, c.sigma) for c in ctx.contracts}
    bad = dict(ref)
    first = ctx.contracts[0].id
    bad[first] = ref[first] * 1.5
    groups = {"modern": {"adapters": {
        "vollib": {"status": "ok", "reason": "", "seconds": 1, "results": {"prices": {"european": ref}}},
        "qfin": {"status": "ok", "reason": "", "seconds": 1, "results": {"prices": {"european": bad}, "known_issue": "demo"}}}}}
    tables = report.pricing_tables(ctx, groups)
    assert tables["deviations"]["vollib"]["european"]["flagged"] == []
    assert tables["deviations"]["qfin"]["european"]["flagged"] == [first]
    md, _ = report.build(ctx, groups, write_history=False)
    assert "demo" in md and f"{first}" in md


def test_report_write_creates_files(tmp_path, workspace):
    ctx = lab_ctx(tmp_path)
    info = report.write(ctx, groups={})
    assert (workspace / "reports" / "lab" / f"{ctx.today.isoformat()}.md").exists()
    assert json.loads((workspace / "results" / "context.json").read_text())["asof"] == ctx.today.isoformat()
    assert info["bytes"] > 500


def test_video_template_fill_is_complete_and_safe():
    cfg = context.load_config()["video"]
    data = {"portfolios": [], "signals": [], "note": "</script><b>{{WIDTH}}</b>"}
    html = render.fill_template(data, cfg)
    assert 'data-composition-id="main"' in html and f'data-duration="{cfg["seconds"]:g}"' in html
    assert "</script><b>" not in html          # data cannot close the script tag
    import re
    left = [t for t in re.findall(r"\{\{[A-Z_]+\}\}", html) if t != "{{WIDTH}}"]
    assert not left, left
    assert "window.__timelines" in html and "paused: true" in html
