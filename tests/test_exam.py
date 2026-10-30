"""Exam Monte Carlo against deterministic series where the answer is known."""
import numpy as np
import pandas as pd
import pytest

from weathervane import exam
from weathervane.config import Phase, Rules


def rules(**kw) -> Rules:
    base = dict(firm="T", program="T", account_size=100000.0,
                phases=(Phase("Challenge", 0.10, 4), Phase("Verification", 0.05, 4)),
                daily_loss=0.05, max_loss=0.10)
    base.update(kw)
    return Rules(**base)


def const(rc, rw=None, n=60, active=1):
    rw = rc if rw is None else rw
    return pd.DataFrame({"r_close": [rc] * n, "r_worst": [rw] * n, "active": [active] * n})


def test_constant_gain_passes_phase_one_in_ten_days_and_phase_two_in_five():
    res = exam.run_exam(const(1.0), rules(), [0.01], n_sims=50, seed=1)[0]
    assert res.pass_p1 == 1.0 and res.pass_p2 == 1.0 and res.pass_combined == 1.0
    assert res.days_median == 15 and res.days_p90 == 15


def test_min_trading_days_delays_the_pass():
    res = exam.run_exam(const(1.0), rules(phases=(Phase("A", 0.02, 6),)), [0.01], n_sims=20, seed=1)[0]
    assert res.days_median == 6        # 2% is reached on day 2 but six trading days are required


def test_inactive_days_do_not_count_as_trading_days():
    d = const(1.0, active=0)
    d.loc[::2, "active"] = 1           # every other day is a trading day
    res = exam.run_exam(d, rules(phases=(Phase("A", 0.02, 4),)), [0.01], n_sims=40, seed=1)[0]
    assert res.pass_combined == 1.0 and res.days_median >= 7


def test_six_percent_day_fails_the_daily_limit():
    res = exam.run_exam(const(-6.0), rules(), [0.01], n_sims=30, seed=2)[0]
    assert res.pass_combined == 0.0
    assert res.fail_reasons["daily"] == 1.0


def test_intraday_worst_breaches_even_if_the_day_closes_flat():
    d = const(0.0, rw=-6.0)
    d["r_close"] = 0.001                # closes up a hair, but the worst point was -6%
    res = exam.run_exam(d, rules(), [0.01], n_sims=30, seed=2)[0]
    assert res.fail_reasons["daily"] == 1.0


def test_static_floor_fails_on_cumulative_loss():
    res = exam.run_exam(const(-3.0), rules(), [0.01], n_sims=30, seed=3)[0]    # -3% a day: floor 90% hit on day 4
    assert res.fail_reasons["max_loss"] == 1.0 and res.fail_reasons["daily"] == 0.0


def test_no_activity_never_passes():
    res = exam.run_exam(const(0.0), rules(), [0.01], n_sims=10)[0]
    assert res.pass_combined == 0.0


def _step_days(r, days):
    st = exam.State.fresh(1)
    out = []
    for x, w in days:
        fail, win = exam.step(st, np.array([x]), np.array([w]), np.array([1]), r, 10.0, 99)
        out.append(int(fail[0]))
    return out


def test_static_vs_trailing_floors():
    # up 6%, then a 10.5% loss: equity 0.955. The static floor is 0.90, the eod trailing floor 1.06 - 0.10 = 0.96.
    days = [(0.06, 0.06), (-0.105, -0.105)]
    assert _step_days(rules(max_loss_type="static", daily_loss=0.5), days) == [0, 0]
    assert _step_days(rules(max_loss_type="eod_trailing", daily_loss=0.5), days) == [0, 2]


def test_trailing_lock_stops_the_floor_at_the_initial_balance():
    days = [(0.12, 0.12), (-0.105, -0.105)]       # high-water 1.12; unlocked floor 1.02, locked floor 1.00; equity 1.015
    assert _step_days(rules(max_loss_type="eod_trailing", daily_loss=0.5), days) == [0, 2]
    assert _step_days(rules(max_loss_type="eod_trailing", lock_at_start=True, daily_loss=0.5), days) == [0, 0]
    low = [(0.12, 0.12), (-0.13, -0.13)]          # equity 0.99: below the locked floor
    assert _step_days(rules(max_loss_type="eod_trailing", lock_at_start=True, daily_loss=0.5), low) == [0, 2]


def test_intraday_trailing_uses_the_day_peak():
    # day 1 closes +1% but its proxy peak is equity + max(0, x) = 1.01; floor 0.91; worst -9.5% -> breach
    st = exam.State.fresh(1)
    r = rules(max_loss_type="intraday_trailing", daily_loss=0.5)
    fail, _ = exam.step(st, np.array([0.10]), np.array([-0.0905]), np.array([1]), r, 10.0, 99)
    assert fail[0] == 2


def test_consistency_rule_delays_the_pass():
    # one big +9% day then +1% days: the best day stays above 30% of profit until profit reaches 30%
    st = exam.State.fresh(1)
    r = rules(consistency=0.3)
    results = []
    for x in [0.09] + [0.01] * 40:
        _, win = exam.step(st, np.array([x]), np.array([x]), np.array([1]), r, 0.10, 1)
        results.append(bool(win[0]))
    assert not results[1] and any(results)
    assert results.index(True) > 5


def test_time_limit_fails_with_time_reason():
    res = exam.run_exam(const(0.1, n=200), rules(time_limit_days=14), [0.01], n_sims=20, seed=1)[0]
    assert res.fail_reasons["time"] == 1.0 and res.pass_combined == 0.0


def test_optimise_risk_prefers_lower_risk_on_ties_and_avoids_ruin():
    best, table = exam.optimise_risk(const(1.0), rules(), n_sims=40, seed=1)
    assert all(t.pass_combined == 1.0 for t in table)
    assert best.risk == 0.0025
    noisy = pd.DataFrame({"r_close": np.r_[np.full(30, 1.0), np.full(30, -2.0)],
                          "r_worst": np.r_[np.full(30, 0.0), np.full(30, -2.5)], "active": 1})
    best, table = exam.optimise_risk(noisy, rules(), n_sims=300, seed=1)
    assert best.risk < max(t.risk for t in table)


def test_risk_grid_is_capped_by_rules_max_risk():
    assert exam.allowed_grid(rules(max_risk=0.005), exam.DEFAULT_GRID) == [0.0025, 0.005]


def test_bootstrap_blocks_have_the_expected_mean_length():
    rng = np.random.default_rng(0)
    idx = exam.stationary_bootstrap(1000, 200, 400, 10, rng)
    runs = (np.diff(idx, axis=1) != 1) & (np.diff(idx, axis=1) != -999)
    assert 6 < 1 / runs.mean() < 16
    assert idx.min() >= 0 and idx.max() < 1000


def test_results_are_deterministic_for_a_seed():
    d = pd.DataFrame({"r_close": np.random.default_rng(0).normal(0.05, 1, 500)})
    d["r_worst"] = d["r_close"] - 0.3
    d["active"] = 1
    a = exam.run_exam(d, rules(), [0.005, 0.01], n_sims=100, seed=5)
    b = exam.run_exam(d, rules(), [0.005, 0.01], n_sims=100, seed=5)
    assert [x.pass_combined for x in a] == [x.pass_combined for x in b]


def test_tracker_restarts_after_a_breach_and_advances_phases():
    t = exam.ExamTracker(rules(phases=(Phase("A", 0.02, 1), Phase("B", 0.02, 1))))
    assert t.day("2025-01-01", 0.025, 0.025, 1)["event"] == "passed_phase_1"
    assert t.snapshot()["phase"] == 2
    assert t.day("2025-01-02", -0.06, -0.06, 1)["event"] == "breach_daily"
    snap = t.snapshot()
    assert snap["phase"] == 1 and snap["attempt"] == 2 and snap["equity"] == 1.0
