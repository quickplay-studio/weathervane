"""Simulator behaviour against hand-worked Pine broker emulator cases. ATR is a constant 2.0 in these cases."""
import numpy as np
import pytest
from helpers import bars_from, random_walk_df, signals_for, sleeve_with

from weathervane import simulator as sm
from weathervane.signals_core import Bars, build_signals


def run(o, h, l, c, long=(), short=(), exit_long=(), exit_short=(), half=0.0, start=0, stop_idx=None, **kw):
    bars = bars_from(o, h, l, c)
    sigs = signals_for(len(o), long, short, exit_long, exit_short)
    return sm.simulate(sleeve_with(**kw), bars, half, "X", sigs, start, stop_idx)


def flat_tail(o, h, l, c, extra=3, px=None):
    px = c[-1] if px is None else px
    return [*o, *[px] * extra], [*h, *[px] * extra], [*l, *[px] * extra], [*c, *[px] * extra]


# signal bar 0: close 100, atr 2, stop_atr 1.5 -> R0 = 3, stop 97, (target_atr 2 -> 104)
def long_case(bar1, extra_bars=(), **kw):
    o = [100, bar1[0], *[b[0] for b in extra_bars]]
    h = [100, bar1[1], *[b[1] for b in extra_bars]]
    l = [100, bar1[2], *[b[2] for b in extra_bars]]
    c = [100, bar1[3], *[b[3] for b in extra_bars]]
    o, h, l, c = flat_tail(o, h, l, c)
    return run(o, h, l, c, long=[0], **kw)


def test_stop_hit_on_fill_bar_costs_one_r():
    sim = long_case((100, 101, 96, 98))
    tr = sim.trades_frame().iloc[0]
    assert (tr.entry_price, tr.exit_price, tr.stop) == (100, 97, 97)
    assert tr.r == pytest.approx(-1.0) and tr.reason == "stop" and tr.bars == 0
    assert sim.r_close[1] == pytest.approx(-1.0) and sim.r_worst[1] == pytest.approx(-1.0)
    assert sim.active[1] == 1 and sim.active[0] == 0


def test_target_hit_when_high_comes_first():
    sim = long_case((100, 105, 99, 104), target_atr=2.0)
    tr = sim.trades_frame().iloc[0]
    assert tr.exit_price == 104 and tr.reason == "target"
    assert tr.r == pytest.approx(4 / 3)


def test_path_rule_picks_low_first_on_tie():
    # |o-h| = |o-l| = 4 -> not strictly less -> open, low, high, close: the stop is first
    sim = long_case((100, 104, 96, 100), target_atr=2.0)
    assert sim.trades_frame().iloc[0].reason == "stop"


def test_path_rule_picks_high_first_when_closer():
    # |o-h| = 4 < |o-l| = 4.5 -> open, high, low, close: the target is first
    sim = long_case((100, 104, 95.5, 100), target_atr=2.0)
    tr = sim.trades_frame().iloc[0]
    assert tr.reason == "target" and tr.r == pytest.approx(4 / 3)


def test_gap_through_stop_fills_at_open():
    sim = long_case((100, 101, 99.5, 100.5), extra_bars=[(95, 96, 94, 95.5)])
    tr = sim.trades_frame().iloc[0]
    assert tr.exit_price == 95 and tr.r == pytest.approx(-5 / 3) and tr.reason == "stop"


def test_gap_through_target_fills_at_open():
    sim = long_case((100, 101, 99.5, 100.5), extra_bars=[(106, 107, 105, 106)], target_atr=2.0)
    tr = sim.trades_frame().iloc[0]
    assert tr.exit_price == 106 and tr.r == pytest.approx(2.0) and tr.reason == "target"


def test_entry_gapping_beyond_the_stop_exits_at_the_same_open():
    sim = long_case((95, 96, 94, 95.5))
    tr = sim.trades_frame().iloc[0]
    assert tr.entry_price == 95 and tr.exit_price == 95 and tr.r == 0


def test_short_mirror_stop_and_target():
    o, h, l, c = flat_tail([100, 100], [100, 104], [100, 99], [100, 103])
    o, h, l, c = flat_tail([100, 100], [100, 101], [100, 95], [100, 96])
    sim = run(o, h, l, c, short=[0], target_atr=2.0)    # stop 103, target 96; |o-h|=1 < |o-l|=5 -> high first, then 96
    tr = sim.trades_frame().iloc[0]
    assert tr.side == "short" and tr.exit_price == 96 and tr.r == pytest.approx(4 / 3)
    o, h, l, c = flat_tail([100, 100], [100, 104], [100, 99], [100, 103])
    tr = run(o, h, l, c, short=[0], target_atr=2.0).trades_frame().iloc[0]    # |o-h|=4 > |o-l|=1 -> low first, no 96, then stop
    assert tr.exit_price == 103 and tr.reason == "stop" and tr.r == pytest.approx(-1.0)


def test_carried_position_marks_to_market_and_worst_uses_low():
    sim = long_case((100, 101, 99, 101), extra_bars=[(101, 103, 100, 102)])
    # bar 1: close 101 -> +1/3R ; worst low 99 -> -1/3R
    assert sim.r_close[1] == pytest.approx(1 / 3)
    assert sim.r_worst[1] == pytest.approx(-1 / 3)
    # bar 2: close 102 vs 101 -> +1/3R ; worst low 100 vs prev close 101 -> -1/3R
    assert sim.r_close[2] == pytest.approx(1 / 3)
    assert sim.r_worst[2] == pytest.approx(-1 / 3)
    assert sim.active[2] == 1


def test_open_trade_is_not_in_trades_but_in_state():
    sim = long_case((100, 101, 99, 101))
    assert sim.state[sm.S_POS] == 1
    o, h, l, c = [100, 100, 101], [100, 101, 102], [100, 99.5, 100.5], [100, 101, 101.5]
    sim = run(o, h, l, c, long=[0])
    assert sim.trades_frame().empty and sim.state[sm.S_POS] == 1 and sim.state[sm.S_STOP0] == 97


def test_costs_are_charged_on_entry_and_exit():
    sim = long_case((100, 101, 96, 98), half=0.001)
    tr = sim.trades_frame().iloc[0]
    assert tr.entry_price == pytest.approx(100.1)
    assert tr.exit_price == pytest.approx(97 * 0.999)
    assert tr.r == pytest.approx((97 * 0.999 - 100.1) / 3)


def test_reversal_closes_and_flips_at_next_open():
    o = [100, 100, 101, 100, 99, 99, 99]
    h = [100, 101, 102, 101, 100, 99, 99]
    l = [100, 99.5, 100.5, 98.5, 98.5, 99, 99]
    c = [100, 101, 101, 99.5, 99, 99, 99]
    sim = run(o, h, l, c, long=[0], short=[2])           # reverse on at close of bar 2 -> flips at open of bar 3
    tr = sim.trades_frame()
    assert tr.iloc[0].reason == "reversal" and tr.iloc[0].exit_date == tr.iloc[0].exit_date
    assert tr.iloc[0].exit_price == 100 and tr.iloc[0].bars == 2
    assert sim.state[sm.S_POS] == -1 and sim.state[sm.S_ENTRY_PX] == 100


def test_opposite_signal_ignored_without_reverse():
    o = [100, 100, 101, 100, 99, 99, 99]
    h = [100, 101, 102, 101, 100, 99, 99]
    l = [100, 99.5, 100.5, 98.5, 98.5, 99, 99]
    c = [100, 101, 101, 99.5, 99, 99, 99]
    sim = run(o, h, l, c, long=[0], short=[2], reverse=False)
    assert sim.state[sm.S_POS] == 1 and sim.trades_frame().empty


def test_same_direction_signal_does_not_pyramid():
    o = [100, 100, 100, 100, 100, 100]
    sim = run(o, [101] * 6, [99.5] * 6, [100] * 6, long=[0, 1, 2])
    assert sim.state[sm.S_ENTRY_I] == 1 and len(sim.trades_frame()) == 0


def test_time_stop_exits_at_next_open_after_n_bars():
    n = 9
    sim = run([100] * n, [100.5] * n, [99.5] * n, [100] * n, long=[0], time_stop_bars=3)
    tr = sim.trades_frame().iloc[0]
    # fill bar 1 (held 0); at close of bar 4 held == 3 -> exit at the open of bar 5
    assert tr.reason == "time_stop" and tr.bars == 4


def test_exit_signal_exits_at_next_open():
    n = 8
    sim = run([100] * n, [100.5] * n, [99.5] * n, [100, 100, 100, 100, 100, 100, 100, 100],
              long=[0], exit_long=[3])
    tr = sim.trades_frame().iloc[0]
    assert tr.reason == "exit_signal" and tr.bars == 3


def test_trailing_stop_moves_on_close_and_applies_next_bar():
    # entry 100, R0 3, trail 1 ATR (2.0). Bar1 high 104 -> new stop 102 after the close.
    bar1 = (100, 104, 99, 103)
    bar2 = (103, 103.5, 101.5, 102.5)       # |o-h| = .5 < 1.5: high then low, low touches 102
    sim = long_case(bar1, extra_bars=[bar2], trail_atr=1.0)
    tr = sim.trades_frame().iloc[0]
    assert tr.reason == "trail" and tr.exit_price == pytest.approx(102) and tr.r == pytest.approx(2 / 3)
    assert tr.stop == 97


def test_trailing_stop_does_not_apply_on_the_bar_that_set_it():
    # low 101 is under the new 102 stop, but the stop only applies from the next bar
    sim = run([100, 100], [100, 104], [100, 101], [100, 103], long=[0], trail_atr=1.0)
    assert sim.trades_frame().empty
    assert sim.state[sm.S_STOP] == pytest.approx(102) and sim.state[sm.S_STOP_PREV] == pytest.approx(97)


def test_gap_below_trailed_stop_fills_at_open():
    sim = long_case((100, 104, 99, 103), extra_bars=[(100.5, 101, 100, 100.8)], trail_atr=1.0)
    tr = sim.trades_frame().iloc[0]
    assert tr.exit_price == pytest.approx(100.5) and tr.reason == "trail"


def test_window_start_and_forced_exit():
    n = 10
    o = [100] * n
    h, l, c = [100.5] * n, [99.5] * n, [100] * n
    sim = run(o, h, l, c, long=[0, 1], start=1)               # the bar 0 signal is ignored
    assert sim.state[sm.S_ENTRY_I] == 2
    sim = run(o, h, l, c, long=[0], stop_idx=5)               # flatten at the open of bar 5
    tr = sim.trades_frame().iloc[0]
    assert tr.reason == "window_end" and tr.bars == 4 and sim.state[sm.S_POS] == 0


def test_sum_of_trade_r_equals_sum_of_daily_r_when_flat():
    df = random_walk_df(1200, seed=4)
    bars = Bars(df)
    sleeve = {"id": "t", "instruments": ["X"], "direction": "both", "atr_len": 20,
              "entry": {"type": "donchian_breakout", "len": 30}, "filters": [],
              "exit": {"stop_atr": 2.0, "target_atr": 6.0, "trail_atr": 3.0, "time_stop_bars": 40,
                       "exit_signal": {"type": "donchian_exit", "len": 15}, "reverse_on_opposite": True}}
    sim = sm.simulate(sleeve, bars, 0.0001, "X")
    trades = sim.trades_frame()
    assert len(trades) > 10
    open_mtm = 0.0
    if sim.state[sm.S_POS] != 0:
        open_mtm = sim.state[sm.S_POS] * (bars.close[-1] - sim.state[sm.S_ENTRY_PX]) / sim.state[sm.S_R0]
    assert sim.r_close.sum() == pytest.approx(trades.r.sum() + open_mtm, abs=1e-9)
    assert (sim.r_worst <= sim.r_close + 1e-12).all()


def test_numba_and_python_loops_agree():
    df = random_walk_df(800, seed=9, vol=0.02)
    bars = Bars(df)
    sleeve = {"id": "t", "instruments": ["X"], "direction": "both", "atr_len": 14,
              "entry": {"type": "keltner_breakout", "len": 20, "mult": 1.5}, "filters": [],
              "exit": {"stop_atr": 1.5, "target_atr": 4.0, "trail_atr": 2.5, "time_stop_bars": 25,
                       "exit_signal": None, "reverse_on_opposite": True}}
    sig = build_signals(sleeve, bars)
    args = (bars.open, bars.high, bars.low, bars.close, sig.atr, sig.long, sig.short, sig.exit_long, sig.exit_short,
            1.5, 4.0, 2.5, 25, True, 0.0002, 0, 801)
    fast = sm._run(*args)
    slow = sm._run.py_func(*args) if hasattr(sm._run, "py_func") else sm._run(*args)
    for a, b in zip(fast, slow):
        assert np.allclose(a, b, equal_nan=True)


def test_aggregate_sums_on_master_grid():
    a = long_case((100, 101, 96, 98))
    b = long_case((100, 101, 99, 101))
    grid = sm.master_grid([bars_from([1] * 6, [1] * 6, [1] * 6, [1] * 6).index])
    rc, rw, act = sm.aggregate([a, b], grid)
    assert rc[1] == pytest.approx(a.r_close[1] + b.r_close[1])
    assert act[1] == 1
    frame = sm.daily_frame(grid, rc, rw, act)
    assert list(frame.columns) == ["r_close", "r_worst", "active"]
