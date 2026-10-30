"""Live signals: state to action mapping, consistency with the next bar, sizing, risk guard, output files."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from helpers import random_walk_df

from weathervane import config, live, spec
from weathervane import simulator as sm
from weathervane.signals_core import Bars, build_signals

UNIVERSE = config.load_universe()


def state(**kw):
    s = np.zeros(sm.STATE_LEN)
    s[sm.S_STOP] = s[sm.S_STOP_PREV] = np.nan
    s[sm.S_TARGET] = s[sm.S_PEND_STOP] = s[sm.S_PEND_TARGET] = np.nan
    names = {"pos": sm.S_POS, "pend_entry": sm.S_PEND_ENTRY, "pend_exit": sm.S_PEND_EXIT, "stop": sm.S_STOP, "stop_prev": sm.S_STOP_PREV}
    for k, v in kw.items():
        s[names[k]] = v
    return s


@pytest.mark.parametrize("kw,expected", [
    ({}, "FLAT"), ({"pend_entry": 1}, "ENTER_LONG"), ({"pend_entry": -1}, "ENTER_SHORT"),
    ({"pos": 1, "stop": 9.0, "stop_prev": 9.0}, "HOLD"), ({"pos": -1, "stop": 10.0, "stop_prev": 10.0}, "HOLD"),
    ({"pos": 1, "stop": 11.0, "stop_prev": 10.0}, "MOVE_STOP"),
    ({"pos": 1, "pend_exit": sm.R_SIGNAL}, "EXIT"), ({"pos": -1, "pend_exit": sm.R_TIME}, "EXIT"),
    ({"pos": 1, "pend_entry": -1, "pend_exit": sm.R_REVERSAL}, "REVERSE_TO_SHORT"),
    ({"pos": -1, "pend_entry": 1, "pend_exit": sm.R_REVERSAL}, "REVERSE_TO_LONG"),
])
def test_action_from_state(kw, expected):
    assert live.action_from_state(state(**kw))[0] == expected


def mini_champion(entry=None, exit_=None, filters=()):
    sleeve = {"id": "t", "instruments": ["EURUSD", "US500"], "direction": "both",
              "entry": entry or {"type": "donchian_breakout", "len": 15}, "filters": list(filters),
              "exit": exit_ or {"stop_atr": 2.0, "trail_atr": 3.0, "exit_signal": {"type": "donchian_exit", "len": 8},
                                "reverse_on_opposite": True}, "atr_len": 14, "param_space": {}}
    return {"id": "mini", "portfolio": "ftmo", "risk": 0.01, "sleeves": [spec.normalise(sleeve)], "live_params": {}}


def data_set(n=900, seed=3):
    return {"EURUSD": random_walk_df(n, seed, vol=0.01), "US500": random_walk_df(n, seed + 1, vol=0.012)}


def test_live_action_matches_what_happens_on_the_next_bar():
    """Truncate the history at k, read the action, then check the simulator on k+1 bars did exactly that at bar k."""
    champion = mini_champion()
    full = data_set()
    seen = set()
    for k in range(300, 880, 3):
        cut = {i: df.iloc[:k] for i, df in full.items()}
        nxt = {i: df.iloc[: k + 1] for i, df in full.items()}
        for run in live.run_live(champion, cut, UNIVERSE):
            action, _ = live.action_from_state(run.sim.state)
            inst = run.instrument.id
            after = sm.simulate(run.sleeve, Bars(nxt[inst]), UNIVERSE[inst].half_cost, inst)
            trades = after.trades_frame()
            opened_at_k = (after.state[sm.S_POS] != 0 and int(after.state[sm.S_ENTRY_I]) == k) or \
                          (len(after.trade_i) and (after.trade_i[:, 0] == k).any())
            side_at_k = None
            if opened_at_k:
                side_at_k = after.state[sm.S_POS] if after.state[sm.S_POS] != 0 and int(after.state[sm.S_ENTRY_I]) == k else \
                    after.trade_f[after.trade_i[:, 0] == k][0, 0]
            closed_at_k = bool(len(after.trade_i) and (after.trade_i[:, 1] == k).any()
                               and sm.REASONS[int(after.trade_reason[after.trade_i[:, 1] == k][0])] in ("time_stop", "exit_signal", "reversal"))
            if action.startswith("ENTER") or action.startswith("REVERSE"):
                assert opened_at_k and side_at_k == (1 if action.endswith("LONG") else -1), (k, inst, action)
            else:
                assert not opened_at_k, (k, inst, action)
            assert closed_at_k == (action == "EXIT" or action.startswith("REVERSE")), (k, inst, action)
            seen.add(action)
    assert {"FLAT", "HOLD", "EXIT"} <= seen
    assert seen & {"ENTER_LONG", "ENTER_SHORT"} and seen & {"MOVE_STOP"}


def test_rows_are_sized_to_the_risk_fraction():
    champion = mini_champion()
    rules = config.load_rules("ftmo_2step_swing")
    rows, risk = live.signal_rows("ftmo", champion, data_set(), UNIVERSE, rules, config.load_engine())
    assert risk == 0.01
    for r in rows:
        if r["stop_distance"]:
            assert r["risk_cash"] == pytest.approx(rules.account_size * 0.01 * r["guard_scale"], abs=0.01)
            assert r["units"] * r["stop_distance"] == pytest.approx(r["risk_cash"], rel=1e-9)
            sign = 1 if r["side"] == "long" else -1
            assert sign * (r["entry_price"] - r["stop"]) > 0 or r["action"] in ("HOLD", "MOVE_STOP")
        else:
            assert r["action"] == "FLAT" and r["units"] == 0
        assert r["tv"] == UNIVERSE[r["instrument"]].tv and r["contract_note"]


def test_max_risk_per_trade_caps_the_live_risk():
    base = config.load_rules("ftmo_2step_swing")
    capped = config.Rules(**{**base.__dict__, "max_risk": 0.004})
    assert config.live_risk({"risk": 0.01}, config.load_engine(), capped) == 0.004
    assert config.live_risk({"risk": None}, config.load_engine(), base) == config.load_engine()["research"]["default_risk"]


def test_risk_guard_scales_new_entries_only():
    rules = config.load_rules("ftmo_2step_swing")            # budget = 0.6 * 5% * 100000 = 3000
    def row(action, risk_cash, units=1.0, side="long", stop=95.0):
        return {"action": action, "risk_cash": risk_cash, "units": units, "side": side, "close": 100.0, "stop": stop,
                "stop_distance": 5.0, "risk_fraction": 0.01, "guard_scale": 1.0, "guard_scaled": False, "reason": "x"}
    held = row("HOLD", 0, units=400.0)                      # open risk 400 * 5 = 2000
    new1, new2 = row("ENTER_LONG", 1000, 200.0), row("ENTER_SHORT", 1000, 200.0, "short", 105.0)
    live.apply_risk_guard([held, new1, new2], rules, rules.account_size)
    assert new1["guard_scaled"] and new2["guard_scaled"] and held["guard_scale"] == 1.0
    assert new1["guard_scale"] == pytest.approx(0.5) and new1["risk_cash"] == pytest.approx(500.0, abs=0.01)
    assert "risk guard" in new1["reason"]
    total = live.open_risk(held) + new1["risk_cash"] + new2["risk_cash"]
    assert total <= 3000 + 1e-6
    ok = row("ENTER_LONG", 500, 100.0)
    live.apply_risk_guard([held, ok], rules, rules.account_size)
    assert not ok["guard_scaled"]


def test_generate_writes_outputs_and_history_once(workspace, cfg):
    payload = live.generate(["ftmo", "etf"], cfg, synthetic=True)
    latest = json.loads((workspace / "signals" / "latest.json").read_text())
    assert set(latest["portfolios"]) == {"ftmo", "etf"} and latest["data_date"] == "2026-10-28"
    md = (workspace / "signals" / "latest.md").read_text()
    assert "| Instrument | TradingView | Action |" in md and "OANDA:EURUSD" in md and "AMEX:SPY" in md
    assert "—" not in md
    n_first = payload["history_added"]
    hist = pd.read_csv(workspace / "signals" / "history.csv") if n_first else None
    again = live.generate(["ftmo", "etf"], cfg, synthetic=True)
    assert again["history_added"] == 0
    if hist is not None:
        assert len(hist) == n_first and not hist["action"].isin(["HOLD", "FLAT"]).any()
    actions = {r["action"] for p in latest["portfolios"].values() for r in p["rows"]}
    assert actions <= {"ENTER_LONG", "ENTER_SHORT", "EXIT", "REVERSE_TO_LONG", "REVERSE_TO_SHORT", "HOLD", "MOVE_STOP", "FLAT"}
    for p in latest["portfolios"].values():
        assert p["risk_fraction"] == pytest.approx(0.005) and p["account_size"] == 100000


def test_generate_without_data_raises(workspace, cfg):
    with pytest.raises(SystemExit):
        live.generate(["ftmo"], cfg, synthetic=False)


def test_live_params_are_used(workspace):
    ch = mini_champion()
    ch["sleeves"][0]["param_space"] = {"entry.len": [10, 40, 5]}
    ch["live_params"] = {"t": {"entry.len": 35}}
    assert live.live_sleeves(ch)[0]["entry"]["len"] == 35
    assert ch["sleeves"][0]["entry"]["len"] == 15, "champion file content is not modified"
