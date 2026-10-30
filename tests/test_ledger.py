"""Internal paper ledger: replay from the start date, exam state, champion changes, files."""
import copy

import numpy as np
import pandas as pd
import pytest

from weathervane import config, ledger, simulator as sm
from weathervane.config import Rules

START = "2026-04-01"


def run_ledger(cfg, ids=("ftmo",), start=START):
    return ledger.update(list(ids), cfg, synthetic=True, start_override=start)


def read_ledger():
    return pd.read_csv(ledger.ledger_path()), pd.read_csv(ledger.equity_path(), parse_dates=["date"])


def test_ledger_writes_files_and_state(workspace, cfg):
    state = run_ledger(cfg)
    trades, eq = read_ledger()
    st = state["portfolios"]["ftmo"]
    assert st["start"] == START and st["data_date"] == "2026-10-28" and st["days"] == len(eq) > 100
    assert eq["date"].iloc[0] == pd.Timestamp(START) and eq["date"].is_monotonic_increasing
    assert state["paper_start_date"] == START
    assert list(trades.columns) == ledger.LEDGER_COLUMNS and list(eq.columns) == ledger.EQUITY_COLUMNS
    assert set(trades["champion"]) == {"seed-ftmo"} and set(eq["champion"]) == {"seed-ftmo"}
    assert set(trades["status"]) <= {"open", "closed"}
    assert (ledger.state_path()).exists()


def test_equity_reconciles_with_trades(workspace, cfg):
    run_ledger(cfg)
    trades, eq = read_ledger()
    account = 100000.0
    assert eq["equity"].iloc[-1] - account == pytest.approx(trades["pnl_cash"].sum(), rel=1e-6, abs=0.5)
    assert eq["equity"].iloc[0] == pytest.approx(account)
    assert eq["equity"].to_numpy() == pytest.approx(account * (1 + eq["ret"].cumsum().to_numpy()))


def test_trades_fill_at_next_open_and_never_before_start(workspace, cfg):
    run_ledger(cfg)
    trades, _ = read_ledger()
    assert (pd.to_datetime(trades["entry_date"]) > pd.Timestamp(START)).all()
    closed = trades[trades["status"] == "closed"]
    assert (pd.to_datetime(closed["exit_date"]) >= pd.to_datetime(closed["entry_date"])).all()
    from weathervane.data import load
    data = load(["EURUSD"], cfg, synthetic=True)["EURUSD"]
    eur = trades[(trades["instrument"] == "EURUSD")]
    half = config.load_universe()["EURUSD"].half_cost
    for r in eur.itertuples():
        open_px = data.loc[pd.Timestamp(r.entry_date), "open"]
        expected = open_px * (1 + half) if r.side == "long" else open_px * (1 - half)
        assert r.entry_price == pytest.approx(expected, rel=1e-5)


def test_ledger_is_idempotent(workspace, cfg):
    run_ledger(cfg)
    a_trades, a_eq = read_ledger()
    run_ledger(cfg)
    b_trades, b_eq = read_ledger()
    pd.testing.assert_frame_equal(a_trades, b_trades)
    pd.testing.assert_frame_equal(a_eq, b_eq)


def test_first_run_start_is_the_data_date_when_unset(workspace, cfg):
    state = ledger.update(["etf"], cfg, synthetic=True)
    assert state["paper_start_date"] == "2026-10-28"
    assert state["portfolios"]["etf"]["days"] == 1


def test_changing_the_configured_start_restarts_the_ledger(workspace, cfg):
    run_ledger(cfg, start="2026-06-01")
    state = run_ledger(cfg, start="2026-09-01")
    assert state["paper_start_date"] == "2026-09-01" and state["portfolios"]["ftmo"]["start"] == "2026-09-01"


def test_updating_one_portfolio_keeps_the_other(workspace, cfg):
    run_ledger(cfg, ids=("ftmo", "etf"))
    run_ledger(cfg, ids=("etf",))
    trades, eq = read_ledger()
    assert {"ftmo", "etf"} == set(eq["portfolio"]) and {"ftmo", "etf"} <= set(trades["portfolio"]) | {"ftmo", "etf"}
    assert (eq["portfolio"] == "ftmo").sum() > 100


def test_exam_tracker_events_follow_the_rules(workspace, cfg):
    run_ledger(cfg)
    state = ledger.load_state()["portfolios"]["ftmo"]
    _, eq = read_ledger()
    rules = config.load_rules("ftmo_2step_swing")
    from weathervane.exam import ExamTracker
    tracker = ExamTracker(rules)
    events = []
    for r in eq.itertuples():
        out = tracker.day(str(r.date.date()), r.ret, r.worst, r.active)
        if out["event"]:
            events.append(out["event"])
    assert [e["event"] for e in state["exam"]["events"]] == events
    assert state["exam"]["phase"] == tracker.phase + 1 and state["exam"]["attempt"] == tracker.attempt


def force_daily_breach(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily.loc[daily.index[10], ["ret", "worst"]] = [-0.06, -0.06]
    return daily


def test_play_exam_records_breach_and_restarts():
    rules = config.load_rules("ftmo_2step_swing")
    days = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=30), "ret": 0.0005, "worst": -0.001, "active": 1})
    out, tracker = ledger.play_exam(force_daily_breach(days), rules)
    events = out[out["exam_event"] != ""]
    assert list(events["exam_event"]) == ["breach_daily"]
    assert out["exam_attempt"].iloc[-1] == 2 and out["exam_equity"].iloc[11] == pytest.approx(1.0 + 0.0005)


def test_play_exam_passes_both_phases():
    rules = config.load_rules("ftmo_2step_swing")
    days = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=40), "ret": 0.011, "worst": -0.001, "active": 1})
    out, tracker = ledger.play_exam(days, rules)
    assert list(out[out["exam_event"] != ""]["exam_event"])[:2] == ["passed_phase_1", "passed_all"]


def two_segment_state(cfg, change="2026-09-01"):
    """Segments as if the champion had changed on `change`: the first sleeve gets a different stop."""
    ch = config.load_champion("ftmo")
    new = copy.deepcopy(ch)
    new["id"] = "challenger-1"
    new["sleeves"][0]["exit"]["stop_atr"] = 1.5
    new["live_params"]["trend_a"]["exit.stop_atr"] = 1.5
    risk = 0.005
    return [ledger.segment_for(ch, risk, START), ledger.segment_for(new, 0.0075, change)]


def test_champion_change_flattens_old_positions_and_continues(workspace, cfg):
    from weathervane.data import load
    rules = config.load_rules("ftmo_2step_swing")
    segments = two_segment_state(cfg)
    data = load(list(config.load_portfolios()["ftmo"].instruments), cfg, synthetic=True)
    daily, trades, held = ledger.replay(segments, data, config.load_universe(), rules, "ftmo")
    assert set(trades["champion"]) == {"seed-ftmo", "challenger-1"}
    changed = trades[trades["reason"] == "window_end"]
    assert len(changed) >= 1 and (changed["exit_date"] == pd.Timestamp("2026-09-01")).all()    # flattened at the change day's open
    assert (changed["champion"] == "seed-ftmo").all()
    old = trades[trades["champion"] == "seed-ftmo"]
    assert (old["exit_date"] <= pd.Timestamp("2026-09-01")).all()
    new = trades[trades["champion"] == "challenger-1"]
    assert (new["entry_date"] > pd.Timestamp("2026-09-01")).all()
    assert set(daily["champion"]) == {"seed-ftmo", "challenger-1"}
    assert daily.loc[daily["date"] < "2026-09-01", "risk"].eq(0.005).all() and daily.loc[daily["date"] >= "2026-09-01", "risk"].eq(0.0075).all()
    assert all(p["champion"] == "challenger-1" for p in held)


def test_update_segments_logic():
    ch = {"id": "a", "sleeves": [], "live_params": {}}
    segs = ledger.update_segments([], ch, 0.01, "2026-05-05", "2026-05-05")
    assert len(segs) == 1 and segs[0]["start"] == "2026-05-05"
    same = ledger.update_segments(segs, {**ch, "live_params": {"x": 1}}, 0.02, "2026-06-01", "2026-05-05")
    assert len(same) == 1 and same[0]["risk"] == 0.02 and same[0]["live_params"] == {"x": 1} and same[0]["start"] == "2026-05-05"
    new = ledger.update_segments(same, {**ch, "id": "b"}, 0.01, "2026-06-01", "2026-05-05")
    assert [s["champion_id"] for s in new] == ["a", "b"] and new[1]["start"] == "2026-06-01"
    again = ledger.update_segments(new, {**ch, "id": "c"}, 0.01, "2026-06-01", "2026-05-05")
    assert [s["champion_id"] for s in again] == ["a", "c"], "a change on the same day replaces the segment"


def test_champion_file_change_creates_second_segment_through_update(workspace, cfg):
    run_ledger(cfg, start="2026-06-01")
    ch = config.load_champion("ftmo")
    ch["id"] = "ftmo-new"
    config.save_champion(ch)
    state = ledger.update(["ftmo"], cfg, synthetic=True, start_override="2026-06-01")
    segs = state["portfolios"]["ftmo"]["segments"]
    assert [s["champion_id"] for s in segs] == ["seed-ftmo", "ftmo-new"] and segs[1]["start"] == "2026-10-28"
    _, eq = read_ledger()
    assert eq["champion"].iloc[-1] == "ftmo-new" and eq["champion"].iloc[0] == "seed-ftmo"


def test_no_data_raises(workspace, cfg):
    with pytest.raises(SystemExit):
        ledger.update(["ftmo"], cfg, synthetic=False)
