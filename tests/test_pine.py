"""Pine v6 generation: structure, inputs, header, lint, and the order logic against the simulator."""
import copy
import itertools
import re

import numpy as np
import pytest
from helpers import random_walk_df

from weathervane import config, mutate, pine, spec
from weathervane import simulator as sm
from weathervane.live import live_sleeves
from weathervane.signals_core import Bars, build_signals

UNIVERSE = config.load_universe()


def champion(pid="ftmo"):
    return config.read_json(config.root() / "champion" / f"{pid}.json")


def render_sleeve(sleeve, pid="ftmo", risk=0.005):
    return pine.render(pid, "test-champ", sleeve, risk, UNIVERSE, "2026-10-29")


def input_vars(text):
    return dict(re.findall(r"^(i_\w+) = input\.(\w+)\(", pine.strip_strings_and_comments(text), re.M))


def sleeve_with(entry=None, filters=(), exit_=None, direction="both"):
    base = copy.deepcopy(champion()["sleeves"][0])
    base["param_space"] = {}
    if entry:
        base["entry"] = entry
    base["filters"] = list(filters)
    if exit_:
        base["exit"] = exit_
    base["direction"] = direction
    return spec.normalise(base)


EXITS = [
    {"stop_atr": 2.0, "reverse_on_opposite": True},
    {"stop_atr": 1.5, "target_atr": 3.0, "trail_atr": None, "reverse_on_opposite": False},
    {"stop_atr": 2.5, "trail_atr": 3.0, "time_stop_bars": 20, "exit_signal": {"type": "donchian_exit", "len": 15}},
    {"stop_atr": 3.0, "target_atr": 5.0, "trail_atr": 4.0, "time_stop_bars": 30,
     "exit_signal": {"type": "ma_exit", "kind": "sma", "len": 40}, "reverse_on_opposite": True},
]


def expected_input_count(s):
    n = 3 + len(spec.ENTRY_TYPES[s["entry"]["type"]]) + sum(len(spec.FILTER_TYPES[f["type"]]) for f in s["filters"]) + 2
    ex = s["exit"]
    n += sum(ex.get(k) is not None for k in ("target_atr", "trail_atr", "time_stop_bars"))
    if ex.get("exit_signal"):
        n += len(spec.EXIT_SIGNAL_TYPES[ex["exit_signal"]["type"]])
    return n


def test_seed_scripts_have_version_header_and_lint_clean():
    for pid in ("ftmo", "etf"):
        for sleeve in live_sleeves(champion(pid)):
            text = render_sleeve(sleeve, pid)
            lines = text.splitlines()
            assert lines[0] == "//@version=6"
            assert "generated, do not edit" in text.lower()
            assert f"Champion: test-champ" in text and "Generated: 2026-10-29" in text
            assert pine.lint(text) == []


def test_header_lists_instruments_and_tradingview_symbols():
    text = render_sleeve(live_sleeves(champion("ftmo"))[0])
    for inst_id in champion("ftmo")["sleeves"][0]["instruments"]:
        assert UNIVERSE[inst_id].tv in text and inst_id in text


def test_strategy_call_matches_the_design():
    text = render_sleeve(live_sleeves(champion("ftmo"))[0])
    for needle in ("overlay = true", "process_orders_on_close = false", "calc_on_every_tick = false", "pyramiding = 0",
                   "initial_capital = 100000", "commission_type = strategy.commission.percent"):
        assert needle in text
    sleeve = champion("ftmo")["sleeves"][0]
    insts = [UNIVERSE[i] for i in sleeve["instruments"]]
    expected = sum(i.cost_bps for i in insts) / len(insts) / 2 / 100
    value = float(re.search(r"commission_value = ([0-9.]+)", text).group(1))
    assert value == pytest.approx(expected, abs=1e-7)


def test_inputs_default_to_live_params_and_risk():
    ch = champion("ftmo")
    ch["live_params"]["trend_a"] = {"entry.len": 35, "exit.stop_atr": 2.25, "exit.trail_atr": 3.5}
    sleeve = live_sleeves(ch)[0]
    text = render_sleeve(sleeve, risk=0.0125)
    assert re.search(r"i_entry_len = input\.int\(35,", text)
    assert re.search(r"i_stop_atr = input\.float\(2\.25,", text)
    assert re.search(r"i_trail_atr = input\.float\(3\.5,", text)
    assert re.search(r"i_risk = input\.float\(1\.25,", text)
    assert re.search(r'i_direction = input\.string\("both",', text)
    assert re.search(r"i_f0_len = input\.int\(200,", text) and re.search(r"i_xs_len = input\.int\(20,", text)
    assert re.search(r"i_reverse = input\.bool\(true,", text)


@pytest.mark.parametrize("entry_type", list(spec.ENTRY_TYPES))
@pytest.mark.parametrize("exit_", EXITS)
def test_every_entry_type_and_exit_combination_renders_complete_inputs(entry_type, exit_):
    sleeve = sleeve_with(entry=mutate.default_block(spec.ENTRY_TYPES, entry_type), exit_=copy.deepcopy(exit_))
    text = render_sleeve(sleeve)
    assert pine.lint(text) == []
    assert len(input_vars(text)) == expected_input_count(sleeve)
    ex = sleeve["exit"]
    assert ("i_target_atr" in text) == (ex.get("target_atr") is not None)
    assert ("i_trail_atr" in text) == (ex.get("trail_atr") is not None)
    assert ("i_time_stop" in text) == (ex.get("time_stop_bars") is not None)
    assert ("i_xs_len" in text) == bool(ex.get("exit_signal"))
    assert "style = plot.style_linebr" in text and "strategy.exit(" in text and "strategy.entry(" in text
    assert ("strategy.close_all" in text)


@pytest.mark.parametrize("kinds", [list(c) for r in (1, 2) for c in itertools.combinations(spec.FILTER_TYPES, r)])
def test_filters_render_with_their_inputs(kinds):
    filters = [mutate.default_block(spec.FILTER_TYPES, k) for k in kinds]
    sleeve = sleeve_with(filters=filters)
    text = render_sleeve(sleeve)
    assert pine.lint(text) == []
    assert len(input_vars(text)) == expected_input_count(sleeve)
    for k in range(len(kinds)):
        assert f"f{k}Long" in text and f"f{k}Short" in text
        assert f"and f{k}Long" in text and f"and f{k}Short" in text


def test_four_filters_render():
    filters = [mutate.default_block(spec.FILTER_TYPES, k) for k in ("trend_ma", "adx_min", "atr_rank", "roc_dir")]
    text = render_sleeve(sleeve_with(filters=filters))
    assert pine.lint(text) == [] and "f3Long" in text


@pytest.mark.parametrize("direction", ["both", "long", "short"])
def test_direction_is_an_input(direction):
    text = render_sleeve(sleeve_with(direction=direction))
    assert f'input.string("{direction}", "Direction"' in text
    assert 'i_direction != "short"' in text and 'i_direction != "long"' in text


def test_semantics_lines_present():
    """The lines that carry the simulator's rules (DESIGN 5 and 6)."""
    sleeve = sleeve_with(exit_=copy.deepcopy(EXITS[3]))
    text = render_sleeve(sleeve)
    for needle in ["longRaw = close > ta.highest(high, i_entry_len)[1]", "shortRaw = close < ta.lowest(low, i_entry_len)[1]",
                   "goLong = longSig and not shortSig and validAtr", "stopDist = i_stop_atr * atrv",
                   "qty = stopDist > 0 ? (strategy.equity * i_risk / 100.0) / (stopDist * syminfo.pointvalue) : 0.0",
                   "stopLvl := close - sig * stopDist", "tgtLvl := close + sig * i_target_atr * atrv",
                   "extreme := barsHeld == 0 ? (side > 0 ? high : low) : (side > 0 ? math.max(extreme, high) : math.min(extreme, low))",
                   "stopLvl := side > 0 ? math.max(stopLvl, cand) : math.min(stopLvl, cand)",
                   "cand = extreme - side * i_trail_atr * atrv", "isReverse = inPos and i_reverse and sig == -side",
                   "exitNow = inPos and (barsHeld >= i_time_stop or (side > 0 ? xLong : xShort))",
                   'strategy.exit("XL", from_entry = "L", stop = stopLvl, limit = tgtLvl)',
                   'strategy.entry("S", strategy.short, qty = qty)', 'strategy.close_all(comment = "exit")',
                   "xMa = f_ma(i_xs_kind, close, i_xs_len)"]:
        assert needle in text, needle


def test_indicator_builtins_per_entry_type():
    expect = {"donchian_breakout": "ta.highest(high", "ma_cross": "ta.crossover(maFast, maSlow)",
              "keltner_breakout": "ta.atr(i_entry_len)", "momentum_roc": "ta.roc(close, i_entry_len)",
              "rsi_reversion": "ta.rsi(close, i_entry_len)", "bollinger_reversion": "ta.stdev(close, i_entry_len)"}
    for kind, needle in expect.items():
        text = render_sleeve(sleeve_with(entry=mutate.default_block(spec.ENTRY_TYPES, kind)))
        assert needle in text
    text = render_sleeve(sleeve_with(filters=[mutate.default_block(spec.FILTER_TYPES, "adx_min")]))
    assert "ta.dmi(i_f0_len, i_f0_len)" in text
    text = render_sleeve(sleeve_with(filters=[mutate.default_block(spec.FILTER_TYPES, "atr_rank")]))
    assert "ta.percentrank(ta.atr(i_f0_len) / close, i_f0_lookback)" in text


def test_invalid_sleeve_is_refused():
    s = sleeve_with()
    s["entry"]["len"] = 1
    with pytest.raises(spec.SpecError):
        render_sleeve(s)


def test_lint_catches_problems():
    good = render_sleeve(live_sleeves(champion("ftmo"))[0])
    assert pine.lint(good) == []
    assert pine.lint(good.replace("//@version=6", "//@version=5"))
    assert pine.lint(good.replace("ta.atr(i_atr_len)", "ta.atr(i_atr_len", 1))
    assert pine.lint(good.replace("i_entry_len = input.int", "i_entry_length = input.int", 1))
    assert pine.lint(good.replace("generated, do not edit", "x", 1).replace("Generated, do not edit", "x"))
    assert pine.lint(good.replace("\n    stopLvl", "\n\tstopLvl", 1))
    assert pine.lint(good.replace("\n    extreme :=", "\n  extreme :=", 1))


def test_generate_writes_files_and_removes_stale(workspace, cfg):
    stale = workspace / "pine" / "ftmo_old_sleeve.pine"
    other = workspace / "pine" / "keepme.pine"
    stale.parent.mkdir()
    stale.write_text("old")
    other.write_text("other")
    paths = pine.generate(["ftmo", "etf"], cfg, today="2026-01-02")
    assert sorted(p.name for p in paths) == ["etf_etf_trend.pine", "ftmo_trend_a.pine"]
    assert not stale.exists() and other.exists()
    text = (workspace / "pine" / "ftmo_trend_a.pine").read_text()
    assert "Generated: 2026-01-02" in text and "Champion: seed-ftmo" in text
    assert re.search(r"i_risk = input\.float\(0\.5,", text), "unscored champion uses the default risk"
    assert "—" not in text


def test_generated_text_never_contains_em_dash():
    for exit_ in EXITS:
        assert "—" not in render_sleeve(sleeve_with(exit_=copy.deepcopy(exit_)))


def test_pynescript_parses_every_generated_script():
    pyne = pytest.importorskip("pynescript.ast", reason="pynescript is an optional syntax checker")
    for entry_type, exit_ in itertools.product(spec.ENTRY_TYPES, EXITS):
        text = render_sleeve(sleeve_with(entry=mutate.default_block(spec.ENTRY_TYPES, entry_type), exit_=copy.deepcopy(exit_)))
        pyne.parse(text)


# ---- the order logic against the simulator ---------------------------------------------------------------------------

MODEL_SLEEVES = [
    ("donchian_breakout", {"len": 10}, [], {"stop_atr": 1.5, "reverse_on_opposite": True}),
    ("donchian_breakout", {"len": 12}, [{"type": "trend_ma", "kind": "ema", "len": 50}],
     {"stop_atr": 2.0, "trail_atr": 2.5, "reverse_on_opposite": True}),
    ("donchian_breakout", {"len": 8}, [], {"stop_atr": 1.5, "target_atr": 2.5, "reverse_on_opposite": False}),
    ("ma_cross", {"kind": "ema", "fast": 8, "slow": 21}, [], {"stop_atr": 2.0, "trail_atr": 3.0, "time_stop_bars": 12,
                                                              "reverse_on_opposite": True}),
    ("rsi_reversion", {"len": 7, "lower": 30.0, "upper": 70.0}, [], {"stop_atr": 1.5, "target_atr": 2.0, "time_stop_bars": 8,
                                                                      "reverse_on_opposite": True}),
    ("keltner_breakout", {"len": 15, "mult": 1.5}, [], {"stop_atr": 2.0, "trail_atr": 2.0,
                                                         "exit_signal": {"type": "donchian_exit", "len": 8},
                                                         "reverse_on_opposite": True}),
    ("donchian_breakout", {"len": 10}, [], {"stop_atr": 1.5, "trail_atr": 2.0, "target_atr": 6.0, "time_stop_bars": 25,
                                           "exit_signal": {"type": "ma_exit", "kind": "ema", "len": 20},
                                           "reverse_on_opposite": False}),
]


@pytest.mark.parametrize("direction", ["both", "long", "short"])
@pytest.mark.parametrize("case", range(len(MODEL_SLEEVES)))
@pytest.mark.parametrize("seed", [1, 2])
def test_pine_logic_matches_the_simulator(case, seed, direction):
    from pine_model import run as run_model
    kind, entry, filters, exit_ = MODEL_SLEEVES[case]
    sleeve = sleeve_with(entry={"type": kind, **entry}, filters=filters, exit_=copy.deepcopy(exit_), direction=direction)
    spec.validate_sleeve(sleeve)
    bars = Bars(random_walk_df(n=1200, seed=seed * 10 + case, vol=0.012))
    half = 0.0001
    sig = build_signals(sleeve, bars)
    sim = sm.simulate(sleeve, bars, half, "X", sig).trades_frame()
    model = run_model(sleeve, bars, sig, half)
    closed = [t for t in model]
    assert len(closed) == len(sim), (len(closed), len(sim))
    assert len(sim) > 3 or direction != "both"
    for m, (_, s) in zip(closed, sim.iterrows()):
        assert m["side"] == (1 if s.side == "long" else -1)
        assert bars.index[m["entry_bar"]] == s.entry_date and bars.index[m["exit_bar"]] == s.exit_date
        assert m["entry"] == pytest.approx(s.entry_price) and m["exit"] == pytest.approx(s.exit_price)
        assert m["r"] == pytest.approx(s.r)
