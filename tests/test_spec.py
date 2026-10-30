"""Strategy grammar validation, hashing and parameter helpers."""
import copy

import pytest

from weathervane import config, spec

KNOWN = list(config.load_universe())


def seed_sleeve():
    return copy.deepcopy(config.read_json(config.root() / "champion" / "ftmo.json")["sleeves"][0])


def check(mutator, match=None):
    s = seed_sleeve()
    mutator(s)
    with pytest.raises(spec.SpecError, match=match):
        spec.validate_sleeve(s, KNOWN)


def test_seed_champions_are_valid():
    for pid in ("ftmo", "etf"):
        ch = config.read_json(config.root() / "champion" / f"{pid}.json")
        spec.validate_portfolio(ch, KNOWN, config.load_portfolios()[pid].instruments)


def test_unknown_keys_are_rejected():
    check(lambda s: s.update(colour="red"), "unknown")
    check(lambda s: s["entry"].update(extra=1), "unknown keys")
    check(lambda s: s["exit"].update(extra=1), "unknown keys")
    check(lambda s: s["filters"][0].update(extra=1), "unknown keys")


def test_missing_required_keys():
    check(lambda s: s.pop("entry"), "missing")
    check(lambda s: s["entry"].pop("len"), "missing keys")
    check(lambda s: s["exit"].pop("stop_atr"), "stop_atr")


@pytest.mark.parametrize("path,value", [("entry.len", 1), ("entry.len", 401), ("atr_len", 1), ("exit.stop_atr", 0.4),
                                         ("exit.stop_atr", 10.5), ("exit.trail_atr", 11), ("filters.0.len", 500)])
def test_numeric_bounds(path, value):
    def mut(s):
        spec.set_path(s, path, value)
    check(mut)


def test_type_errors():
    check(lambda s: s["entry"].update(len="55"))
    check(lambda s: s["entry"].update(len=55.5), "integer")
    check(lambda s: s["entry"].update(len=True))
    check(lambda s: s["entry"].update(type="nope"), "entry.type")
    check(lambda s: s.update(direction="up"), "direction")
    check(lambda s: s["filters"][0].update(kind="wma"), "one of")
    check(lambda s: s["exit"].update(reverse_on_opposite="yes"), "true or false")


def test_instrument_checks():
    check(lambda s: s.update(instruments=["EURUSD", "XXXXXX"]), "unknown instruments")
    check(lambda s: s.update(instruments=[]), "non-empty")
    check(lambda s: s.update(instruments=["EURUSD", "EURUSD"]), "unique")
    s = seed_sleeve()
    with pytest.raises(spec.SpecError, match="not allowed"):
        spec.validate_sleeve(s, KNOWN, allowed=["EURUSD"])


def test_at_most_four_filters():
    s = seed_sleeve()
    s["filters"] = [{"type": "roc_dir", "len": 20}] * 4
    spec.validate_sleeve(s, KNOWN)
    s["filters"].append({"type": "roc_dir", "len": 20})
    with pytest.raises(spec.SpecError, match="at most 4"):
        spec.validate_sleeve(s, KNOWN)


def test_cross_constraints():
    s = {"id": "x", "instruments": ["EURUSD"], "direction": "both", "entry": {"type": "ma_cross", "kind": "ema", "fast": 60, "slow": 50},
         "exit": {"stop_atr": 2.0}, "atr_len": 20}
    with pytest.raises(spec.SpecError, match="fast must be below"):
        spec.validate_sleeve(spec.normalise(s), KNOWN)
    s["entry"] = {"type": "rsi_reversion", "len": 14, "lower": 60.0, "upper": 55.0}
    with pytest.raises(spec.SpecError, match="outside"):          # the grammar bounds already keep lower below upper
        spec.validate_sleeve(spec.normalise(s), KNOWN)
    s["entry"] = {"type": "donchian_breakout", "len": 20}
    s["filters"] = [{"type": "atr_rank", "len": 14, "lookback": 100, "min": 50.0, "max": 40.0}]
    with pytest.raises(spec.SpecError, match="min must be below max"):
        spec.validate_sleeve(spec.normalise(s), KNOWN)


def test_param_space_rules():
    check(lambda s: s["param_space"].update({"entry.nope": [1, 2, 1]}), "param_space")
    check(lambda s: s["param_space"].update({"entry.len": [20, 100]}), r"\[low, high, step\]")
    check(lambda s: s["param_space"].update({"entry.len": [100, 20, 5]}), "low <= high")
    check(lambda s: s["param_space"].update({"entry.len": [20, 100, 0]}), "step > 0")
    check(lambda s: s["param_space"].update({"entry.len": [1, 100, 5]}), "outside")
    check(lambda s: s["param_space"].update({"entry.len": [20.5, 100, 5]}), "integers")
    check(lambda s: s["param_space"].update({"exit.target_atr": [2.0, 4.0, 0.5]}), "param_space")      # target is null
    check(lambda s: s["param_space"].update({"filters.0.kind": [1, 2, 1]}), "param_space")             # not a number
    check(lambda s: s["param_space"].update({"filters.5.len": [1, 2, 1]}), "param_space")


def test_portfolio_limits():
    s = seed_sleeve()
    pf = {"sleeves": [{**copy.deepcopy(s), "id": f"s{i}"} for i in range(4)]}
    with pytest.raises(spec.SpecError, match="at most 3"):
        spec.validate_portfolio(pf, KNOWN)
    pf = {"sleeves": [copy.deepcopy(s), copy.deepcopy(s)]}
    with pytest.raises(spec.SpecError, match="unique"):
        spec.validate_portfolio(pf, KNOWN)
    with pytest.raises(spec.SpecError):
        spec.validate_portfolio({"sleeves": []}, KNOWN)
    with pytest.raises(spec.SpecError):
        spec.validate_portfolio("nope", KNOWN)


def test_every_entry_filter_and_exit_type_validates_with_defaults():
    from weathervane import mutate
    base = seed_sleeve()
    for kind in spec.ENTRY_TYPES:
        s = copy.deepcopy(base)
        s["entry"] = mutate.default_block(spec.ENTRY_TYPES, kind)
        s["param_space"] = {}
        spec.validate_sleeve(s, KNOWN)
    for kind in spec.FILTER_TYPES:
        s = copy.deepcopy(base)
        s["filters"] = [mutate.default_block(spec.FILTER_TYPES, kind)]
        s["param_space"] = {}
        spec.validate_sleeve(s, KNOWN)
    for kind in spec.EXIT_SIGNAL_TYPES:
        s = copy.deepcopy(base)
        s["exit"]["exit_signal"] = mutate.default_block(spec.EXIT_SIGNAL_TYPES, kind)
        s["param_space"] = {}
        spec.validate_sleeve(s, KNOWN)


def test_normalise_makes_equal_strategies_hash_equal():
    a = seed_sleeve()
    b = copy.deepcopy(a)
    b["exit"].pop("target_atr")
    b["exit"].pop("time_stop_bars")
    b.pop("param_space")
    c = copy.deepcopy(a)
    c["param_space"] = {}
    assert spec.sleeve_hash(b) == spec.sleeve_hash(c)
    assert spec.sleeve_hash(a) != spec.sleeve_hash(c)
    d = copy.deepcopy(c)
    d["exit"]["stop_atr"] = 2.75
    assert spec.sleeve_hash(d) != spec.sleeve_hash(c)


def test_portfolio_hash_ignores_sleeve_order_and_scores():
    s1, s2 = seed_sleeve(), {**seed_sleeve(), "id": "other"}
    assert spec.portfolio_hash({"sleeves": [s1, s2]}) == spec.portfolio_hash({"sleeves": [s2, s1], "risk": 0.5})


def test_grid_values_and_apply_params():
    assert spec.grid_values([20, 100, 20]) == [20, 40, 60, 80, 100]
    assert spec.grid_values([1.5, 2.5, 0.25]) == [1.5, 1.75, 2.0, 2.25, 2.5]
    s = seed_sleeve()
    out = spec.apply_params(s, {"entry.len": 30.4, "exit.stop_atr": 2})
    assert out["entry"]["len"] == 30 and isinstance(out["entry"]["len"], int)
    assert out["exit"]["stop_atr"] == 2.0 and isinstance(out["exit"]["stop_atr"], float)
    assert s["entry"]["len"] == 55, "the input sleeve is not modified"
    assert spec.default_params(s) == {"entry.len": 55, "exit.stop_atr": 2.5, "exit.trail_atr": 3.0}


def test_describe_grammar_mentions_every_type():
    text = spec.describe_grammar()
    for name in (*spec.ENTRY_TYPES, *spec.FILTER_TYPES, *spec.EXIT_SIGNAL_TYPES):
        assert name in text
