"""Strategy grammar: sleeve and portfolio validation, parameter paths, hashing.

Every construct here has an exact Pine Script v6 equivalent (see pine.py). Unknown keys are rejected.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

DIRECTIONS = ("both", "long", "short")
MA_KINDS = ("ema", "sma")
LEN_MIN, LEN_MAX = 2, 400
ATR_MIN, ATR_MAX = 0.5, 10.0
MAX_PARAM_SPACE = 8
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")


class SpecError(ValueError):
    """Raised when a sleeve or portfolio breaks the grammar."""


@dataclass(frozen=True)
class Param:
    """One grammar parameter: kind int | float | choice, bounds, default and default search range."""

    kind: str
    low: float | None = None
    high: float | None = None
    default: Any = None
    space: tuple | None = None
    choices: tuple = ()


def _int(low: int, high: int, default: int, space: tuple) -> Param:
    return Param("int", low, high, default, space)


def _float(low: float, high: float, default: float, space: tuple) -> Param:
    return Param("float", low, high, default, space)


def _kind(default: str = "ema") -> Param:
    return Param("choice", default=default, choices=MA_KINDS)


ENTRY_TYPES: dict[str, dict[str, Param]] = {
    "donchian_breakout": {"len": _int(LEN_MIN, LEN_MAX, 55, (20, 100, 5))},
    "ma_cross": {"kind": _kind(), "fast": _int(LEN_MIN, LEN_MAX, 20, (10, 50, 5)),
                 "slow": _int(LEN_MIN, LEN_MAX, 100, (60, 200, 10))},
    "keltner_breakout": {"len": _int(LEN_MIN, LEN_MAX, 20, (10, 60, 5)), "mult": _float(ATR_MIN, ATR_MAX, 2.0, (1.0, 3.5, 0.25))},
    "momentum_roc": {"len": _int(LEN_MIN, LEN_MAX, 60, (20, 120, 10)), "thr": _float(0.0, 50.0, 5.0, (2.0, 15.0, 1.0))},
    "rsi_reversion": {"len": _int(LEN_MIN, LEN_MAX, 14, (5, 30, 1)), "lower": _float(1.0, 49.0, 30.0, (15.0, 40.0, 1.0)),
                      "upper": _float(51.0, 99.0, 70.0, (60.0, 85.0, 1.0))},
    "bollinger_reversion": {"len": _int(LEN_MIN, LEN_MAX, 20, (10, 50, 5)), "mult": _float(ATR_MIN, 5.0, 2.0, (1.5, 3.0, 0.25))},
}

FILTER_TYPES: dict[str, dict[str, Param]] = {
    "trend_ma": {"kind": _kind(), "len": _int(LEN_MIN, LEN_MAX, 200, (100, 300, 20))},
    "adx_min": {"len": _int(LEN_MIN, LEN_MAX, 14, (7, 30, 1)), "min": _float(0.0, 60.0, 20.0, (10.0, 35.0, 1.0))},
    "adx_max": {"len": _int(LEN_MIN, LEN_MAX, 14, (7, 30, 1)), "max": _float(5.0, 100.0, 30.0, (20.0, 50.0, 1.0))},
    "atr_rank": {"len": _int(LEN_MIN, LEN_MAX, 14, (7, 30, 1)), "lookback": _int(LEN_MIN, LEN_MAX, 100, (50, 250, 10)),
                 "min": _float(0.0, 100.0, 0.0, (0.0, 40.0, 5.0)), "max": _float(0.0, 100.0, 80.0, (60.0, 100.0, 5.0))},
    "roc_dir": {"len": _int(LEN_MIN, LEN_MAX, 20, (10, 60, 5))},
}

EXIT_SIGNAL_TYPES: dict[str, dict[str, Param]] = {
    "donchian_exit": {"len": _int(LEN_MIN, LEN_MAX, 20, (10, 50, 5))},
    "ma_exit": {"kind": _kind(), "len": _int(LEN_MIN, LEN_MAX, 50, (20, 100, 10))},
}

EXIT_PARAMS: dict[str, Param] = {
    "stop_atr": _float(ATR_MIN, ATR_MAX, 2.5, (1.5, 4.0, 0.25)),
    "target_atr": _float(ATR_MIN, ATR_MAX, 4.0, (2.0, 8.0, 0.5)),
    "trail_atr": _float(ATR_MIN, ATR_MAX, 3.0, (2.0, 5.0, 0.25)),
    "time_stop_bars": _int(1, LEN_MAX, 30, (10, 60, 5)),
}
ATR_LEN = _int(LEN_MIN, LEN_MAX, 20, (10, 40, 5))

SLEEVE_KEYS = {"id", "instruments", "direction", "entry", "filters", "exit", "atr_len", "param_space"}
SLEEVE_REQUIRED = {"id", "instruments", "direction", "entry", "exit", "atr_len"}
EXIT_KEYS = {"stop_atr", "target_atr", "trail_atr", "time_stop_bars", "exit_signal", "reverse_on_opposite"}


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _check_param(name: str, value: Any, p: Param) -> None:
    if p.kind == "choice":
        if value not in p.choices:
            raise SpecError(f"{name} must be one of {p.choices}, got {value!r}")
        return
    if not _num(value):
        raise SpecError(f"{name} must be a number, got {value!r}")
    if p.kind == "int" and (isinstance(value, float) and value != int(value)):
        raise SpecError(f"{name} must be an integer, got {value!r}")
    if not p.low <= value <= p.high:
        raise SpecError(f"{name}={value} outside [{p.low}, {p.high}]")


def _check_typed_block(name: str, block: Any, table: dict[str, dict[str, Param]]) -> None:
    if not isinstance(block, dict) or block.get("type") not in table:
        raise SpecError(f"{name}.type must be one of {sorted(table)}")
    params = table[block["type"]]
    extra = set(block) - set(params) - {"type"}
    missing = set(params) - set(block)
    if extra or missing:
        raise SpecError(f"{name} ({block['type']}): unknown keys {sorted(extra)}, missing keys {sorted(missing)}")
    for key, p in params.items():
        _check_param(f"{name}.{key}", block[key], p)


def _check_cross_constraints(sleeve: dict) -> None:
    entry = sleeve["entry"]
    if entry["type"] == "ma_cross" and entry["fast"] >= entry["slow"]:
        raise SpecError("entry.fast must be below entry.slow")
    if entry["type"] == "rsi_reversion" and entry["lower"] >= entry["upper"]:
        raise SpecError("entry.lower must be below entry.upper")
    for i, flt in enumerate(sleeve.get("filters", [])):
        if flt["type"] == "atr_rank" and flt["min"] >= flt["max"]:
            raise SpecError(f"filters.{i}: min must be below max")


def _check_exit(ex: Any) -> None:
    if not isinstance(ex, dict):
        raise SpecError("exit must be an object")
    extra = set(ex) - EXIT_KEYS
    if extra or "stop_atr" not in ex:
        raise SpecError(f"exit: unknown keys {sorted(extra)} or missing stop_atr")
    _check_param("exit.stop_atr", ex["stop_atr"], EXIT_PARAMS["stop_atr"])
    for key in ("target_atr", "trail_atr", "time_stop_bars"):
        if ex.get(key) is not None:
            _check_param(f"exit.{key}", ex[key], EXIT_PARAMS[key])
    if ex.get("exit_signal") is not None:
        _check_typed_block("exit.exit_signal", ex["exit_signal"], EXIT_SIGNAL_TYPES)
    if not isinstance(ex.get("reverse_on_opposite", True), bool):
        raise SpecError("exit.reverse_on_opposite must be true or false")


def normalise(sleeve: dict) -> dict:
    """Deep copy with optional keys filled in so equal strategies hash equally."""
    out = copy.deepcopy(sleeve)
    out.setdefault("filters", [])
    out.setdefault("param_space", {})
    ex = out.get("exit")
    if isinstance(ex, dict):
        for key in ("target_atr", "trail_atr", "time_stop_bars", "exit_signal"):
            ex.setdefault(key, None)
        ex.setdefault("reverse_on_opposite", True)
    return out


def validate_sleeve(sleeve: Any, known: Iterable[str] | None = None, allowed: Iterable[str] | None = None,
                    max_filters: int = 4) -> None:
    """Raise SpecError unless the sleeve is a legal grammar object."""
    if not isinstance(sleeve, dict):
        raise SpecError("sleeve must be an object")
    keys = set(sleeve)
    if keys - SLEEVE_KEYS or SLEEVE_REQUIRED - keys:
        raise SpecError(f"sleeve keys: unknown {sorted(keys - SLEEVE_KEYS)}, missing {sorted(SLEEVE_REQUIRED - keys)}")
    if not isinstance(sleeve["id"], str) or not ID_RE.match(sleeve["id"]):
        raise SpecError("sleeve id must match [a-z][a-z0-9_]{0,23}")
    _check_instruments(sleeve["instruments"], known, allowed)
    if sleeve["direction"] not in DIRECTIONS:
        raise SpecError(f"direction must be one of {DIRECTIONS}")
    _check_param("atr_len", sleeve["atr_len"], ATR_LEN)
    _check_typed_block("entry", sleeve["entry"], ENTRY_TYPES)
    filters = sleeve.get("filters", [])
    if not isinstance(filters, list) or len(filters) > max_filters:
        raise SpecError(f"filters must be a list of at most {max_filters}")
    for i, flt in enumerate(filters):
        _check_typed_block(f"filters.{i}", flt, FILTER_TYPES)
    _check_exit(sleeve["exit"])
    _check_cross_constraints(sleeve)
    _check_param_space(sleeve)


def _check_instruments(ids: Any, known: Iterable[str] | None, allowed: Iterable[str] | None) -> None:
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise SpecError("instruments must be a non-empty list of unique ids")
    if known is not None and not set(ids) <= set(known):
        raise SpecError(f"unknown instruments {sorted(set(ids) - set(known))}")
    if allowed is not None and not set(ids) <= set(allowed):
        raise SpecError(f"instruments not allowed in this portfolio {sorted(set(ids) - set(allowed))}")


def param_bounds(sleeve: dict, path: str) -> Param:
    """Grammar Param describing the numeric value at a dotted path of the sleeve."""
    parts = path.split(".")
    if path == "atr_len":
        return ATR_LEN
    if parts[0] == "entry" and len(parts) == 2:
        return _lookup(ENTRY_TYPES, sleeve["entry"], parts[1], path)
    if parts[0] == "filters" and len(parts) == 3 and parts[1].isdigit() and int(parts[1]) < len(sleeve.get("filters", [])):
        return _lookup(FILTER_TYPES, sleeve["filters"][int(parts[1])], parts[2], path)
    if parts[0] == "exit" and len(parts) == 2 and parts[1] in EXIT_PARAMS and sleeve["exit"].get(parts[1]) is not None:
        return EXIT_PARAMS[parts[1]]
    if parts[0] == "exit" and len(parts) == 3 and parts[1] == "exit_signal" and sleeve["exit"].get("exit_signal"):
        return _lookup(EXIT_SIGNAL_TYPES, sleeve["exit"]["exit_signal"], parts[2], path)
    raise SpecError(f"param_space path {path!r} does not point at a tunable number")


def _lookup(table: dict, block: dict, key: str, path: str) -> Param:
    p = table[block["type"]].get(key)
    if p is None or p.kind == "choice":
        raise SpecError(f"param_space path {path!r} is not a tunable number")
    return p


def _check_param_space(sleeve: dict) -> None:
    space = sleeve.get("param_space", {})
    if not isinstance(space, dict) or len(space) > MAX_PARAM_SPACE:
        raise SpecError(f"param_space must be an object with at most {MAX_PARAM_SPACE} entries")
    for path, rng in space.items():
        p = param_bounds(sleeve, path)
        if not isinstance(rng, list) or len(rng) != 3 or not all(_num(v) for v in rng):
            raise SpecError(f"param_space[{path}] must be [low, high, step]")
        low, high, step = rng
        if step <= 0 or low > high:
            raise SpecError(f"param_space[{path}] needs low <= high and step > 0")
        if not p.low <= low <= high <= p.high:
            raise SpecError(f"param_space[{path}] outside [{p.low}, {p.high}]")
        if p.kind == "int" and not all(float(v) == int(v) for v in rng):
            raise SpecError(f"param_space[{path}] must be integers")


def validate_portfolio(pf: Any, known: Iterable[str] | None = None, allowed: Iterable[str] | None = None,
                       max_sleeves: int = 3, max_filters: int = 4) -> None:
    if not isinstance(pf, dict) or not isinstance(pf.get("sleeves"), list) or not pf["sleeves"]:
        raise SpecError("portfolio needs a non-empty sleeves list")
    if len(pf["sleeves"]) > max_sleeves:
        raise SpecError(f"at most {max_sleeves} sleeves")
    ids = [s.get("id") if isinstance(s, dict) else None for s in pf["sleeves"]]
    if len(set(ids)) != len(ids):
        raise SpecError("sleeve ids must be unique")
    for sleeve in pf["sleeves"]:
        validate_sleeve(sleeve, known, allowed, max_filters)


def get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    last = parts[-1]
    if isinstance(obj, list):
        obj[int(last)] = value
    else:
        obj[last] = value


def grid_values(rng: list) -> list:
    """Values low, low+step, ... up to high. Ints when low, high and step are all ints."""
    low, high, step = rng
    as_int = all(float(v) == int(v) for v in rng)
    n = int(round((high - low) / step + 1e-9))
    vals = [low + k * step for k in range(n + 1) if low + k * step <= high + 1e-9]
    return [int(v) for v in vals] if as_int else [round(float(v), 10) for v in vals]


def default_params(sleeve: dict) -> dict[str, Any]:
    """Current value of every path in the sleeve's param_space."""
    return {path: get_path(sleeve, path) for path in sleeve.get("param_space", {})}


def apply_params(sleeve: dict, params: dict[str, Any]) -> dict:
    """Copy of the sleeve with param values written at their dotted paths."""
    out = copy.deepcopy(sleeve)
    for path, value in params.items():
        p = param_bounds(out, path)
        set_path(out, path, int(round(value)) if p.kind == "int" else float(value))
    return out


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def spec_hash(obj: Any) -> str:
    """12 hex chars of the sha1 of the canonical JSON."""
    return hashlib.sha1(canonical_json(obj).encode()).hexdigest()[:12]


def portfolio_hash(pf: dict) -> str:
    """Hash of the strategy structure: normalised sleeves sorted by id. Risk and scores are excluded."""
    sleeves = sorted((normalise(s) for s in pf["sleeves"]), key=lambda s: s["id"])
    return spec_hash(sleeves)


def sleeve_hash(sleeve: dict) -> str:
    return spec_hash(normalise(sleeve))


def describe_grammar() -> str:
    """Compact grammar reference placed in the LLM prompt. Generated from the same tables as validation."""
    def block(table: dict) -> list[str]:
        rows = []
        for name, params in table.items():
            args = ", ".join(_param_doc(k, p) for k, p in params.items())
            rows.append(f'  "{name}": {{{args}}}')
        return rows

    lines = ["sleeve = {id, instruments[], direction: both|long|short, entry, filters[<=4], exit, atr_len: int 2..400,",
             "          param_space: {dotted.path: [low, high, step]}}", "entry types (put \"type\" plus these keys):"]
    lines += block(ENTRY_TYPES)
    lines.append("filter types (all must pass; they gate entries only):")
    lines += block(FILTER_TYPES)
    lines.append("exit = {stop_atr: 0.5..10 (required), target_atr: 0.5..10|null, trail_atr: 0.5..10|null, "
                 "time_stop_bars: 1..400|null, exit_signal: null|one of the types below, reverse_on_opposite: bool}")
    lines += block(EXIT_SIGNAL_TYPES)
    lines.append("param_space paths: entry.<key>, filters.<index>.<key>, exit.stop_atr, exit.target_atr, exit.trail_atr, "
                 "exit.time_stop_bars, exit.exit_signal.<key>, atr_len (only numeric keys that exist in the sleeve).")
    return "\n".join(lines)


def _param_doc(name: str, p: Param) -> str:
    if p.kind == "choice":
        return f'{name}: {"|".join(p.choices)}'
    return f"{name}: {p.kind} {p.low}..{p.high}"
