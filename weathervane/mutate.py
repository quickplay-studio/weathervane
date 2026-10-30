"""Random single-change mutations of a candidate portfolio. Used when no LLM is available or its reply is unusable."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from . import spec


@dataclass
class Context:
    allowed: list[str]
    max_sleeves: int = 3
    max_filters: int = 4
    weak: list[str] = field(default_factory=list)       # instruments with negative OOS expectancy in the champion


def default_block(table: dict, kind: str) -> dict:
    return {"type": kind, **{k: p.default for k, p in table[kind].items()}}


def numeric_space(table: dict, kind: str, prefix: str, limit: int = 2) -> dict[str, list]:
    """Default search ranges for the first `limit` numeric parameters of a block type."""
    keys = [k for k, p in table[kind].items() if p.kind != "choice"][:limit]
    return {f"{prefix}.{k}": list(table[kind][k].space) for k in keys}


def tunable_paths(sleeve: dict) -> list[str]:
    paths = ["atr_len"]
    paths += [f"entry.{k}" for k, p in spec.ENTRY_TYPES[sleeve["entry"]["type"]].items() if p.kind != "choice"]
    for i, flt in enumerate(sleeve["filters"]):
        paths += [f"filters.{i}.{k}" for k, p in spec.FILTER_TYPES[flt["type"]].items() if p.kind != "choice"]
    ex = sleeve["exit"]
    paths += [f"exit.{k}" for k in spec.EXIT_PARAMS if ex.get(k) is not None]
    if ex.get("exit_signal"):
        paths += [f"exit.exit_signal.{k}" for k, p in spec.EXIT_SIGNAL_TYPES[ex["exit_signal"]["type"]].items() if p.kind != "choice"]
    return paths


def drop_space(sleeve: dict, prefix: str) -> None:
    sleeve["param_space"] = {k: v for k, v in sleeve["param_space"].items() if not k.startswith(prefix)}


def include_in_space(sleeve: dict, path: str) -> None:
    """Widen the path's search range, if it has one, so that it holds the sleeve's current value."""
    rng = sleeve["param_space"].get(path)
    if rng:
        value = spec.get_path(sleeve, path)
        sleeve["param_space"][path] = [min(rng[0], value), max(rng[1], value), rng[2]]


def reindex_filters(sleeve: dict, removed: int) -> None:
    """Drop the removed filter's ranges and shift the later filters' ranges down by one."""
    out = {}
    for path, rng in sleeve["param_space"].items():
        parts = path.split(".")
        if parts[0] == "filters":
            idx = int(parts[1])
            if idx == removed:
                continue
            if idx > removed:
                path = ".".join(["filters", str(idx - 1), *parts[2:]])
        out[path] = rng
    sleeve["param_space"] = out


def _add_space(sleeve: dict, ranges: dict[str, list]) -> None:
    for path, rng in ranges.items():
        if len(sleeve["param_space"]) < spec.MAX_PARAM_SPACE:
            sleeve["param_space"][path] = rng


# each operator edits the portfolio in place and returns a description, or None when it does not apply

def tweak_range(pf: dict, ctx: Context, rng: random.Random) -> str | None:
    sleeve = rng.choice(pf["sleeves"])
    space = sleeve["param_space"]
    free = [p for p in tunable_paths(sleeve) if p not in space]
    modes = (["widen", "narrow", "shift", "drop"] if space else []) + (["add"] if free and len(space) < spec.MAX_PARAM_SPACE else [])
    if not modes:
        return None
    mode = rng.choice(modes)
    if mode == "add":
        path = rng.choice(free)
        bounds = spec.param_bounds(sleeve, path)
        value = spec.get_path(sleeve, path)
        low, high, step = bounds.space or (bounds.low, bounds.high, 1)
        space[path] = [min(low, value), max(high, value), step]
        return f"{sleeve['id']}: tune {path} over {space[path]}"
    path = rng.choice(list(space))
    if mode == "drop":
        del space[path]
        return f"{sleeve['id']}: stop tuning {path}"
    return _reshape_range(sleeve, path, mode, rng)


def _reshape_range(sleeve: dict, path: str, mode: str, rng: random.Random) -> str | None:
    bounds = spec.param_bounds(sleeve, path)
    old = list(sleeve["param_space"][path])
    vals, step = spec.grid_values(old), old[2]
    cur = spec.get_path(sleeve, path)
    k = max(1, round(0.25 * len(vals)))
    if mode == "widen":
        new_lo, new_hi = vals[0], vals[-1]
        for _ in range(k):
            new_lo = new_lo - step if new_lo - step >= bounds.low - 1e-9 else new_lo
            new_hi = new_hi + step if new_hi + step <= bounds.high + 1e-9 else new_hi
    elif mode == "narrow":
        if len(vals) < 5:
            return None
        cur_i = min(range(len(vals)), key=lambda j: abs(vals[j] - cur))
        new_lo, new_hi = vals[min(k, cur_i)], vals[max(len(vals) - 1 - k, cur_i)]
    else:
        shift = rng.choice([-k, k])
        new_lo, new_hi = vals[0] + shift * step, vals[-1] + shift * step
        if new_lo < bounds.low - 1e-9 or new_hi > bounds.high + 1e-9 or not new_lo - 1e-9 <= cur <= new_hi + 1e-9:
            return None
    fix = (lambda v: int(round(v))) if bounds.kind == "int" else (lambda v: round(float(v), 6))
    new = [fix(new_lo), fix(new_hi), step]
    if new == old:
        return None
    sleeve["param_space"][path] = new
    return f"{sleeve['id']}: {mode} range of {path} to {new}"


def swap_entry(pf: dict, ctx: Context, rng: random.Random) -> str | None:
    sleeve = rng.choice(pf["sleeves"])
    old = sleeve["entry"]["type"]
    new = rng.choice([t for t in spec.ENTRY_TYPES if t != old])
    sleeve["entry"] = default_block(spec.ENTRY_TYPES, new)
    drop_space(sleeve, "entry.")
    _add_space(sleeve, numeric_space(spec.ENTRY_TYPES, new, "entry"))
    return f"{sleeve['id']}: entry {old} -> {new}"


def add_filter(pf: dict, ctx: Context, rng: random.Random) -> str | None:
    sleeve = rng.choice(pf["sleeves"])
    if len(sleeve["filters"]) >= ctx.max_filters:
        return None
    used = {f["type"] for f in sleeve["filters"]}
    kind = rng.choice([t for t in spec.FILTER_TYPES if t not in used] or list(spec.FILTER_TYPES))
    sleeve["filters"].append(default_block(spec.FILTER_TYPES, kind))
    _add_space(sleeve, numeric_space(spec.FILTER_TYPES, kind, f"filters.{len(sleeve['filters']) - 1}", 1))
    return f"{sleeve['id']}: add filter {kind}"


def remove_filter(pf: dict, ctx: Context, rng: random.Random) -> str | None:
    sleeve = rng.choice([s for s in pf["sleeves"] if s["filters"]] or [None])
    if sleeve is None:
        return None
    idx = rng.randrange(len(sleeve["filters"]))
    kind = sleeve["filters"].pop(idx)["type"]
    reindex_filters(sleeve, idx)
    return f"{sleeve['id']}: remove filter {kind}"


def replace_filter(pf: dict, ctx: Context, rng: random.Random) -> str | None:
    sleeve = rng.choice([s for s in pf["sleeves"] if s["filters"]] or [None])
    if sleeve is None:
        return None
    idx = rng.randrange(len(sleeve["filters"]))
    old = sleeve["filters"][idx]["type"]
    new = rng.choice([t for t in spec.FILTER_TYPES if t != old])
    sleeve["filters"][idx] = default_block(spec.FILTER_TYPES, new)
    drop_space(sleeve, f"filters.{idx}.")
    _add_space(sleeve, numeric_space(spec.FILTER_TYPES, new, f"filters.{idx}", 1))
    return f"{sleeve['id']}: replace filter {old} with {new}"


def _scale_stop(sleeve: dict, factor: float) -> str:
    bounds = spec.EXIT_PARAMS["stop_atr"]
    value = round(min(bounds.high, max(bounds.low, sleeve["exit"]["stop_atr"] * factor)) * 4) / 4
    sleeve["exit"]["stop_atr"] = value
    include_in_space(sleeve, "exit.stop_atr")
    return f"{sleeve['id']}: stop_atr to {value}"


def _toggle_exit(sleeve: dict, key: str) -> str:
    ex = sleeve["exit"]
    if ex.get(key) is None:
        ex[key] = spec.EXIT_PARAMS[key].default
        return f"{sleeve['id']}: enable {key} at {ex[key]}"
    ex[key] = None
    drop_space(sleeve, f"exit.{key}")
    return f"{sleeve['id']}: disable {key}"


def _toggle_exit_signal(sleeve: dict, rng: random.Random) -> str:
    ex = sleeve["exit"]
    drop_space(sleeve, "exit.exit_signal.")
    if ex.get("exit_signal") is None or rng.random() < 0.5:
        kind = rng.choice(list(spec.EXIT_SIGNAL_TYPES))
        ex["exit_signal"] = default_block(spec.EXIT_SIGNAL_TYPES, kind)
        _add_space(sleeve, numeric_space(spec.EXIT_SIGNAL_TYPES, kind, "exit.exit_signal", 1))
        return f"{sleeve['id']}: exit signal {kind}"
    ex["exit_signal"] = None
    return f"{sleeve['id']}: remove exit signal"


def change_exit(pf: dict, ctx: Context, rng: random.Random) -> str | None:
    sleeve = rng.choice(pf["sleeves"])
    choice = rng.choice(["stop_up", "stop_down", "target", "trail", "time", "signal", "reverse"])
    if choice == "stop_up":
        return _scale_stop(sleeve, 1.25)
    if choice == "stop_down":
        return _scale_stop(sleeve, 0.8)
    if choice in ("target", "trail"):
        return _toggle_exit(sleeve, f"{choice}_atr")
    if choice == "time":
        return _toggle_exit(sleeve, "time_stop_bars")
    if choice == "signal":
        return _toggle_exit_signal(sleeve, rng)
    sleeve["exit"]["reverse_on_opposite"] = not sleeve["exit"]["reverse_on_opposite"]
    return f"{sleeve['id']}: reverse_on_opposite to {sleeve['exit']['reverse_on_opposite']}"


def change_direction(pf: dict, ctx: Context, rng: random.Random) -> str | None:
    sleeve = rng.choice(pf["sleeves"])
    new = rng.choice([d for d in spec.DIRECTIONS if d != sleeve["direction"]])
    old, sleeve["direction"] = sleeve["direction"], new
    return f"{sleeve['id']}: direction {old} -> {new}"


def add_instrument(pf: dict, ctx: Context, rng: random.Random) -> str | None:
    sleeve = rng.choice(pf["sleeves"])
    free = [i for i in ctx.allowed if i not in sleeve["instruments"]]
    if not free:
        return None
    inst = rng.choice(free)
    sleeve["instruments"].append(inst)
    return f"{sleeve['id']}: add instrument {inst}"


def remove_instrument(pf: dict, ctx: Context, rng: random.Random) -> str | None:
    sleeve = rng.choice([s for s in pf["sleeves"] if len(s["instruments"]) > 1] or [None])
    if sleeve is None:
        return None
    weak = [i for i in sleeve["instruments"] if i in ctx.weak]
    inst = rng.choice(weak if weak and rng.random() < 0.7 else sleeve["instruments"])
    sleeve["instruments"].remove(inst)
    return f"{sleeve['id']}: remove instrument {inst}"


def add_sleeve(pf: dict, ctx: Context, rng: random.Random) -> str | None:
    if len(pf["sleeves"]) >= ctx.max_sleeves:
        return None
    kind = rng.choice(list(spec.ENTRY_TYPES))
    taken = {s["id"] for s in pf["sleeves"]}
    sid = next(f"{kind[:12]}_{n}" for n in range(1, 99) if f"{kind[:12]}_{n}" not in taken)
    n_inst = min(len(ctx.allowed), rng.randint(3, 8))
    sleeve = {"id": sid, "instruments": sorted(rng.sample(ctx.allowed, n_inst)), "direction": "both",
              "entry": default_block(spec.ENTRY_TYPES, kind), "filters": [],
              "exit": {"stop_atr": 2.5, "target_atr": None, "trail_atr": 3.0, "time_stop_bars": None,
                       "exit_signal": None, "reverse_on_opposite": True},
              "atr_len": 20, "param_space": {}}
    _add_space(sleeve, numeric_space(spec.ENTRY_TYPES, kind, "entry"))
    sleeve["param_space"]["exit.stop_atr"] = list(spec.EXIT_PARAMS["stop_atr"].space)
    pf["sleeves"].append(sleeve)
    return f"add sleeve {sid} ({kind} on {n_inst} instruments)"


def remove_sleeve(pf: dict, ctx: Context, rng: random.Random) -> str | None:
    if len(pf["sleeves"]) < 2:
        return None
    gone = pf["sleeves"].pop(rng.randrange(len(pf["sleeves"])))
    return f"remove sleeve {gone['id']}"


OPERATORS = [(tweak_range, 3), (swap_entry, 1), (add_filter, 2), (remove_filter, 1), (replace_filter, 1),
             (change_exit, 3), (change_direction, 1), (add_instrument, 2), (remove_instrument, 2),
             (add_sleeve, 1), (remove_sleeve, 1)]


def mutate(parent: dict, ctx: Context, rng: random.Random, known: list[str] | None = None, tries: int = 50
           ) -> tuple[dict, str]:
    """A valid portfolio that differs from the parent by exactly one operator, and what the change was."""
    ops, weights = zip(*OPERATORS)
    parent_hash = spec.portfolio_hash(parent)
    for _ in range(tries):
        child = {"sleeves": [spec.normalise(s) for s in copy.deepcopy(parent["sleeves"])]}
        desc = rng.choices(ops, weights)[0](child, ctx, rng)
        if desc is None:
            continue
        try:
            spec.validate_portfolio(child, known, ctx.allowed, ctx.max_sleeves, ctx.max_filters)
        except spec.SpecError:
            continue
        if spec.portfolio_hash(child) != parent_hash:
            return child, desc
    raise RuntimeError("no valid mutation found")
