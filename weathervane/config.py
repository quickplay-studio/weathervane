"""Paths, engine settings, universe, exam rules, portfolios and champion files."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MAX_LOSS_TYPES = ("static", "eod_trailing", "intraday_trailing")


def root() -> Path:
    """Repo root. WV_ROOT overrides it (tests and selftest use a temp workspace)."""
    env = os.environ.get("WV_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[1]


def path(*parts: str) -> Path:
    return root().joinpath(*parts)


def read_json(p: Path, default: Any = None) -> Any:
    if not p.exists():
        return default
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_engine(overrides: dict | None = None) -> dict:
    with path("config", "engine.yaml").open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return deep_merge(cfg, overrides) if overrides else cfg


@dataclass(frozen=True)
class Instrument:
    id: str
    name: str
    asset_class: str
    yf: str
    oanda: str | None
    tv: str
    alpaca_proxy: str | None
    ftmo: bool
    cost_bps: float
    contract_note: str

    @property
    def half_cost(self) -> float:
        """Half of the round-trip cost as a price fraction, charged on every fill."""
        return self.cost_bps / 2.0 / 1e4


def load_universe() -> dict[str, Instrument]:
    with path("config", "universe.yaml").open(encoding="utf-8") as fh:
        rows = yaml.safe_load(fh)["instruments"]
    return {r["id"]: Instrument(**r) for r in rows}


@dataclass(frozen=True)
class Phase:
    name: str
    target: float
    min_days: int


@dataclass(frozen=True)
class Rules:
    """Exam rules with all percentages converted to account fractions."""

    firm: str
    program: str
    account_size: float
    phases: tuple[Phase, ...]
    daily_loss: float
    max_loss: float
    max_loss_type: str = "static"
    lock_at_start: bool = False
    consistency: float | None = None
    time_limit_days: int | None = None
    max_risk: float | None = None
    risk_guard: float = 0.6

    @classmethod
    def from_dict(cls, d: dict) -> "Rules":
        kind = d.get("max_loss_type", "static")
        if kind not in MAX_LOSS_TYPES:
            raise ValueError(f"max_loss_type must be one of {MAX_LOSS_TYPES}, got {kind!r}")
        phases = tuple(Phase(p["name"], p["profit_target_pct"] / 100.0, int(p.get("min_trading_days", 0)))
                       for p in d["phases"])
        if not phases:
            raise ValueError("rules need at least one phase")
        return cls(
            firm=d.get("firm", ""), program=d.get("program", ""), account_size=float(d["account_size"]),
            phases=phases, daily_loss=d["daily_loss_limit_pct"] / 100.0, max_loss=d["max_loss_pct"] / 100.0,
            max_loss_type=kind, lock_at_start=bool(d.get("trailing_lock_at_start", False)),
            consistency=_pct(d.get("consistency_best_day_pct")),
            time_limit_days=d.get("time_limit_days"), max_risk=_pct(d.get("max_risk_per_trade_pct")),
            risk_guard=float(d.get("risk_guard_pct_of_daily", 60)) / 100.0,
        )


def _pct(v: float | None) -> float | None:
    return None if v is None else v / 100.0


def load_rules(name: str) -> Rules:
    with path("config", "rules", f"{name}.yaml").open(encoding="utf-8") as fh:
        return Rules.from_dict(yaml.safe_load(fh))


@dataclass(frozen=True)
class Portfolio:
    id: str
    rules: str
    instruments: tuple[str, ...]
    brokers: tuple[str, ...]
    regime_reference: str


def load_portfolios() -> dict[str, Portfolio]:
    with path("config", "portfolios.yaml").open(encoding="utf-8") as fh:
        rows = yaml.safe_load(fh)["portfolios"]
    return {r["id"]: Portfolio(r["id"], r["rules"], tuple(r["instruments"]), tuple(r.get("brokers", [])),
                               r["regime_reference"]) for r in rows}


def select_portfolios(pid: str | None) -> list[Portfolio]:
    allp = load_portfolios()
    if pid is None:
        return list(allp.values())
    if pid not in allp:
        raise SystemExit(f"unknown portfolio {pid!r}; choose from {sorted(allp)}")
    return [allp[pid]]


def champion_path(pid: str) -> Path:
    return path("champion", f"{pid}.json")


def load_champion(pid: str) -> dict:
    ch = read_json(champion_path(pid))
    if ch is None:
        raise FileNotFoundError(f"no champion file for portfolio {pid!r}: {champion_path(pid)}")
    return ch


def save_champion(ch: dict) -> None:
    write_json(champion_path(ch["portfolio"]), ch)


def archive_champion(ch: dict, stamp: str) -> Path:
    """Copy a champion into champion/history/ before it is replaced."""
    dest = path("champion", "history", f"{ch['portfolio']}_{stamp}_{ch['id']}.json")
    write_json(dest, ch)
    return dest


def live_risk(champion: dict, cfg: dict, rules: Rules) -> float:
    """Risk fraction used for live sizing: the champion's optimised risk, capped by the rules."""
    risk = champion.get("risk") or cfg["research"]["default_risk"]
    return min(risk, rules.max_risk) if rules.max_risk else risk
