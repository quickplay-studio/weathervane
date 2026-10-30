"""Optuna TPE search over the shared parameters of a candidate's sleeves."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import optuna

from . import spec

optuna.logging.set_verbosity(optuna.logging.ERROR)


class BudgetExceeded(Exception):
    """Raised when the caller's time budget is used up in the middle of an evaluation."""


@dataclass(frozen=True)
class Dim:
    sleeve: str
    path: str
    values: tuple
    default_idx: int

    @property
    def name(self) -> str:
        return f"{self.sleeve}:{self.path}"


@dataclass
class SearchResult:
    params: dict[str, dict]     # sleeve id -> {dotted path: value}
    value: float
    trials: int


def dims_for(sleeves: list[dict]) -> list[Dim]:
    out = []
    for sleeve in sleeves:
        for path, rng in sleeve.get("param_space", {}).items():
            values = tuple(spec.grid_values(rng))
            current = spec.get_path(sleeve, path)
            nearest = min(range(len(values)), key=lambda k: abs(values[k] - current))
            out.append(Dim(sleeve["id"], path, values, nearest))
    return out


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise BudgetExceeded()


def optimise(sleeves: list[dict], objective: Callable[[dict[str, dict]], float], n_trials: int, seed: int,
             deadline: float | None = None) -> SearchResult:
    """Maximise objective(params) over every sleeve's param_space. The sleeves' current values are tried first."""
    dims = dims_for(sleeves)
    base = {s["id"]: spec.default_params(s) for s in sleeves}
    if not dims:
        return SearchResult(base, objective(base), 1)
    memo: dict[tuple, float] = {}

    def params_at(idx: dict[str, int]) -> dict[str, dict]:
        params: dict[str, dict] = {s["id"]: {} for s in sleeves}
        for d in dims:
            params[d.sleeve][d.path] = d.values[idx[d.name]]
        return params

    def run(trial: optuna.Trial) -> float:
        check_deadline(deadline)
        idx = {d.name: trial.suggest_int(d.name, 0, len(d.values) - 1) for d in dims}
        key = tuple(sorted(idx.items()))
        if key not in memo:
            memo[key] = objective(params_at(idx))
        return memo[key]

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(3, n_trials // 3))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({d.name: d.default_idx for d in dims})
    study.optimize(run, n_trials=n_trials)
    best = {d.name: study.best_trial.params[d.name] for d in dims}
    return SearchResult(params_at(best), float(study.best_value), len(study.trials))
