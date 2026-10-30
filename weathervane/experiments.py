"""Append-only experiment log, leaderboard of challengers, and the per-instrument asset scan."""
from __future__ import annotations

import csv

import pandas as pd

from . import config

COLUMNS = ["ts", "portfolio", "exp_id", "parent_hash", "source", "change_description", "oos_pass", "oos_trades", "dsr",
           "regime_ok", "maxdd", "risk", "decision", "runtime_s", "spec_hash", "oos_sharpe"]
COMPACT = ["exp_id", "change_description", "oos_pass", "oos_trades", "dsr", "maxdd", "decision"]


def experiments_path():
    return config.path("results", "experiments.csv")


def append_experiment(row: dict) -> None:
    """Append one row, writing the header first when the file is new."""
    path = experiments_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    clean = {k: (str(row.get(k, "")).replace("\n", " ") if isinstance(row.get(k), str) else row.get(k, "")) for k in COLUMNS}
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(clean)


def read_experiments(portfolio: str | None = None) -> pd.DataFrame:
    path = experiments_path()
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"spec_hash": str, "parent_hash": str})
    return df if portfolio is None else df[df["portfolio"] == portfolio].reset_index(drop=True)


def known_hashes(portfolio: str) -> set[str]:
    return set(read_experiments(portfolio)["spec_hash"].dropna().astype(str))


def prior_sharpes(portfolio: str) -> list[float]:
    return read_experiments(portfolio)["oos_sharpe"].dropna().astype(float).tolist()


def recent_rows(portfolio: str, n: int = 25) -> list[dict]:
    """The last n experiments in compact form for the LLM prompt."""
    df = read_experiments(portfolio).tail(n)
    out = []
    for rec in df[COMPACT].to_dict("records"):
        out.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
    return out


def leaderboard_path():
    return config.path("results", "leaderboard.json")


def load_leaderboard() -> dict:
    return config.read_json(leaderboard_path(), {})


def board_for(board: dict, portfolio: str) -> dict:
    return board.setdefault(portfolio, {"entries": [], "vault_checks": {}})


def add_to_leaderboard(portfolio: str, entry: dict, size: int) -> None:
    """Keep the best `size` challengers by OOS pass probability (ties by deflated Sharpe)."""
    board = load_leaderboard()
    b = board_for(board, portfolio)
    b["entries"].append(entry)
    b["entries"].sort(key=lambda e: (e["oos_pass"], e["dsr"]), reverse=True)
    b["entries"] = b["entries"][:size]
    config.write_json(leaderboard_path(), board)


def asset_scan(oos_trades: pd.DataFrame) -> dict[str, dict]:
    """OOS trades per instrument: count, total and mean R, win rate, share of the portfolio's R."""
    if oos_trades.empty:
        return {}
    total = float(oos_trades["r"].sum())
    out = {}
    for inst, g in oos_trades.groupby("instrument"):
        out[str(inst)] = {"trades": int(len(g)), "total_r": round(float(g["r"].sum()), 3),
                          "expectancy_r": round(float(g["r"].mean()), 4), "win_rate": round(float((g["r"] > 0).mean()), 3),
                          "share_of_r": round(float(g["r"].sum() / total), 3) if total else 0.0}
    return out


def save_asset_scan(portfolio: str, data_date: str, table: dict) -> None:
    path = config.path("results", "asset_scan.json")
    scans = config.read_json(path, {})
    scans[portfolio] = {"data_date": data_date, "instruments": table}
    config.write_json(path, scans)


def load_asset_scan(portfolio: str) -> dict:
    return config.read_json(config.path("results", "asset_scan.json"), {}).get(portfolio, {})
