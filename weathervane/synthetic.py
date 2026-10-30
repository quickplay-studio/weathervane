"""Deterministic synthetic OHLC generator with regime switching, used by tests and selftest."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

SYNTHETIC_END = "2026-10-28"
DEFAULT_BARS = 5800
BASE_VOL = {"fx": 0.0055, "index": 0.011, "commodity": 0.015, "crypto": 0.032, "etf": 0.0095}
START_PRICE = {"fx": 1.1, "index": 3000.0, "commodity": 100.0, "crypto": 5000.0, "etf": 100.0}

# regimes: (daily drift in units of base vol, vol multiplier, lag-1 autocorrelation)
REGIMES = {"up": (0.09, 0.8, 0.02), "down": (-0.09, 1.1, 0.02), "chop": (0.0, 0.8, -0.18), "storm": (0.0, 2.0, 0.0)}
REGIME_NAMES = list(REGIMES)
MEAN_DURATION = 70


def _seed(name: str) -> int:
    return zlib.crc32(name.encode())


def regime_path(n: int, rng: np.random.Generator) -> np.ndarray:
    """Markov chain over the four regimes with geometric durations (mean MEAN_DURATION bars)."""
    path = np.empty(n, dtype=np.int8)
    i, state = 0, int(rng.integers(0, 4))
    while i < n:
        length = int(rng.geometric(1.0 / MEAN_DURATION))
        path[i: i + length] = state
        i += length
        state = int(rng.choice([s for s in range(4) if s != state]))
    return path


def _returns(regimes: np.ndarray, base_vol: float, factor: np.ndarray, beta: float,
             rng: np.random.Generator) -> np.ndarray:
    drift = np.array([REGIMES[k][0] for k in REGIME_NAMES])[regimes] * base_vol
    vol = np.array([REGIMES[k][1] for k in REGIME_NAMES])[regimes] * base_vol
    rho = np.array([REGIMES[k][2] for k in REGIME_NAMES])[regimes]
    shocks = rng.normal(0.0, 1.0, regimes.shape[0]) * vol
    ret = np.empty_like(shocks)
    prev = 0.0
    for i in range(shocks.shape[0]):
        ret[i] = drift[i] + rho[i] * prev + shocks[i]
        prev = ret[i]
    return ret + beta * factor * base_vol


def generate(name: str, asset_class: str = "index", n_bars: int = DEFAULT_BARS, end: str = SYNTHETIC_END) -> pd.DataFrame:
    """OHLCV for one instrument. The same (name, asset_class, n_bars, end) always gives the same frame."""
    base = BASE_VOL.get(asset_class, 0.01)
    rng = np.random.default_rng(_seed(name))
    regimes = regime_path(n_bars, rng)
    factor = np.random.default_rng(12345).normal(0.0, 1.0, n_bars)     # shared market factor
    beta = float(rng.uniform(-0.3, 0.6))
    ret = _returns(regimes, base, factor, beta, rng)
    gap = rng.normal(0.0, 0.25 * base, n_bars)
    close = START_PRICE.get(asset_class, 100.0) * np.exp(np.cumsum(ret))
    prev_close = np.r_[close[0], close[:-1]]
    open_ = prev_close * np.exp(gap)
    spread = np.abs(rng.normal(0.0, 0.5 * base, (2, n_bars)))
    high = np.maximum(open_, close) * np.exp(spread[0])
    low = np.minimum(open_, close) * np.exp(-spread[1])
    volume = np.exp(rng.normal(12.0, 0.4, n_bars)).round()
    idx = pd.bdate_range(end=end, periods=n_bars, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def load_synthetic(universe: dict, ids: list[str], n_bars: int = DEFAULT_BARS) -> dict[str, pd.DataFrame]:
    """Synthetic frames for the requested instrument ids (class volatility comes from the universe)."""
    return {i: generate(i, universe[i].asset_class, n_bars) for i in ids if i in universe}
