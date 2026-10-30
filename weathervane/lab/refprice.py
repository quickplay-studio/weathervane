"""Reference pricing in plain numpy/scipy: Black-Scholes-Merton and a CRR tree.

Used by the unit tests as the reference, by the lab as a fallback when QuantLib is not installed, and by adapters that
need a closed form (for example the jump diffusion checks). No third-party finance library is involved.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def bs_price(cp: str, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """European Black-Scholes-Merton price. cp is 'c' or 'p'."""
    if T <= 0 or sigma <= 0:
        fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
        return max(fwd if cp == "c" else -fwd, 0.0)
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd
    d2 = d1 - sd
    if cp == "c":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(cp: str, S: float, K: float, T: float, r: float, q: float, sigma: float) -> dict:
    """delta, gamma, vega (per 1.00 of vol), theta (per year) of a European option."""
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd
    d2 = d1 - sd
    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    pdf = norm.pdf(d1)
    if cp == "c":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * math.sqrt(T)) - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1))
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * math.sqrt(T)) + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1))
    return {"delta": delta, "gamma": disc_q * pdf / (S * sd), "vega": S * disc_q * pdf * math.sqrt(T), "theta": theta}


def implied_vol(cp: str, price: float, S: float, K: float, T: float, r: float, q: float,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Bisection implied volatility; NaN when the price is outside the no-arbitrage band."""
    intrinsic = bs_price(cp, S, K, T, r, q, 1e-8)
    if not (intrinsic - 1e-12 <= price <= bs_price(cp, S, K, T, r, q, hi)):
        return float("nan")
    a, b = lo, hi
    for _ in range(80):
        mid = 0.5 * (a + b)
        if bs_price(cp, S, K, T, r, q, mid) < price:
            a = mid
        else:
            b = mid
        if b - a < 1e-10:
            break
    return 0.5 * (a + b)


def crr_american(cp: str, S: float, K: float, T: float, r: float, q: float, sigma: float, steps: int = 1500) -> float:
    """American option on a Cox-Ross-Rubinstein tree with continuous dividend yield (vectorised backward induction)."""
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1)
    sign = 1.0 if cp == "c" else -1.0
    values = np.maximum(sign * (S * u ** (steps - j) * d ** j - K), 0.0)
    for step in range(steps - 1, -1, -1):
        idx = np.arange(step + 1)
        prices = S * u ** (step - idx) * d ** idx
        values = disc * (p * values[: step + 1] + (1 - p) * values[1: step + 2])
        values = np.maximum(values, sign * (prices - K))
    return float(values[0])


def sabr_vol_beta1(alpha: float, rho: float, nu: float, F: float, K: float, T: float) -> float:
    """Hagan (2002) lognormal SABR implied volatility for beta = 1 (independent of any SABR package)."""
    x = math.log(F / K)
    corr = 1.0 + (rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0) * T
    if abs(nu * x) < 1e-8 * alpha:
        return alpha * corr
    z = nu * x / alpha
    xz = math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    return alpha * (z / xz) * corr
