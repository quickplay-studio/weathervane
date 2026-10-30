"""Indicator semantics against hand-checked arrays and loop references written from the Pine definitions."""
import numpy as np
import pytest

from weathervane import indicators as ind

nan = np.nan


def arr(*v):
    return np.array(v, dtype=float)


def assert_same(actual, expected, tol=1e-9):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    assert actual.shape == expected.shape
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), (actual, expected)
    m = ~np.isnan(expected)
    assert np.allclose(actual[m], expected[m], atol=tol, rtol=0)


def test_sma_needs_full_window():
    assert_same(ind.sma(arr(1, 2, 3, 4, 5), 3), [nan, nan, 2, 3, 4])


def test_sma_window_with_nan_is_nan():
    assert_same(ind.sma(arr(nan, 2, 3, 4), 2), [nan, nan, 2.5, 3.5])


def test_ema_seeds_with_sma_of_first_n():
    # alpha 0.5, seed 4 at index 2, then 0.5*8+0.5*4 = 6, 6.5, 5.75
    assert_same(ind.ema(arr(2, 4, 6, 8, 7, 5), 3), [nan, nan, 4, 6, 6.5, 5.75])


def test_rma_is_wilder_with_sma_seed():
    out = ind.rma(arr(2, 4, 6, 8, 7, 5), 3)
    third = 16 / 3
    exp = [nan, nan, 4, third, (7 + 2 * third) / 3, None]
    exp[5] = (5 + 2 * exp[4]) / 3
    assert_same(out, exp)


def test_ema_with_leading_nan_seeds_at_first_full_window():
    out = ind.ema(arr(nan, 2, 4, 6, 8), 2)
    assert_same(out, [nan, nan, 3, 5, 7])


def test_rsi_small_hand_checked():
    # changes: +1, -1, +1; rma(2): up 0.5 then 0.75, down 0.5 then 0.25
    assert_same(ind.rsi(arr(1, 2, 1, 2), 2), [nan, nan, 50, 75])


def test_rsi_all_gains_is_100():
    out = ind.rsi(np.arange(1.0, 12.0), 5)
    assert_same(out[5:], np.full(6, 100.0))


def test_atr_first_tr_is_high_low_then_wilder():
    high, low, close = arr(10, 13, 12), arr(8, 12, 10), arr(9, 12.5, 11)
    # tr = 2, 4, 2.5 ; rma(2) seed (2+4)/2 = 3 ; then (2.5 + 3)/2
    assert_same(ind.atr(high, low, close, 2), [nan, 3, 2.75])


def test_true_range_variants():
    high, low, close = arr(10, 13), arr(8, 12), arr(9, 12.5)
    assert_same(ind.true_range(high, low, close, True), [2, 4])
    assert_same(ind.true_range(high, low, close, False), [nan, 4])


def test_highest_lowest_include_current_bar():
    s = arr(3, 1, 4, 1, 5)
    assert_same(ind.highest(s, 3), [nan, nan, 4, 4, 5])
    assert_same(ind.lowest(s, 3), [nan, nan, 1, 1, 1])


def test_roc_percent_change_over_n():
    assert_same(ind.roc(arr(100, 110, 99), 1), [nan, 10, -10])


def test_stdev_is_population():
    out = ind.stdev(arr(2, 4, 4, 4, 5, 5, 7, 9), 8)
    assert out[-1] == pytest.approx(2.0)
    assert_same(ind.stdev(arr(1, 2, 3, 4), 2), [nan, 0.5, 0.5, 0.5])


def test_percentrank_counts_previous_values_le_current():
    out = ind.percentrank(arr(1, 3, 2, 5, 4), 3)
    assert_same(out, [nan, nan, nan, 100.0, 200.0 / 3])


def test_bollinger_uses_sma_and_population_stdev():
    basis, up, lo = ind.bollinger(arr(1, 2, 3, 4), 2, 2.0)
    assert_same(basis, [nan, 1.5, 2.5, 3.5])
    assert_same(up, [nan, 2.5, 3.5, 4.5])
    assert_same(lo, [nan, 0.5, 1.5, 2.5])


def test_keltner_uses_ema_and_atr_of_same_length():
    high, low, close = arr(10, 13, 12), arr(8, 12, 10), arr(9, 12.5, 11)
    mid, up, lo = ind.keltner(high, low, close, 2, 1.5)
    atr = ind.atr(high, low, close, 2)
    assert_same(mid, ind.ema(close, 2))
    assert_same(up, mid + 1.5 * atr)
    assert_same(lo, mid - 1.5 * atr)


def test_crossover_and_crossunder():
    assert_same(ind.crossover(arr(1, 2, 3), arr(2, 2, 2)).astype(float), [0, 0, 1])
    assert_same(ind.crossover(arr(1, 3, 4), 2.0).astype(float), [0, 1, 0])
    assert_same(ind.crossunder(arr(3, 2, 1), arr(2, 2, 2)).astype(float), [0, 0, 1])
    assert_same(ind.crossunder(arr(3, 1, 0), 2.0).astype(float), [0, 1, 0])


def test_crossover_ignores_nan():
    assert not ind.crossover(arr(nan, 3), 2.0).any()


def test_dmi_trending_series_hand_checked():
    high, low, close = arr(1, 2, 3, 4, 5), arr(0, 1, 2, 3, 4), arr(0.5, 1.5, 2.5, 3.5, 4.5)
    plus, minus, adx = ind.dmi(high, low, close, 2)
    assert_same(plus, [nan, nan, 100 / 1.5, 100 / 1.5, 100 / 1.5])
    assert_same(minus, [nan, nan, 0, 0, 0])
    assert_same(adx, [nan, nan, nan, 100, 100])


def _dmi_reference(high, low, close, n):
    """Direct transcription of the Pine ta.dmi source using python loops."""
    size = len(high)
    tr = [nan] + [max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
                  for i in range(1, size)]
    pdm, mdm = [nan], [nan]
    for i in range(1, size):
        up, down = high[i] - high[i - 1], -(low[i] - low[i - 1])
        pdm.append(up if (up > down and up > 0) else 0.0)
        mdm.append(down if (down > up and down > 0) else 0.0)

    def rma(x):
        out, prev = [], nan
        for i, v in enumerate(x):
            if np.isnan(prev):
                w = x[max(0, i - n + 1): i + 1]
                cur = float(np.mean(w)) if i >= n - 1 and not np.isnan(w).any() else nan
            else:
                cur = v / n + (1 - 1 / n) * prev
            out.append(cur)
            prev = cur
        return np.array(out)

    t, p, m = rma(tr), rma(pdm), rma(mdm)
    plus, minus = 100 * p / t, 100 * m / t
    s = plus + minus
    return plus, minus, 100 * rma(np.abs(plus - minus) / np.where(s == 0, 1, s))


def test_dmi_matches_reference_on_random_walk():
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    high = close + rng.uniform(0.1, 1.5, 300)
    low = close - rng.uniform(0.1, 1.5, 300)
    plus, minus, adx = ind.dmi(high, low, close, 14)
    rp, rm, ra = _dmi_reference(high, low, close, 14)
    assert_same(plus, rp, 1e-8)
    assert_same(minus, rm, 1e-8)
    assert_same(adx, ra, 1e-8)


def test_numba_and_python_recursions_agree():
    rng = np.random.default_rng(5)
    x = rng.normal(size=500).cumsum()
    py = getattr(ind._seeded_ma, "py_func", ind._seeded_ma)
    assert_same(ind.ema(x, 20), py(x, 20, 2 / 21))
