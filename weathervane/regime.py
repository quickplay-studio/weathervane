"""Market regimes from a reference series: 3-state Gaussian HMM, with a rule-based fallback.

Regimes label trades for the OOS gate and shade charts. They are never used inside trading rules (not Pine-portable).
The HMM is fitted on training data only. Labels for later dates come from decoding the full series with that model.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("weathervane.regime")
LABELS = ("bull", "bear", "chop")
VOL_WINDOW = 20
TREND_WINDOW = 200


def features(ref: pd.DataFrame) -> pd.DataFrame:
    """Daily log return and 20-day realised volatility."""
    ret = np.log(ref["close"]).diff()
    return pd.DataFrame({"ret": ret, "vol": ret.rolling(VOL_WINDOW).std()}, index=ref.index)


@dataclass
class RegimeModel:
    kind: str                       # "hmm" or "rules"
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    hmm: object | None = None
    state_labels: dict | None = None
    vol_cut: tuple[float, float] | None = None

    def label(self, ref: pd.DataFrame) -> pd.Series:
        """bull / bear / chop for every date of the reference frame (warm-up dates are chop)."""
        feats = features(ref)
        labels = pd.Series("chop", index=ref.index, dtype=object)
        if self.kind == "hmm":
            ok = feats.notna().all(axis=1).to_numpy()
            if ok.sum() > 0:
                z = (feats.to_numpy()[ok] - self.mean) / self.std
                states = self.hmm.predict(z)
                labels.iloc[np.flatnonzero(ok)] = [self.state_labels[int(s)] for s in states]
            return labels
        return rule_labels(ref, feats, self.vol_cut)


def rule_labels(ref: pd.DataFrame, feats: pd.DataFrame, vol_cut: tuple[float, float]) -> pd.Series:
    """200d trend sign x volatility tercile: uptrend with calm vol is bull, downtrend is bear, the rest is chop."""
    sma = ref["close"].rolling(TREND_WINDOW).mean()
    up, down = ref["close"] > sma, ref["close"] < sma
    high_vol = feats["vol"] > vol_cut[1]
    labels = pd.Series("chop", index=ref.index, dtype=object)
    labels[up & ~high_vol] = "bull"
    labels[down] = "bear"
    return labels


def fit(ref: pd.DataFrame, train_end: pd.Timestamp | None = None, seed: int = 0) -> RegimeModel:
    """Fit on rows before train_end (all rows if None). Falls back to rules when hmmlearn is missing or fails."""
    feats = features(ref)
    train = feats[feats.index < train_end] if train_end is not None else feats
    train = train.dropna()
    vol_cut = (float(train["vol"].quantile(1 / 3)), float(train["vol"].quantile(2 / 3))) if len(train) else (0.0, 0.0)
    if len(train) < 250:
        return RegimeModel("rules", vol_cut=vol_cut)
    try:
        from hmmlearn.hmm import GaussianHMM
        mean, std = train.to_numpy().mean(axis=0), train.to_numpy().std(axis=0)
        z = (train.to_numpy() - mean) / std
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=seed).fit(z)
        order = np.argsort(hmm.means_[:, 0])           # by mean standardised return: bear < chop < bull
        names = {int(order[0]): "bear", int(order[1]): "chop", int(order[2]): "bull"}
        return RegimeModel("hmm", mean, std, hmm, names, vol_cut)
    except Exception as exc:  # hmmlearn missing, or the fit failed to converge
        log.warning("HMM regimes unavailable (%s: %s), using rule-based labels", type(exc).__name__, str(exc)[:80])
        return RegimeModel("rules", vol_cut=vol_cut)


def spans(labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Consecutive runs of one label as (first date, last date, label)."""
    if labels.empty:
        return []
    change = labels.ne(labels.shift()).cumsum()
    return [(g.index[0], g.index[-1], g.iloc[0]) for _, g in labels.groupby(change)]


def label_trades(trades: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Regime on each trade's entry date (the last reference date on or before it; chop before the series starts)."""
    if trades.empty:
        return pd.Series(dtype=object)
    pos = labels.index.searchsorted(pd.DatetimeIndex(trades["entry_date"]), side="right") - 1
    values = np.where(pos >= 0, labels.to_numpy()[np.clip(pos, 0, None)], "chop")
    return pd.Series(values, index=trades.index)
