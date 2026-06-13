"""Rung 4: gradient-boosted model on engineered features — the ML comparison.

The point of this rung is NOT to assume ML wins. It is to engineer sensible
PRE-MATCH features, train a gradient-boosting classifier on the 1X2 outcome, and
backtest it against the Dixon-Coles + Elo ensemble on the SAME split with the SAME
metric (RPS). We keep it only if it beats the baseline.

Leakage is the danger with ML here, so `build_features` walks matches in time
order and records each team's Elo and rolling goal form exactly as they stood
*before* each match — then updates them afterwards.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .elo import _k_base, _g_multiplier


def build_features(df: pd.DataFrame, home_adv: float = 65.0, base: float = 1500.0, form_n: int = 5):
    """Return (X, y, dates): one pre-match feature row per match + the outcome.

    Features (all known before kickoff): Elo of each side and their gap, a
    home flag, rolling average goals for/against over the last `form_n` games,
    rolling result form, and how many games each team has on record.
    """
    df = df.sort_values("date", kind="stable").reset_index(drop=True)  # stable: keep within-date order
    elo = defaultdict(lambda: base)
    gf = defaultdict(lambda: deque(maxlen=form_n))
    ga = defaultdict(lambda: deque(maxlen=form_n))
    rf = defaultdict(lambda: deque(maxlen=form_n))
    rows, ys, dates = [], [], []

    for r in df.itertuples():
        h, a = r.home_team, r.away_team
        neutral = bool(getattr(r, "neutral", False))
        eh, ea = elo[h], elo[a]
        ha = 0.0 if neutral else home_adv
        rows.append({
            # use the Elo GAP, not absolute ratings — absolute Elo drifts over
            # the years and wrecks tree extrapolation on the recent test set.
            "elo_diff": eh - ea + ha,
            "home_adv": 0 if neutral else 1,
            "gf_home": np.mean(gf[h]) if gf[h] else 1.0,
            "ga_home": np.mean(ga[h]) if ga[h] else 1.0,
            "gf_away": np.mean(gf[a]) if gf[a] else 1.0,
            "ga_away": np.mean(ga[a]) if ga[a] else 1.0,
            "form_home": np.mean(rf[h]) if rf[h] else 0.5,
            "form_away": np.mean(rf[a]) if rf[a] else 0.5,
            "n_home": len(gf[h]), "n_away": len(gf[a]),
        })
        if r.home_score > r.away_score:
            ys.append("home"); sh = 1.0
        elif r.home_score < r.away_score:
            ys.append("away"); sh = 0.0
        else:
            ys.append("draw"); sh = 0.5
        dates.append(r.date)

        # update state AFTER recording the row (so features stay pre-match)
        exp = 1.0 / (1.0 + 10 ** (-(eh - ea + ha) / 400.0))
        k = _k_base(getattr(r, "tournament", "")) * _g_multiplier(r.home_score - r.away_score)
        d = k * (sh - exp)
        elo[h] = eh + d; elo[a] = ea - d
        gf[h].append(r.home_score); ga[h].append(r.away_score); rf[h].append(sh)
        gf[a].append(r.away_score); ga[a].append(r.home_score); rf[a].append(1 - sh)

    return pd.DataFrame(rows), np.array(ys), pd.to_datetime(pd.Series(dates))


def fit_classifier(X_train, y_train, **kw):
    """A gradient-boosting 1X2 classifier (HistGradientBoosting — fast, robust)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    params = dict(max_iter=200, learning_rate=0.05, max_depth=3,
                  l2_regularization=2.0, random_state=0)
    params.update(kw)
    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


def predict_1x2(clf, X) -> np.ndarray:
    """Return probabilities ordered [home, draw, away]."""
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    idx = [classes.index(c) for c in ("home", "draw", "away")]
    return proba[:, idx]
