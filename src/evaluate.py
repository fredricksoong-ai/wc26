"""Probabilistic scoring for 1X2 predictions.

All three metrics are 'lower is better'. Use them to compare:
    your model  vs  bookmaker-implied probs  vs  API-Football /predictions.

Convention: probabilities are ordered [home, draw, away] and the observed
outcome is one of "home" / "draw" / "away".
"""
from __future__ import annotations

import numpy as np

OUTCOMES = ("home", "draw", "away")


def _onehot(outcome: str) -> np.ndarray:
    return np.array([1.0 if o == outcome else 0.0 for o in OUTCOMES])


def rps(probs, outcome: str) -> float:
    """Ranked Probability Score for a single ordered-categorical prediction.

    RPS respects ordering (home < draw < away), which is why it's the standard
    for 1X2: predicting a draw when the truth is a home win is penalised less
    than predicting an away win. 0 = perfect, higher = worse.
    """
    p = np.asarray(probs, dtype=float)
    o = _onehot(outcome)
    cum_p = np.cumsum(p)
    cum_o = np.cumsum(o)
    return float(np.sum((cum_p - cum_o) ** 2) / (len(OUTCOMES) - 1))


def brier(probs, outcome: str) -> float:
    """Multiclass Brier score: mean squared error vs the one-hot outcome."""
    p = np.asarray(probs, dtype=float)
    o = _onehot(outcome)
    return float(np.mean((p - o) ** 2))


def log_loss(probs, outcome: str, eps: float = 1e-15) -> float:
    """Negative log-likelihood of the observed outcome."""
    p = np.asarray(probs, dtype=float)
    idx = OUTCOMES.index(outcome)
    return float(-np.log(np.clip(p[idx], eps, 1.0)))


def mean_scores(prob_rows, outcomes) -> dict[str, float]:
    """Average each metric over a backtest set.

    prob_rows: iterable of [home, draw, away] arrays.
    outcomes:  iterable of "home"/"draw"/"away" strings, same length.
    """
    prob_rows = list(prob_rows)
    outcomes = list(outcomes)
    assert len(prob_rows) == len(outcomes), "probs and outcomes length mismatch"
    return {
        "rps": float(np.mean([rps(p, o) for p, o in zip(prob_rows, outcomes)])),
        "brier": float(np.mean([brier(p, o) for p, o in zip(prob_rows, outcomes)])),
        "log_loss": float(np.mean([log_loss(p, o) for p, o in zip(prob_rows, outcomes)])),
        "n": len(outcomes),
    }


def result_to_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    return "draw"


def implied_probs_from_odds(home_odds: float, draw_odds: float, away_odds: float):
    """Decimal odds -> normalised implied probabilities (removes the overround)."""
    raw = np.array([1 / home_odds, 1 / draw_odds, 1 / away_odds])
    return raw / raw.sum()
