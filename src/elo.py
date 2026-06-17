"""Rung 3: Elo ratings as a strength prior.

Why this rung matters most for the World Cup: group-stage nations play few
recent comparable games, so the Poisson/DC strengths estimated from goals alone
are noisy. Elo is a single running rating updated after every international ever
played — it carries information from the whole history and is far more stable for
thin-data teams. We use it two ways:

  1. As a standalone 1X2 model (a strong, low-variance baseline).
  2. Blended (ensemble) with Dixon–Coles — the "prior + data" combination.

Elo mechanics (the World Football Elo variant):
    expected_home = 1 / (1 + 10^(-(R_home - R_away + home_adv) / 400))
    R' = R + K * G * (actual - expected)
where actual ∈ {1 win, 0.5 draw, 0 loss}, K scales with match importance, and
G is a goal-difference multiplier (bigger wins move ratings more).

Turning a rating gap into home/draw/away needs a draw model. We fit a tiny,
symmetric ordered-logistic on a SINGLE feature — the pre-match rating gap —
using only information available before each match (no leakage).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# match importance -> base K (World Football Elo style)
ELO_K = {
    "FIFA World Cup": 60,
    "Confederations Cup": 50,
    "UEFA Euro": 50,
    "Copa América": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Gold Cup": 50,
    "UEFA Nations League": 40,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 40,
    "Friendly": 20,
}
DEFAULT_K = 30


def _k_base(tournament: str) -> float:
    if tournament in ELO_K:
        return ELO_K[tournament]
    for key, val in ELO_K.items():  # partial match, e.g. "... qualification"
        if key in str(tournament):
            return val
    return DEFAULT_K


def _g_multiplier(goal_diff: int) -> float:
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def compute_ratings(df: pd.DataFrame, home_adv: float = 65.0, base: float = 1500.0,
                    xg_weight: float = 0.0, xg_tau: float = 1.0):
    """Single chronological pass. Returns (ratings dict, calib DataFrame).

    `calib` has one row per match: the PRE-match rating gap `dr` (home minus away,
    plus home advantage when not neutral) and the realised `outcome`. Because dr
    is recorded before the update, fitting the gap->1X2 mapping on it is leak-free.

    xG layer (Phase B): when a row carries measured match xG (columns xg_home/xg_away,
    NaN where unknown) and xg_weight > 0, the rating UPDATE for that game uses the xG
    performance instead of (or blended with) the goals result — the better signal for
    what a team's level really was. The pre-match gap `dr` and the calibration
    `outcome` stay on the REAL result (the gap->1X2 map must predict actual outcomes),
    so only how much ratings move changes, never what we predict from a given gap.

        sa_xg = 0.5 + 0.5*tanh((xg_home - xg_away) / xg_tau)   # soft, bounded margin
        sa    = xg_weight*sa_xg + (1 - xg_weight)*sa_goals
    """
    df = df.sort_values("date")
    has_xg = xg_weight > 0 and "xg_home" in df.columns and "xg_away" in df.columns
    ratings: dict[str, float] = {}
    drs, outs = [], []
    for r in df.itertuples():
        ra = ratings.get(r.home_team, base)
        rb = ratings.get(r.away_team, base)
        neutral = bool(getattr(r, "neutral", False))
        dr = ra - rb + (0.0 if neutral else home_adv)
        drs.append(dr)
        if r.home_score > r.away_score:
            outs.append("home"); sa = 1.0
        elif r.home_score < r.away_score:
            outs.append("away"); sa = 0.0
        else:
            outs.append("draw"); sa = 0.5
        gmult = _g_multiplier(r.home_score - r.away_score)
        if has_xg:
            xgh, xga = getattr(r, "xg_home", float("nan")), getattr(r, "xg_away", float("nan"))
            if xgh == xgh and xga == xga:          # both present (not NaN)
                margin = float(xgh) - float(xga)
                sa_xg = 0.5 + 0.5 * np.tanh(margin / xg_tau)
                sa = xg_weight * sa_xg + (1.0 - xg_weight) * sa
                gmult = _g_multiplier(round(margin))
        exp = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        k = _k_base(getattr(r, "tournament", "")) * gmult
        ratings[r.home_team] = ra + k * (sa - exp)
        ratings[r.away_team] = rb + k * (exp - sa)  # away gains the mirror image
    calib = pd.DataFrame({"dr": drs, "outcome": outs})
    return ratings, calib


def _probs_from_gap(dr, theta, s):
    """Symmetric ordered model: P(home), P(draw), P(away) from rating gap dr.

    Home/away win probs are logistic in dr with a shared draw band of width 2*theta;
    s controls steepness. Returns an (N,3) array.
    """
    dr = np.atleast_1d(np.asarray(dr, dtype=float))
    p_home = 1.0 / (1.0 + np.exp(-(dr - theta) / s))
    p_away = 1.0 / (1.0 + np.exp(-(-dr - theta) / s))
    p_draw = np.clip(1.0 - p_home - p_away, 1e-6, None)
    out = np.stack([p_home, p_draw, p_away], axis=1)
    return out / out.sum(axis=1, keepdims=True)


@dataclass
class EloModel:
    ratings: dict[str, float]
    home_adv: float
    theta: float
    s: float
    base: float = 1500.0
    teams: list[str] = field(default_factory=list)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def outcome_probs(self, home_team: str, away_team: str, neutral: bool = True) -> dict[str, float]:
        dr = self.rating(home_team) - self.rating(away_team) + (0.0 if neutral else self.home_adv)
        p = _probs_from_gap(dr, self.theta, self.s)[0]
        return {"home": float(p[0]), "draw": float(p[1]), "away": float(p[2])}

    # so fixtures.predict_fixtures' most_likely_scores call still works:
    def most_likely_scores(self, home_team: str, away_team: str, top: int = 1, neutral: bool = True):
        """Elo has no scoreline model; return the modal outcome as a coarse 'score'."""
        o = self.outcome_probs(home_team, away_team, neutral=neutral)
        pick = max(o, key=o.get)
        return [({"home": (1, 0), "draw": (1, 1), "away": (0, 1)}[pick], o[pick])]


def fit(df: pd.DataFrame, home_adv: float = 65.0, base: float = 1500.0,
        xg_weight: float = 0.0, xg_tau: float = 1.0) -> EloModel:
    """Compute ratings, then calibrate the gap->1X2 mapping by minimising log-loss."""
    ratings, calib = compute_ratings(df, home_adv=home_adv, base=base,
                                     xg_weight=xg_weight, xg_tau=xg_tau)
    y = calib["outcome"].map({"home": 0, "draw": 1, "away": 2}).to_numpy()
    dr = calib["dr"].to_numpy()

    def neg_ll(params):
        theta, s = params
        if s <= 1e-3:
            return 1e9
        p = _probs_from_gap(dr, theta, s)
        return -np.mean(np.log(p[np.arange(len(y)), y]))

    res = minimize(neg_ll, x0=[50.0, 150.0], method="Nelder-Mead",
                   options={"xatol": 1e-2, "fatol": 1e-6, "maxiter": 1000})
    theta, s = res.x
    return EloModel(ratings=ratings, home_adv=home_adv, theta=float(theta), s=float(s),
                    base=base, teams=sorted(ratings))


def ensemble_probs(p_a: dict, p_b: dict, w: float = 0.5) -> dict[str, float]:
    """Weighted blend of two 1X2 dicts: w*p_a + (1-w)*p_b. Renormalised."""
    out = {k: w * p_a[k] + (1 - w) * p_b[k] for k in ("home", "draw", "away")}
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}
