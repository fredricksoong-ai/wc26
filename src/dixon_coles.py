"""Rung 2: Dixon-Coles.

Two upgrades over the plain Poisson model (rung 1), both from Dixon & Coles (1997):

1. **Low-score dependence correction.** Independent Poissons under-count the
   chunky 0-0 / 1-0 / 0-1 / 1-1 results that football actually produces. DC
   multiplies those four cells by a correction factor tau(i, j; lambda_h, lambda_a, rho),
   governed by a single parameter rho. rho < 0 lifts draws/low scores.

2. **Time-decay weighting.** Each match's contribution to the likelihood is
   weighted exp(-xi * age_in_days). Recent form counts more; old games fade
   smoothly instead of being cut off by a hard date window.

Unlike rung 1 (a GLM) this needs a custom maximum-likelihood fit, so we optimise
the log-likelihood directly with scipy. The parameterisation:
    attack[t]      one per team   (sum-to-zero constrained for identifiability)
    defence[t]     one per team
    home_adv       scalar (set to 0 for neutral venues -- see predict)
    rho            the low-score correction

The predict interface mirrors PoissonModel so the notebook can swap them freely:
    expected_goals, score_matrix, outcome_probs, most_likely_scores.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


# --- the tau correction -----------------------------------------------------
def _tau(i, j, lam, mu, rho):
    """Dixon-Coles adjustment for the four low-score cells; 1 elsewhere."""
    i = np.asarray(i); j = np.asarray(j)
    out = np.ones(np.broadcast(i, j).shape, dtype=float)
    out = np.where((i == 0) & (j == 0), 1 - lam * mu * rho, out)
    out = np.where((i == 0) & (j == 1), 1 + lam * rho, out)
    out = np.where((i == 1) & (j == 0), 1 + mu * rho, out)
    out = np.where((i == 1) & (j == 1), 1 - rho, out)
    return out


def time_weights(dates: pd.Series, xi: float, ref_date=None) -> np.ndarray:
    """exp(-xi * age_in_days). xi=0 -> all matches weighted equally.

    A handy reference: xi ~ 0.0019/day gives a half-life of ~1 year.
    """
    ref = pd.Timestamp(ref_date) if ref_date is not None else dates.max()
    age_days = (ref - dates).dt.days.to_numpy().astype(float)
    return np.exp(-xi * age_days)


@dataclass
class DixonColesModel:
    teams: list[str]
    attack: dict[str, float]
    defence: dict[str, float]
    home_adv: float
    rho: float
    max_goals: int = 10

    def expected_goals(self, home_team: str, away_team: str, neutral: bool = True):
        """lambda_home, lambda_away. neutral=True zeroes home advantage (most WC games)."""
        ha = 0.0 if neutral else self.home_adv
        lam_h = np.exp(self.attack[home_team] - self.defence[away_team] + ha)
        lam_a = np.exp(self.attack[away_team] - self.defence[home_team])
        return float(lam_h), float(lam_a)

    def score_matrix(self, home_team: str, away_team: str, neutral: bool = True) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(home_team, away_team, neutral=neutral)
        k = np.arange(self.max_goals + 1)
        base = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
        ii, jj = np.meshgrid(k, k, indexing="ij")
        m = base * _tau(ii, jj, lam_h, lam_a, self.rho)
        return m / m.sum()  # renormalise after the correction

    def outcome_probs(self, home_team: str, away_team: str, neutral: bool = True) -> dict[str, float]:
        m = self.score_matrix(home_team, away_team, neutral=neutral)
        return {
            "home": float(np.tril(m, -1).sum()),
            "draw": float(np.trace(m)),
            "away": float(np.triu(m, 1).sum()),
        }

    def most_likely_scores(self, home_team: str, away_team: str, top: int = 5, neutral: bool = True):
        m = self.score_matrix(home_team, away_team, neutral=neutral)
        idx = np.dstack(np.unravel_index(np.argsort(m.ravel())[::-1], m.shape))[0]
        return [((int(i), int(j)), float(m[i, j])) for i, j in idx[:top]]


def fit(df: pd.DataFrame, xi: float = 0.001, max_goals: int = 10, verbose: bool = False) -> DixonColesModel:
    """Maximum-likelihood fit with time-decay weighting.

    xi controls the decay. Default 0.001 (~2-year half-life) was the value that
    beat the rung-1 baseline on the international-results backtest -- but always
    re-tune it on your own split (see notebook 02). Set xi=0 to weight all
    matches equally (an apples-to-apples compare against rung 1).
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: k for k, t in enumerate(teams)}
    n = len(teams)

    h = df["home_team"].map(idx).to_numpy()
    a = df["away_team"].map(idx).to_numpy()
    gh = df["home_score"].to_numpy().astype(float)
    ga = df["away_score"].to_numpy().astype(float)
    w = time_weights(df["date"], xi) if "date" in df else np.ones(len(df))

    # params: [attack(n-1 free), defence(n), home_adv, rho]
    # attack is sum-to-zero constrained: attack[last] = -sum(others)
    def unpack(p):
        atk_free = p[: n - 1]
        attack = np.concatenate([atk_free, [-atk_free.sum()]])
        defence = p[n - 1 : 2 * n - 1]
        home_adv = p[2 * n - 1]
        rho = p[2 * n]
        return attack, defence, home_adv, rho

    def neg_ll(p):
        attack, defence, home_adv, rho = unpack(p)
        lam_h = np.exp(attack[h] - defence[a] + home_adv)
        lam_a = np.exp(attack[a] - defence[h])
        # poisson log-pmf for each side
        ll = (gh * np.log(lam_h) - lam_h) + (ga * np.log(lam_a) - lam_a)
        tau = _tau(gh, ga, lam_h, lam_a, rho)
        tau = np.clip(tau, 1e-9, None)  # keep log finite
        ll = ll + np.log(tau)
        return -np.sum(w * ll)

    x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25], [-0.05]])
    res = minimize(neg_ll, x0, method="L-BFGS-B",
                   options={"maxiter": 200, "disp": verbose})
    attack, defence, home_adv, rho = unpack(res.x)
    return DixonColesModel(
        teams=teams,
        attack=dict(zip(teams, attack)),
        defence=dict(zip(teams, defence)),
        home_adv=float(home_adv),
        rho=float(rho),
        max_goals=max_goals,
    )
