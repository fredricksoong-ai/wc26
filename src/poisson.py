"""Rung 1: the Poisson goals model.

The idea, in one paragraph:
    Every team has an *attack* strength (how many goals it tends to score) and a
    *defence* strength (how many it tends to concede). Plus a global home
    advantage. For a given match we compute two expected goal rates -- lambda_home
    and lambda_away -- and treat each side's goal count as an independent Poisson
    random variable. From those two Poissons we build the full grid of scoreline
    probabilities P(home=i, away=j), then collapse it into win/draw/loss and into
    the most-likely scorelines.

We fit the strengths with a Poisson GLM (log link), which is the standard,
fully-explainable way to estimate this. statsmodels does the heavy lifting; the
code below is deliberately short so you can read every line.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import poisson


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape one-row-per-match into two-rows-per-match (one per team's goals).

    Each match becomes:
        team, opponent, goals, home (1/0)
    This 'long' form is what the GLM needs: predict `goals` from team attack,
    opponent defence, and a home flag.
    """
    home = pd.DataFrame(
        {
            "team": df["home_team"],
            "opponent": df["away_team"],
            "goals": df["home_score"],
            "home": 1,
        }
    )
    away = pd.DataFrame(
        {
            "team": df["away_team"],
            "opponent": df["home_team"],
            "goals": df["away_score"],
            "home": 0,
        }
    )
    return pd.concat([home, away], ignore_index=True)


@dataclass
class PoissonModel:
    """A fitted Poisson strength model.

    `result` is the statsmodels GLM result. `predict_match` turns two team
    names into a scoreline-probability matrix.
    """

    result: object
    teams: list[str]
    max_goals: int = 10

    def expected_goals(self, home_team: str, away_team: str, neutral: bool = False) -> tuple[float, float]:
        """lambda_home, lambda_away. neutral=True drops home advantage (World Cup venues)."""
        lam_home = self._mu(team=home_team, opponent=away_team, home=0 if neutral else 1)
        lam_away = self._mu(team=away_team, opponent=home_team, home=0)
        return lam_home, lam_away

    def _mu(self, team: str, opponent: str, home: int) -> float:
        row = pd.DataFrame([{"team": team, "opponent": opponent, "home": home}])
        return float(self.result.predict(row).iloc[0])

    def score_matrix(self, home_team: str, away_team: str, neutral: bool = False) -> np.ndarray:
        """P(home=i, away=j) grid, shape (max_goals+1, max_goals+1).

        Independence assumption: joint = product of the two marginal Poissons.
        (Rung 2 / Dixon-Coles relaxes this for low scores.)
        """
        lam_h, lam_a = self.expected_goals(home_team, away_team, neutral=neutral)
        k = np.arange(self.max_goals + 1)
        p_home = poisson.pmf(k, lam_h)
        p_away = poisson.pmf(k, lam_a)
        return np.outer(p_home, p_away)

    def outcome_probs(self, home_team: str, away_team: str, neutral: bool = False) -> dict[str, float]:
        """Collapse the grid into home-win / draw / away-win probabilities."""
        m = self.score_matrix(home_team, away_team, neutral=neutral)
        home_win = np.tril(m, -1).sum()  # i > j
        draw = np.trace(m)               # i == j
        away_win = np.triu(m, 1).sum()   # i < j
        return {"home": float(home_win), "draw": float(draw), "away": float(away_win)}

    def most_likely_scores(self, home_team: str, away_team: str, top: int = 5):
        """Return the `top` most probable exact scorelines."""
        m = self.score_matrix(home_team, away_team)
        idx = np.dstack(np.unravel_index(np.argsort(m.ravel())[::-1], m.shape))[0]
        return [((int(i), int(j)), float(m[i, j])) for i, j in idx[:top]]


def fit(df: pd.DataFrame, max_goals: int = 10) -> PoissonModel:
    """Fit the Poisson GLM on a results frame.

    Model:  log(E[goals]) = intercept + home*beta + C(team) + C(opponent)
    -> the C(team) coefficients are attack strengths, C(opponent) are (inverse)
    defence strengths, and `home` is the global home advantage.
    """
    import statsmodels.formula.api as smf

    long = to_long(df)
    model = smf.glm(
        formula="goals ~ home + C(team) + C(opponent)",
        data=long,
        family=__import__("statsmodels.api", fromlist=["families"]).families.Poisson(),
    )
    result = model.fit()
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return PoissonModel(result=result, teams=teams, max_goals=max_goals)
