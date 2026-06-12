"""Load and clean historical international results.

Primary source: Kaggle "International football results 1872-present"
(martj42/international-football-results-from-1872-to-2017), columns:
    date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

Drop the results.csv into data/raw/ and point `load_results` at it.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

# friendlies should count less than competitive games (used at rung 3)
COMPETITION_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "FIFA World Cup qualification": 0.9,
    "UEFA Euro": 1.0,
    "UEFA Euro qualification": 0.9,
    "UEFA Nations League": 0.85,
    "Copa América": 1.0,
    "African Cup of Nations": 0.9,
    "Friendly": 0.5,
}
DEFAULT_WEIGHT = 0.8


def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw results CSV and return a typed, sorted frame."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    expected = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def filter_recent(df: pd.DataFrame, since: str | None = None, years: int | None = None) -> pd.DataFrame:
    """Keep matches after a cutoff date, or within the last N years of the data."""
    if since is not None:
        return df[df["date"] >= pd.Timestamp(since)].reset_index(drop=True)
    if years is not None:
        cutoff = df["date"].max() - pd.DateOffset(years=years)
        return df[df["date"] >= cutoff].reset_index(drop=True)
    return df


def filter_teams(df: pd.DataFrame, min_matches: int = 20) -> pd.DataFrame:
    """Keep only matches between teams that appear at least `min_matches` times.

    The raw data has ~280 nations once you include micro-states and one-off
    sides; most play a handful of games and just add noise (and fitting cost)
    to the strength estimates. Restricting to teams with a real sample makes the
    fit faster and the strengths more stable. Applied iteratively because
    dropping a team changes everyone else's counts.
    """
    out = df
    while True:
        counts = pd.concat([out["home_team"], out["away_team"]]).value_counts()
        keep = set(counts[counts >= min_matches].index)
        mask = out["home_team"].isin(keep) & out["away_team"].isin(keep)
        if mask.all():
            break
        out = out[mask]
    return out.reset_index(drop=True)


def add_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `weight` column from the competition type (friendlies count less)."""
    out = df.copy()
    out["weight"] = out["tournament"].map(COMPETITION_WEIGHTS).fillna(DEFAULT_WEIGHT)
    return out


def make_synthetic(n_teams: int = 8, n_matches: int = 400, seed: int = 0) -> pd.DataFrame:
    """Generate a fake results frame so the pipeline runs without a download.

    Each team gets a latent attack/defence strength; goals are Poisson draws.
    Useful for smoke-testing the notebook before the real CSV is in place.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    teams = [f"Team {chr(65 + i)}" for i in range(n_teams)]
    attack = dict(zip(teams, rng.normal(0, 0.35, n_teams)))
    defence = dict(zip(teams, rng.normal(0, 0.35, n_teams)))
    home_adv = 0.25
    rows = []
    base = pd.Timestamp("2018-01-01")
    for k in range(n_matches):
        h, a = rng.choice(teams, size=2, replace=False)
        lam_h = np.exp(0.1 + home_adv + attack[h] - defence[a])
        lam_a = np.exp(0.1 + attack[a] - defence[h])
        rows.append(
            {
                "date": base + pd.Timedelta(days=k),
                "home_team": h,
                "away_team": a,
                "home_score": int(rng.poisson(lam_h)),
                "away_score": int(rng.poisson(lam_a)),
                "tournament": "Friendly",
                "neutral": False,
            }
        )
    return pd.DataFrame(rows)
