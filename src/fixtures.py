"""Load the WC2026 fixture list from openfootball (no API key required).

Schema of data/raw/worldcup2026.json (one tournament, ~104 matches):
    {"name": "World Cup 2026",
     "matches": [
        {"round": "Matchday 1", "date": "2026-06-11", "time": "13:00 UTC-6",
         "team1": "Mexico", "team2": "South Africa", "group": "Group A",
         "ground": "Mexico City",
         "score": {"ft": [h, a], "ht": [h, a]}   # present only once played
        }, ...]}

openfootball mirrors the official FIFA schedule but is hand-updated, so it lags
~a day on (a) match scores and (b) the names of the play-off qualifiers — it
shows "UEFA Path A winner", "IC Path 1 winner", etc. until merged. PLAYOFF_RESOLUTIONS
fills those in from the confirmed results so the schedule is correct immediately.
Knockout slots ("2A", "W74", "3A/B/C/D/F") stay as placeholders until the bracket fills.

Verify the loaded schedule any time with scripts/check_fixtures.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# openfootball play-off placeholder -> real team (martj42 names).
# Confirmed winners: UEFA A=Bosnia, B=Sweden, C=Turkiye, D=Czechia; IC=DR Congo, Iraq.
PLAYOFF_RESOLUTIONS = {
    "UEFA Path A winner": "Bosnia and Herzegovina",
    "UEFA Path B winner": "Sweden",
    "UEFA Path C winner": "Turkey",
    "UEFA Path D winner": "Czech Republic",
    "IC Path 1 winner": "DR Congo",
    "IC Path 2 winner": "Iraq",
}

# openfootball name -> martj42 name. Extend as reconcile_names reports misses.
ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
    "Bosnia": "Bosnia and Herzegovina",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "China PR": "China",
}


def _resolve(name: str) -> str:
    return PLAYOFF_RESOLUTIONS.get(str(name).strip(), name)


def _is_placeholder(name: str) -> bool:
    """True for unresolved bracket/qualifier slots that aren't a real team yet."""
    n = str(name).strip()
    if not n:
        return True
    low = n.lower()
    if "winner" in low or "path" in low or "place" in low or "/" in n:
        return True
    if re.match(r"^[WL]\d{1,3}$", n):       # W74, L101 (knockout)
        return True
    if re.match(r"^\d[A-L]$", n):           # 2A, 1E (group positions)
        return True
    return False


def load_fixtures(path: str | Path = "../data/raw/worldcup2026.json") -> pd.DataFrame:
    """Return one row per match, with play-off names resolved and flags set.

    Columns: round, group, date, time, ground, team1, team2,
             resolved (both sides are real teams), played, home_score, away_score.
    """
    data = json.loads(Path(path).read_text())
    rows = []
    for m in data.get("matches", []):
        t1, t2 = _resolve(m.get("team1", "")), _resolve(m.get("team2", ""))
        score = m.get("score") or {}
        ft = score.get("ft")
        rows.append({
            "round": m.get("round"), "group": m.get("group"),
            "date": m.get("date"), "time": m.get("time"), "ground": m.get("ground"),
            "team1": t1, "team2": t2,
            "resolved": not _is_placeholder(t1) and not _is_placeholder(t2),
            "played": bool(ft),
            "home_score": ft[0] if ft else None, "away_score": ft[1] if ft else None,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def reconcile_names(fixtures: pd.DataFrame, model_teams, aliases: dict | None = None):
    """Map fixture team names onto the model's team set.

    Returns (fixtures_with_mapped_names, unmatched_set). `unmatched_set` lists real
    team names that still don't match after aliasing — add those to ALIASES.
    Placeholder slots are ignored (not 'unmatched').
    """
    aliases = {**ALIASES, **(aliases or {})}
    model_teams = set(model_teams)

    def m(name):
        if _is_placeholder(name):
            return name
        name = name.replace(" & ", " and ")   # "Bosnia & Herzegovina" -> "...and..."
        return aliases.get(name, name)

    out = fixtures.copy()
    out["team1"] = out["team1"].map(m)
    out["team2"] = out["team2"].map(m)

    unmatched = set()
    for col in ("team1", "team2"):
        for name in out[col].unique():
            if not _is_placeholder(name) and name not in model_teams:
                unmatched.add(name)
    return out, unmatched


def predict_fixtures(model, fixtures: pd.DataFrame, neutral: bool = True) -> pd.DataFrame:
    """Run a fitted model over the resolved, unplayed fixtures (Poisson or DC)."""
    known = set(model.teams)
    recs = []
    for r in fixtures.itertuples():
        if r.played or not r.resolved or r.team1 not in known or r.team2 not in known:
            continue
        try:
            probs = model.outcome_probs(r.team1, r.team2, neutral=neutral)
            top = model.most_likely_scores(r.team1, r.team2, neutral=neutral)[0][0]
        except TypeError:
            probs = model.outcome_probs(r.team1, r.team2)
            top = model.most_likely_scores(r.team1, r.team2)[0][0]
        recs.append({
            "date": r.date, "group": r.group, "round": r.round,
            "team1": r.team1, "team2": r.team2,
            "p_home": round(probs["home"], 3), "p_draw": round(probs["draw"], 3),
            "p_away": round(probs["away"], 3), "pred_score": f"{top[0]}-{top[1]}",
        })
    return pd.DataFrame(recs)
