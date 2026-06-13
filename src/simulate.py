"""Monte-Carlo tournament simulation — who-wins-it-all odds.

We can predict one match. To get "% chance to win the World Cup" we play the WHOLE
tournament thousands of times from the model's probabilities and count outcomes:

  * group stage: every group game is a sampled scoreline (Poisson on the model's
    expected goals), tabled into points/goal-difference/goals-for, then ranked.
  * qualifiers: top 2 of each group + the 8 best third-placed teams (the 48-team
    2026 format).
  * knockouts: each tie is decided by the model's win probability, with draws
    going to a coin-flip shootout. Repeat to the final.

The bracket structure (which slot meets which) is read from the openfootball
schedule. The third-place→slot routing uses a valid assignment respecting each
slot's allowed groups (a close approximation of FIFA's official table; it barely
moves champion odds).

Speed: we precompute every team-vs-team expected-goals and win-probability once,
so each simulated match is just a table lookup + a random draw.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

KO_ROUND = {  # game number -> round label (2026 bracket)
    **{n: "R32" for n in range(73, 89)},
    **{n: "R16" for n in range(89, 97)},
    **{n: "QF" for n in range(97, 101)},
    **{n: "SF" for n in (101, 102)},
    103: "Final",
}


def _winner(team1, team2, score):
    """Winning team name from a score dict (penalties beat full-time), else None."""
    s = (score or {}).get("p") or (score or {}).get("ft")
    if not s:
        return None
    return team1 if s[0] > s[1] else team2 if s[1] > s[0] else None


def parse_structure(path: str):
    """Return (groups, group_games, ko_games) from the openfootball schedule.

    groups:      {"A": [team1..team4], ...}
    group_games: {"A": [(t1, t2, played, (h,a) or None), ...]}  the 6 games per group
    ko_games:    ordered [{"num","s1","s2","played","winner"}]  winner set once decided
    All team names are normalised to the model's spelling. Played games carry their
    real results so the caller can CONDITION on what's already happened.
    """
    from .fixtures import normalize_team  # resolve playoff slots + aliases

    data = json.loads(open(path).read())
    groups = defaultdict(list)
    group_games = defaultdict(list)
    ko = {}
    for m in data.get("matches", []):
        rnd = m.get("round", "")
        t1, t2 = normalize_team(m["team1"]), normalize_team(m["team2"])
        score = m.get("score") or {}
        ft = score.get("ft")
        if m.get("group", "").startswith("Group"):
            g = m["group"].split()[-1]
            for t in (t1, t2):
                if t not in groups[g]:
                    groups[g].append(t)
            group_games[g].append((t1, t2, ft is not None, tuple(ft) if ft else None))
        elif rnd == "Final":
            ko[103] = {"num": 103, "s1": "W101", "s2": "W102",
                       "played": ft is not None, "winner": _winner(t1, t2, score)}
        elif "num" in m and rnd in ("Round of 32", "Round of 16", "Quarter-final", "Semi-final"):
            ko[m["num"]] = {"num": m["num"], "s1": m["team1"], "s2": m["team2"],
                            "played": ft is not None, "winner": _winner(t1, t2, score)}
    ko_games = [ko[n] for n in sorted(ko)]
    return dict(groups), dict(group_games), ko_games


def build_tables(teams, dm, em, w_dc=0.6):
    """Precompute expected-goals and advance-probability matrices over `teams`."""
    from . import elo as E
    n = len(teams)
    EG = np.zeros((n, n))
    PADV = np.zeros((n, n))
    for i, ti in enumerate(teams):
        for j, tj in enumerate(teams):
            if i == j:
                continue
            EG[i, j] = dm.expected_goals(ti, tj, neutral=True)[0]
            o = E.ensemble_probs(dm.outcome_probs(ti, tj, neutral=True),
                                 em.outcome_probs(ti, tj, neutral=True), w_dc)
            PADV[i, j] = o["home"] + 0.5 * o["draw"]   # i beats j, draws -> shootout
    return EG, PADV


def _match_thirds(qual, slots, rng):
    """Assign qualifying third-placed teams to third-place slots (valid matching).

    qual: list of (team_idx, group_letter). slots: list of (slot_id, allowed_set).
    Returns {slot_id: team_idx}. Greedy most-constrained-first with random tie-break.
    """
    assign = {}
    used = set()
    order = sorted(slots, key=lambda s: len(s[1]))   # tightest slots first
    for sid, allowed in order:
        cands = [t for t in qual if t[0] not in used and t[1] in allowed]
        if not cands:
            cands = [t for t in qual if t[0] not in used]  # fallback: any unused
        pick = cands[rng.integers(len(cands))]
        assign[sid] = pick[0]; used.add(pick[0])
    return assign


def simulate(teams, groups, group_games, ko_games, EG, PADV, n_sims=10000, seed=0):
    """Run the tournament n_sims times, CONDITIONING on games already played.

    Group games with a real score are banked (fixed points/goals); only unplayed
    games are sampled. Knockout ties already decided use their actual winner.
    Returns dict team -> stage probabilities.
    """
    from .fixtures import normalize_team
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(teams)}

    members = {g: [idx[t] for t in ts if t in idx] for g, ts in groups.items()}
    # group games as indices, carrying played/score
    gg = {g: [(idx[a], idx[b], pl, sc) for (a, b, pl, sc) in games if a in idx and b in idx]
          for g, games in group_games.items()}

    third_slots = []
    for game in ko_games:
        for side in ("s1", "s2"):
            code = game[side]
            if code.startswith("3") and "/" in code:
                third_slots.append(((game["num"], side), set(re.findall(r"[A-L]", code))))

    # precompute fixed knockout outcomes (same every sim)
    ko_pre = []
    for game in ko_games:
        played = bool(game.get("played")) and game.get("winner") in idx
        p1, p2, w = None, None, None
        if played:
            p1 = idx.get(normalize_team(game["s1"])); p2 = idx.get(normalize_team(game["s2"]))
            w = idx[game["winner"]]
            if p1 is None or p2 is None:   # can't place participants -> simulate instead
                played = False
        ko_pre.append({"num": game["num"], "s1": game["s1"], "s2": game["s2"],
                       "played": played, "p1": p1, "p2": p2, "w": w})

    stages = ("win_group", "advance", "R16", "QF", "SF", "Final", "champion")
    count = {t: dict.fromkeys(stages, 0) for t in teams}

    for _ in range(n_sims):
        winner, runner, third = {}, {}, {}
        for g, games in gg.items():
            if len(members[g]) < 4:
                continue
            pts = defaultdict(int); gd = defaultdict(int); gf = defaultdict(int)
            for x, y, played, score in games:
                if played:
                    gx, gy = score
                else:
                    gx = rng.poisson(EG[x, y]); gy = rng.poisson(EG[y, x])
                gf[x] += gx; gf[y] += gy; gd[x] += gx - gy; gd[y] += gy - gx
                if gx > gy: pts[x] += 3
                elif gx < gy: pts[y] += 3
                else: pts[x] += 1; pts[y] += 1
            ranked = sorted(members[g], key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True)
            winner[g], runner[g] = ranked[0], ranked[1]
            third[g] = (ranked[2], pts[ranked[2]], gd[ranked[2]], gf[ranked[2]], g)
            count[teams[ranked[0]]]["win_group"] += 1

        thirds = sorted(third.values(), key=lambda x: (x[1], x[2], x[3], rng.random()), reverse=True)[:8]
        third_assign = _match_thirds([(t[0], t[4]) for t in thirds], third_slots, rng)

        def resolve(code, num, side, results):
            if re.match(r"^1[A-L]$", code): return winner[code[1]]
            if re.match(r"^2[A-L]$", code): return runner[code[1]]
            if code.startswith("3"): return third_assign[(num, side)]
            if code.startswith("W"): return results[int(code[1:])]
            return None

        results = {}
        seen = defaultdict(set)
        for game in ko_pre:
            if game["played"]:
                t1, t2, w = game["p1"], game["p2"], game["w"]
            else:
                t1 = resolve(game["s1"], game["num"], "s1", results)
                t2 = resolve(game["s2"], game["num"], "s2", results)
                w = t1 if rng.random() < PADV[t1, t2] else t2
            seen[KO_ROUND[game["num"]]].update((t1, t2))
            results[game["num"]] = w
        for rnd, key in (("R32", "advance"), ("R16", "R16"), ("QF", "QF"), ("SF", "SF"), ("Final", "Final")):
            for t in seen[rnd]:
                count[teams[t]][key] += 1
        count[teams[results[103]]]["champion"] += 1

    return {t: {s: count[t][s] / n_sims for s in stages} for t in teams}
