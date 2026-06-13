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


def parse_structure(path: str):
    """Return (groups, ko_games) from the openfootball schedule.

    groups: {"A": [team1..team4], ...} (raw names — caller resolves/aliases)
    ko_games: ordered list of {"num", "s1", "s2"} for R32 → Final.
    """
    from .fixtures import normalize_team  # resolve playoff slots + aliases

    data = json.loads(open(path).read())
    groups = defaultdict(list)
    ko = {}
    for m in data.get("matches", []):
        rnd = m.get("round", "")
        if m.get("group", "").startswith("Group"):
            g = m["group"].split()[-1]
            for t in (normalize_team(m["team1"]), normalize_team(m["team2"])):
                if t not in groups[g]:
                    groups[g].append(t)
        elif rnd == "Final":
            ko[103] = {"num": 103, "s1": "W101", "s2": "W102"}
        elif "num" in m and rnd in ("Round of 32", "Round of 16", "Quarter-final", "Semi-final"):
            ko[m["num"]] = {"num": m["num"], "s1": m["team1"], "s2": m["team2"]}
    ko_games = [ko[n] for n in sorted(ko)]
    return dict(groups), ko_games


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


def simulate(teams, groups, ko_games, EG, PADV, n_sims=10000, seed=0):
    """Run the tournament n_sims times. Returns a dict team -> stage probabilities."""
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(teams)}
    # groups as index lists (skip any team not in the model/table)
    gidx = {g: [idx[t] for t in ts if t in idx] for g, ts in groups.items()}
    # third-place slots from the bracket (slots whose code starts with '3')
    third_slots = []
    for game in ko_games:
        for side in ("s1", "s2"):
            code = game[side]
            if code.startswith("3") and "/" in code:
                allowed = set(re.findall(r"[A-L]", code))
                third_slots.append(((game["num"], side), allowed))

    stages = ("win_group", "advance", "R16", "QF", "SF", "Final", "champion")
    count = {t: dict.fromkeys(stages, 0) for t in teams}

    for _ in range(n_sims):
        winner, runner, third = {}, {}, {}
        for g, members in gidx.items():
            if len(members) < 4:
                continue
            pts = defaultdict(int); gd = defaultdict(int); gf = defaultdict(int)
            for a in range(4):
                for b in range(a + 1, 4):
                    x, y = members[a], members[b]
                    gx = rng.poisson(EG[x, y]); gy = rng.poisson(EG[y, x])
                    gf[x] += gx; gf[y] += gy; gd[x] += gx - gy; gd[y] += gy - gx
                    if gx > gy: pts[x] += 3
                    elif gx < gy: pts[y] += 3
                    else: pts[x] += 1; pts[y] += 1
            ranked = sorted(members, key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True)
            winner[g], runner[g], third[g] = ranked[0], ranked[1], ranked[2]
            count[teams[winner[g]]]["win_group"] += 1
            # third-place ranking key
            third[g] = (ranked[2], pts[ranked[2]], gd[ranked[2]], gf[ranked[2]], g)

        # best 8 third-placed teams
        thirds = sorted(third.values(), key=lambda x: (x[1], x[2], x[3], rng.random()), reverse=True)[:8]
        qual = [(t[0], t[4]) for t in thirds]
        third_assign = _match_thirds(qual, third_slots, rng)

        def resolve(code, game_num, side, results):
            if re.match(r"^1[A-L]$", code): return winner[code[1]]
            if re.match(r"^2[A-L]$", code): return runner[code[1]]
            if code.startswith("3"): return third_assign[(game_num, side)]
            if code.startswith("W"): return results[int(code[1:])]
            return None

        results = {}
        seen = defaultdict(set)
        for game in ko_games:
            t1 = resolve(game["s1"], game["num"], "s1", results)
            t2 = resolve(game["s2"], game["num"], "s2", results)
            rnd = KO_ROUND[game["num"]]
            for t in (t1, t2):
                seen[rnd].add(t)
            w = t1 if rng.random() < PADV[t1, t2] else t2
            results[game["num"]] = w
        # tally stage reached
        for t in seen["R32"]: count[teams[t]]["advance"] += 1
        for t in seen["R16"]: count[teams[t]]["R16"] += 1
        for t in seen["QF"]: count[teams[t]]["QF"] += 1
        for t in seen["SF"]: count[teams[t]]["SF"] += 1
        for t in seen["Final"]: count[teams[t]]["Final"] += 1
        count[teams[results[103]]]["champion"] += 1

    return {t: {s: count[t][s] / n_sims for s in stages} for t in teams}
