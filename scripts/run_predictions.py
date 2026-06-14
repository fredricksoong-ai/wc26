#!/usr/bin/env python3
"""Fit the model, freeze picks, score results, write predictions.json.

Two kinds of pick are tracked, kept separate:
  * MODEL pick  — the expected-points-optimal scoreline the model recommends.
                  Frozen in picks.json on the last run before kickoff.
  * YOUR pick   — what you actually entered in the pool, from my_picks.json.
                  You edit that file; this script scores it and shows where your
                  chosen scoreline sat in the model's probabilities.

The pool tally on the dashboard is YOUR points. The model's points are shown
alongside so you can see whether you're beating your own model.

my_picks.json format (key = "YYYY-MM-DD|team1|team2", value = "h-a"):
  { "2026-06-11|Mexico|South Africa": "3-0" }

Run after scripts/update_data.py. No API key required.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ)

from src import data, poisson as pois, dixon_coles as dc, elo as E, evaluate, fixtures as fx, scoring, simulate as sim

RAW = os.path.join(PROJ, "data", "raw")
DEPLOY = os.path.join(PROJ, "deploy")
PICKS = os.path.join(DEPLOY, "picks.json")          # model picks, frozen ledger
MY_PICKS = os.path.join(DEPLOY, "my_picks.json")     # YOUR pool entries (you edit)
OUT = os.path.join(DEPLOY, "predictions.json")

W_DC, RECENT_YEARS, MIN_MATCHES, XI = 0.6, 8, 10, 0.001
RESULT_PTS, EXACT_PTS = scoring.RESULT_PTS, scoring.EXACT_PTS


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _key(date, t1, t2):
    return f"{date}|{t1}|{t2}"


def clean(o):
    """Replace NaN/Inf (e.g. pandas group=NaN on knockout rows) with None.

    Python's json writes bare NaN/Infinity tokens that browsers' JSON.parse
    rejects — which silently breaks the whole dashboard. Sanitise before dump.
    """
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def dump_json(obj, path):
    """Write strictly-valid JSON (allow_nan=False) so a stray NaN fails loudly."""
    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=2, allow_nan=False)


def to_sgt(date, time_str):
    """openfootball 'HH:MM UTC-6' + date -> (iso, friendly label) in Singapore time (UTC+8)."""
    if not date or not time_str:
        return None, None
    m = re.match(r"(\d{1,2}):(\d{2})\s*UTC([+-]\d{1,2})", str(time_str))
    if not m:
        return None, None
    hh, mm, off = int(m.group(1)), int(m.group(2)), int(m.group(3))
    local = datetime.strptime(date, "%Y-%m-%d").replace(hour=hh, minute=mm)
    sgt = local - timedelta(hours=off) + timedelta(hours=8)   # -> UTC -> +8
    return sgt.strftime("%Y-%m-%dT%H:%M"), sgt.strftime("%a %d %b · %H:%M SGT")


def _parse(s):
    a, b = str(s).replace("–", "-").split("-")
    return int(a), int(b)


# --- Host advantage (2026 is co-hosted; hosts play real home games) ------------
# Every other game is neutral, but USA/Mexico/Canada playing in their own country
# get the model's fitted home edge. Venue -> host country from the openfootball
# `ground` city; all 16 venues are in one of the three hosts.
HOST_COUNTRY = {"United States": "USA", "Mexico": "MEX", "Canada": "CAN"}
_MEX = ("Mexico City", "Guadalajara", "Monterrey")
_CAN = ("Toronto", "Vancouver")
_USA = ("Atlanta", "San Francisco", "Los Angeles", "Seattle", "New York",
        "Boston", "Philadelphia", "Miami", "Houston", "Dallas", "Kansas City")


def _venue_country(ground):
    g = (str(ground) if ground is not None else "").strip()
    if g.startswith(_MEX): return "MEX"
    if g.startswith(_CAN): return "CAN"
    if g.startswith(_USA): return "USA"
    return None


def host_side(t1, t2, ground):
    """The side playing at home (a 2026 host in its own country), or None."""
    c = _venue_country(ground)
    if c is None:
        return None
    if HOST_COUNTRY.get(t1) == c: return t1
    if HOST_COUNTRY.get(t2) == c: return t2
    return None


def o_probs(probfn, t1, t2, ground):
    """outcome_probs oriented as t1=home/t2=away, applying host advantage if a host plays home."""
    h = host_side(t1, t2, ground)
    if h is None: return probfn(t1, t2, neutral=True)
    if h == t1:   return probfn(t1, t2, neutral=False)
    p = probfn(t2, t1, neutral=False)                    # host is t2: compute then flip
    return {"home": p["away"], "draw": p["draw"], "away": p["home"]}


def o_matrix(model, t1, t2, ground):
    """score_matrix with rows=t1 goals, cols=t2 goals, host advantage applied."""
    h = host_side(t1, t2, ground)
    if h is None: return model.score_matrix(t1, t2, neutral=True)
    if h == t1:   return model.score_matrix(t1, t2, neutral=False)
    return model.score_matrix(t2, t1, neutral=False).T


def o_xg(model, t1, t2, ground):
    """expected_goals as (t1, t2), host advantage applied."""
    h = host_side(t1, t2, ground)
    if h is None: return model.expected_goals(t1, t2, neutral=True)
    if h == t1:   return model.expected_goals(t1, t2, neutral=False)
    a, b = model.expected_goals(t2, t1, neutral=False)
    return (b, a)


def backtest_rps(rec, full):
    d = rec.sort_values("date"); cut = int(len(d) * 0.8)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    dm = dc.fit(tr, xi=XI); em = E.fit(full[full["date"] < te["date"].min()])
    known = set(dm.teams) & set(em.teams)
    rows, outs = [], []
    for r in te.itertuples():
        if r.home_team in known and r.away_team in known:
            p = E.ensemble_probs(dm.outcome_probs(r.home_team, r.away_team, neutral=False),
                                 em.outcome_probs(r.home_team, r.away_team, neutral=False), W_DC)
            rows.append([p["home"], p["draw"], p["away"]])
            outs.append(evaluate.result_to_outcome(r.home_score, r.away_score))
    ens = evaluate.mean_scores(rows, outs)
    br = (tr.assign(o=[evaluate.result_to_outcome(h, a) for h, a in zip(tr.home_score, tr.away_score)])
          ["o"].value_counts(normalize=True).reindex(["home", "draw", "away"]).fillna(0).values)
    base = evaluate.mean_scores([br] * len(outs), outs)
    return {"ensemble_rps": round(ens["rps"], 4), "baseline_rps": round(base["rps"], 4)}


def calibration_backtest(rec, full):
    """Out-of-sample reliability: are the model's probabilities trustworthy?

    Train the ensemble on everything up to ~12 months ago, predict the most recent
    year, then pool every (predicted prob, did-it-happen) pair across home/draw/away
    and bin by predicted probability. A well-calibrated model has observed frequency
    ≈ predicted probability in each bin. Also reports holdout RPS/Brier/log-loss/acc.
    """
    import numpy as np
    d = rec.sort_values("date")
    test_min = d["date"].max() - pd.Timedelta(days=365)
    tr, te = d[d["date"] < test_min], d[d["date"] >= test_min]
    if len(tr) < 200 or len(te) < 50:          # thin history -> fall back to 80/20
        cut = int(len(d) * 0.8)
        tr, te = d.iloc[:cut], d.iloc[cut:]
        test_min = te["date"].min()
    dm_c = dc.fit(tr, xi=XI)
    em_c = E.fit(full[full["date"] < test_min])
    known = set(dm_c.teams) & set(em_c.teams)
    rows, outs = [], []
    for r in te.itertuples():
        if r.home_team in known and r.away_team in known:
            neutral = bool(getattr(r, "neutral", False))
            p = E.ensemble_probs(dm_c.outcome_probs(r.home_team, r.away_team, neutral=neutral),
                                 em_c.outcome_probs(r.home_team, r.away_team, neutral=neutral), W_DC)
            rows.append([p["home"], p["draw"], p["away"]])
            outs.append(evaluate.result_to_outcome(r.home_score, r.away_score))
    if not rows:
        return None
    pred = [[] for _ in range(10)]; hit = [[] for _ in range(10)]
    for p, o in zip(rows, outs):
        oh = [1 if c == o else 0 for c in ("home", "draw", "away")]
        for c in range(3):
            b = min(int(p[c] * 10), 9)
            pred[b].append(p[c]); hit[b].append(oh[c])
    bins = [{"p": round((b + 0.5) / 10, 2),
             "pred": round(float(np.mean(pred[b])), 3),
             "obs": round(float(np.mean(hit[b])), 3),
             "n": len(pred[b])} for b in range(10) if pred[b]]
    ms = evaluate.mean_scores(rows, outs)
    acc = float(np.mean([["home", "draw", "away"][int(np.argmax(p))] == o for p, o in zip(rows, outs)]))
    br = (tr.assign(o=[evaluate.result_to_outcome(h, a) for h, a in zip(tr.home_score, tr.away_score)])
          ["o"].value_counts(normalize=True).reindex(["home", "draw", "away"]).fillna(0).values)
    base = evaluate.mean_scores([br] * len(outs), outs)
    return {"bins": bins, "n": len(outs), "window": "last 365 days",
            "metrics": {"rps": round(ms["rps"], 4), "brier": round(ms["brier"], 4),
                        "log_loss": round(ms["log_loss"], 4), "acc": round(acc, 3),
                        "baseline_rps": round(base["rps"], 4)}}


def main() -> int:
    full = data.load_results(os.path.join(RAW, "results.csv"))
    rec = data.filter_teams(data.filter_recent(full, years=RECENT_YEARS), MIN_MATCHES)
    dm = dc.fit(rec, xi=XI)
    em = E.fit(full)
    known = set(dm.teams) & set(em.teams)
    try:
        pm = pois.fit(rec)            # rung-1 Poisson, for the model leaderboard
    except Exception as e:
        pm = None; print(f"  (Poisson rung-1 skipped: {e})")

    LB_MODELS = ["poisson", "dixon_coles", "elo", "ensemble", "market"]

    # bookmaker odds (optional): de-vigged consensus probs keyed by team pair.
    odds_path = os.path.join(RAW, "odds.json")
    odds_raw = json.load(open(odds_path)) if os.path.exists(odds_path) else []
    odds_meta = odds_raw if isinstance(odds_raw, dict) else {}          # {fetched_utc, credits_remaining, matches}
    odds_list = odds_meta.get("matches", odds_raw if isinstance(odds_raw, list) else [])
    odds_idx = {frozenset((o["home"], o["away"])): o for o in odds_list}

    def market_probs(t1, t2):
        """De-vigged market 1X2 oriented to t1=home/t2=away, or None if not priced."""
        o = odds_idx.get(frozenset((t1, t2)))
        if not o:
            return None
        if o["home"] == t1: return {"home": o["p_home"], "draw": o["p_draw"], "away": o["p_away"]}
        if o["home"] == t2: return {"home": o["p_away"], "draw": o["p_draw"], "away": o["p_home"]}
        return None

    def model_eval(t1, t2, ground):
        """Each rung's HONEST prediction — not a pool-gamed pick.

        Per model: its most-likely scoreline, its most-likely result, and its 1X2
        probabilities. Judged three ways once a game finishes: correct result,
        correct scoreline, and RPS (probability quality). Elo has no goals model,
        so its 'scoreline' is the naive one implied by its result (it competes on
        result + RPS, not exact scores).
        """
        def mode(matrix):                  # most-likely scoreline AND its probability
            (i, j), p = scoring.top_scorelines(matrix, 1)[0]
            return f"{i}-{j}", round(float(p), 3)
        def amax(p):
            return max(("home", "draw", "away"), key=lambda kk: p[kk])
        def vec(p):
            return [round(p["home"], 4), round(p["draw"], 4), round(p["away"], 4)]
        def res_of(sc):                    # result implied by a submitted scoreline
            i, j = sc.split("-"); return scoring._result(int(i), int(j))
        naive = {"home": "1-0", "draw": "1-1", "away": "0-1"}

        dcm = o_matrix(dm, t1, t2, ground)
        dcp = o_probs(dm.outcome_probs, t1, t2, ground)
        elp = o_probs(em.outcome_probs, t1, t2, ground)
        enp = E.ensemble_probs(dcp, elp, W_DC)
        out = {}
        if pm is not None and t1 in pm.teams and t2 in pm.teams:
            ps, psp = mode(o_matrix(pm, t1, t2, ground))
            out["poisson"] = {"score": ps, "sp": psp, "result": res_of(ps),
                              "probs": vec(o_probs(pm.outcome_probs, t1, t2, ground))}
        ds, dsp = mode(dcm)
        es = naive[amax(elp)]
        out["dixon_coles"] = {"score": ds, "sp": dsp, "result": res_of(ds), "probs": vec(dcp)}
        out["elo"] = {"score": es, "sp": None, "result": res_of(es), "probs": vec(elp)}  # Elo: no score model
        out["ensemble"] = {"score": ds, "sp": dsp, "result": res_of(ds), "probs": vec(enp)}
        mp = market_probs(t1, t2)
        if mp is not None:                      # bookmaker consensus, scored like Elo (naive score, no scoreline model)
            mscore = naive[amax(mp)]
            out["market"] = {"score": mscore, "sp": None, "result": res_of(mscore), "probs": vec(mp)}
        return out

    def score_models(lb, actual):
        # everyone is scored like a pool entry: ONE scoreline, result derived FROM
        # that scoreline (so the League matches the Results tab). Deriving from the
        # frozen score — not a stored 'result' — means re-running corrects old rows.
        ah, aa = actual
        akey, ao = f"{ah}-{aa}", evaluate.result_to_outcome(ah, aa)
        def res_of(sc):
            i, j = sc.split("-"); return scoring._result(int(i), int(j))
        return {m: {"result_hit": res_of(v["score"]) == ao, "exact_hit": v["score"] == akey,
                    "rps": round(evaluate.rps(v["probs"], ao), 4)} for m, v in lb.items()}

    ledger = json.load(open(PICKS)) if os.path.exists(PICKS) else {}
    my_picks = json.load(open(MY_PICKS)) if os.path.exists(MY_PICKS) else {}

    fixtures_out = []
    fpath = os.path.join(RAW, "worldcup2026.json")
    if os.path.exists(fpath):
        fdf, unmatched = fx.reconcile_names(fx.load_fixtures(fpath), known)
        if unmatched:
            print(f"  WARN unmatched fixture names (add to ALIASES): {sorted(unmatched)}")

        for r in fdf.itertuples():
            date = r.date.strftime("%Y-%m-%d") if pd.notna(r.date) else None
            sgt_iso, sgt_label = to_sgt(date, getattr(r, "time", None))
            meta = {"date": date, "round": r.round, "group": r.group, "ground": r.ground,
                    "team1": r.team1, "team2": r.team2,
                    "kickoff_sgt": sgt_iso, "kickoff_label": sgt_label,
                    "host": host_side(r.team1, r.team2, r.ground),
                    "resolved": bool(r.resolved), "played": bool(r.played)}
            if not r.resolved or r.team1 not in known or r.team2 not in known:
                fixtures_out.append(meta)
                continue

            k = _key(date, r.team1, r.team2)
            entry = ledger.get(k)
            matrix = o_matrix(dm, r.team1, r.team2, r.ground)

            if not r.played:
                probs = E.ensemble_probs(o_probs(dm.outcome_probs, r.team1, r.team2, r.ground),
                                         o_probs(em.outcome_probs, r.team1, r.team2, r.ground), W_DC)
                # honest forecast: the most likely scoreline (mode of the DC grid)
                # plus expected goals (the average). No pool-points optimisation.
                tops = [[f"{i}-{j}", round(p, 3)] for (i, j), p in scoring.top_scorelines(matrix, 10)]
                (mi, mj), mprob = scoring.top_scorelines(matrix, 1)[0]
                xgh, xga = o_xg(dm, r.team1, r.team2, r.ground)
                entry = {"p_home": round(probs["home"], 3), "p_draw": round(probs["draw"], 3),
                         "p_away": round(probs["away"], 3),
                         "model_pick": f"{mi}-{mj}", "model_pick_result": scoring._result(mi, mj),
                         "model_pick_prob": round(float(mprob), 3),
                         "xg_home": round(float(xgh), 2), "xg_away": round(float(xga), 2),
                         "top_scores": tops, "lb": model_eval(r.team1, r.team2, r.ground),
                         "updated_utc": _now(), "played": False, "scored": False}
                ledger[k] = entry
            else:  # played — robust: scores the forecast if present, ALWAYS fills the league
                actual = [int(r.home_score), int(r.away_score)]
                ao = evaluate.result_to_outcome(*actual)
                if entry is None:
                    entry = {}
                if "model_pick" in entry and not entry.get("scored"):
                    msc = scoring.score_pick(_parse(entry["model_pick"]), actual, RESULT_PTS, EXACT_PTS)
                    entry.update(scored=True, model_earned=msc["points"], model_result_hit=msc["result_hit"],
                                 model_exact_hit=msc["exact_hit"], max_points=msc["max_points"],
                                 rps=round(evaluate.rps([entry["p_home"], entry["p_draw"], entry["p_away"]], ao), 4))
                lb = entry.get("lb") or model_eval(r.team1, r.team2, r.ground)   # frozen probs, else post-hoc
                entry["lb"] = lb
                entry["lb_scored"] = score_models(lb, actual)          # always recompute (corrects old rows)
                if "model_pick" not in entry:
                    entry["no_prematch_pick"] = True
                entry.update(played=True, actual=actual, actual_outcome=ao)
                ledger[k] = entry

            out = {**meta, **entry}

            # overlay YOUR pick (from my_picks.json) — scored deterministically
            if k in my_picks:
                yi, yj = _parse(my_picks[k])
                out["your_pick"] = f"{yi}-{yj}"
                out["your_pick_result"] = scoring._result(yi, yj)
                if yi <= 10 and yj <= 10:
                    out["your_pick_prob"] = round(float(matrix[yi, yj]), 3)
                if r.played:
                    ysc = scoring.score_pick((yi, yj), [int(r.home_score), int(r.away_score)],
                                             RESULT_PTS, EXACT_PTS)
                    out.update(your_earned=ysc["points"], your_result_hit=ysc["result_hit"],
                               your_exact_hit=ysc["exact_hit"])
            fixtures_out.append(out)
    else:
        print("  WARN no worldcup2026.json — run scripts/update_data.py first.")

    dump_json(ledger, PICKS)

    scored = [f for f in fixtures_out if f.get("scored")]
    rps_games = [f for f in scored if "rps" in f]
    your = [f for f in fixtures_out if "your_earned" in f]
    summary = {
        "model_scored": len(scored),
        "model_points": sum(f.get("model_earned", 0) for f in scored),
        "live_rps": round(sum(f["rps"] for f in rps_games) / len(rps_games), 4) if rps_games else None,
        "your_scored": len(your),
        "your_points": sum(f["your_earned"] for f in your),
        "your_max": (RESULT_PTS + EXACT_PTS) * len(your),
        "your_result_hits": sum(1 for f in your if f["your_result_hit"]),
        "your_exact_hits": sum(1 for f in your if f["your_exact_hit"]),
    }

    # rating movement vs ~1 year ago (drives the up/down chevrons)
    cutoff = full["date"].max() - pd.Timedelta(days=365)
    past, _ = E.compute_ratings(full[full["date"] < cutoff])
    ratings = []
    for t in sorted(known, key=lambda x: -em.rating(x)):
        now = em.rating(t); p = past.get(t)
        ratings.append({"team": t, "elo": round(now),
                        "delta": (round(now - p) if p is not None else None)})

    # recent form (last 5) for the teams actually in the tournament
    wc_teams = {x for f in fixtures_out if f.get("resolved")
                for x in (f.get("team1"), f.get("team2")) if x in known}
    fsort = full.sort_values("date")
    form = {}
    for t in wc_teams:
        sub = fsort[(fsort["home_team"] == t) | (fsort["away_team"] == t)].tail(5)
        seq = []
        for mrow in sub.itertuples():
            if mrow.home_team == t:
                gf, ga, opp = int(mrow.home_score), int(mrow.away_score), mrow.away_team
            else:
                gf, ga, opp = int(mrow.away_score), int(mrow.home_score), mrow.home_team
            seq.append({"r": "W" if gf > ga else "L" if gf < ga else "D",
                        "s": f"{gf}-{ga}", "opp": opp})
        form[t] = seq

    # Monte-Carlo tournament odds (needs the bracket: groups + knockout slots)
    tournament = {}
    tournament_by_model = {}
    if os.path.exists(fpath):
        try:
            groups, group_games, ko_games = sim.parse_structure(fpath)
            sim_teams = [t for ts in groups.values() for t in ts if t in known]
            full_groups = sum(1 for ts in groups.values() if len([t for t in ts if t in known]) == 4)
            if full_groups == 12 and ko_games:
                padv = lambda o: o["home"] + 0.5 * o["draw"]
                def elo_eg(a, b):           # Elo has no goals model: derive a crude scoreline rate
                    s = (em.rating(a) - em.rating(b)) / 200.0
                    return max(0.15, (2.6 + s) / 2.0)
                model_fns = {
                    "ensemble": (lambda a, b: dm.expected_goals(a, b, neutral=True)[0],
                                 lambda a, b: padv(E.ensemble_probs(dm.outcome_probs(a, b, neutral=True),
                                                                    em.outcome_probs(a, b, neutral=True), W_DC))),
                    "dixon_coles": (lambda a, b: dm.expected_goals(a, b, neutral=True)[0],
                                    lambda a, b: padv(dm.outcome_probs(a, b, neutral=True))),
                    "elo": (elo_eg, lambda a, b: padv(em.outcome_probs(a, b, neutral=True))),
                }
                # Poisson stays in the League but not the title odds: its statsmodels
                # backend makes a 48x48 table far too slow to simulate.
                by_model = {}
                for mname in [m for m in LB_MODELS if m in model_fns]:
                    egf, pvf = model_fns[mname]
                    EG, PADV = sim.build_tables_for(sim_teams, egf, pvf)
                    nsim = 15000 if mname == "ensemble" else 8000
                    odds = sim.simulate(sim_teams, groups, group_games, ko_games, EG, PADV, n_sims=nsim, seed=1)
                    by_model[mname] = {t: {k: round(v, 4) for k, v in s.items()} for t, s in odds.items()}
                tournament = by_model["ensemble"]
                tournament_by_model = by_model
                print(f"  tournament sim: {len(by_model)} models, "
                      f"ensemble favourite { max(tournament, key=lambda x: tournament[x]['champion']) }")
            else:
                print(f"  tournament sim skipped (only {full_groups}/12 full groups)")
        except Exception as e:
            print(f"  tournament sim skipped: {e}")

    # model leaderboard: correct results, correct scorelines, and RPS over played games
    lb_src = sorted([f for f in fixtures_out if f.get("lb_scored")],
                    key=lambda f: (f.get("kickoff_sgt") or f.get("date") or ""))
    res = {m: 0 for m in LB_MODELS}; exa = {m: 0 for m in LB_MODELS}; rsum = {m: 0.0 for m in LB_MODELS}
    nm = {m: 0 for m in LB_MODELS}                       # per-model game count (Market joins late)
    cum_rps = {m: [] for m in LB_MODELS}
    lb_games = []
    for f in lb_src:
        sc = f["lb_scored"]
        for m in LB_MODELS:
            if m not in sc:                              # model didn't cover this game (e.g. Market pre-coverage)
                continue
            nm[m] += 1; s = sc[m]
            res[m] += 1 if s.get("result_hit") else 0
            exa[m] += 1 if s.get("exact_hit") else 0
            rsum[m] += s.get("rps", 0.0); cum_rps[m].append(round(rsum[m] / nm[m], 4))
        lb_games.append({"date": f.get("date"), "label": f"{f['team1']} v {f['team2']}",
                         "key": _key(f.get("date"), f["team1"], f["team2"]),
                         "actual": f"{f['actual'][0]}-{f['actual'][1]}",
                         "lb": f.get("lb", {}), "scored": sc})
    ng = len(lb_src)
    standings = {m: {"result_hits": res[m], "exact_hits": exa[m], "games": nm[m],
                     "rps": round(rsum[m] / nm[m], 4) if nm[m] else None} for m in LB_MODELS}
    leaderboard = {"models": LB_MODELS, "labels": ["Poisson", "Dixon–Coles", "Elo", "Ensemble", "Market"],
                   "games": lb_games, "cum_rps": cum_rps, "standings": standings}

    payload = {
        "generated_utc": _now(),
        "model": f"Dixon-Coles (xi={XI}) + Elo ensemble (w_DC={W_DC}); host advantage for USA/Mexico/Canada",
        "scoring": {"result_pts": RESULT_PTS, "exact_pts": EXACT_PTS},
        "backtest": backtest_rps(rec, full),
        "calibration": calibration_backtest(rec, full),
        "odds": ({"credits_remaining": odds_meta.get("credits_remaining"),
                  "fetched_utc": odds_meta.get("fetched_utc"), "priced": len(odds_list)} if odds_meta else None),
        "summary": summary,
        "n_fixtures": len(fixtures_out),
        "fixtures": fixtures_out,
        "ratings": ratings,
        "form": form,
        "tournament": tournament,
        "tournament_by_model": tournament_by_model,
        "leaderboard": leaderboard,
    }
    dump_json(payload, OUT)
    print(f"wrote {OUT}")
    print(f"  fixtures {len(fixtures_out)} | you {summary['your_points']}/{summary['your_max']} "
          f"({summary['your_result_hits']}R {summary['your_exact_hits']}E) "
          f"| model {summary['model_points']} over {summary['model_scored']} | live RPS {summary['live_rps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
