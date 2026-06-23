#!/usr/bin/env python3
"""xG A/B backtest — is the manually-entered xG improving the Elo forecasts?

The xG layer (Phase B) only touches the Elo rating update. This walks the full
match history forward TWICE — Elo with xG-ON and xG-OFF — and compares the 1X2
RPS on the WC2026 games, on the subset where xG could actually have mattered
(a participating team had already played a prior WC game with measured xG).

The gap->1X2 calibration is fit once and reused for both passes, so the ONLY
difference between them is the ratings the xG produced. Lower RPS = better.

Run from the Actions tab (it pulls fresh results first), then read the log.
"""
import os
import sys
import json

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ)
from src import data, elo as E, evaluate                       # noqa: E402
from src.elo import _k_base, _g_multiplier, _probs_from_gap    # noqa: E402

WC_START = "2026-06-11"
HOME_ADV = 65.0
TAU = 1.0


def attach_xg(full, xg):
    """Add oriented xg_home/xg_away columns; return (rows matched, lookup of WC-xG games)."""
    full["xg_home"] = np.nan
    full["xg_away"] = np.nan
    lut = {}
    for k, v in xg.items():
        if isinstance(v, list) and len(v) == 2:
            p = k.split("|")
            if len(p) == 3:
                lut[(p[0], frozenset((p[1], p[2])))] = (p[1], float(v[0]), float(v[1]))
    n = 0
    if lut:
        ds = full["date"].dt.strftime("%Y-%m-%d")
        for idx in full.index[ds.isin({d for (d, _) in lut})]:
            d = ds.at[idx]
            h, a = full.at[idx, "home_team"], full.at[idx, "away_team"]
            rec = lut.get((d, frozenset((h, a))))
            if rec:
                t1, x1, x2 = rec
                full.at[idx, "xg_home"], full.at[idx, "xg_away"] = (x1, x2) if h == t1 else (x2, x1)
                n += 1
    return n, lut


def walk(df, xgw, theta, s, xg_games):
    """One chronological pass; record pre-match RPS for every WC game.

    `xg_games` = set of (date, frozenset(home,away)) that carry measured xG, used
    (the SAME for both passes) to flag which WC games had prior xG to learn from.
    """
    df = df.sort_values("date").reset_index(drop=True)
    has_xg = "xg_home" in df.columns
    ratings, seen, out = {}, set(), []
    for r in df.itertuples():
        date = r.date.strftime("%Y-%m-%d")
        ra = ratings.get(r.home_team, 1500.0)
        rb = ratings.get(r.away_team, 1500.0)
        neutral = bool(getattr(r, "neutral", False))
        dr = ra - rb + (0.0 if neutral else HOME_ADV)
        if date >= WC_START and isinstance(r.home_team, str) and isinstance(r.away_team, str):
            p = _probs_from_gap(dr, theta, s)[0]
            ao = evaluate.result_to_outcome(int(r.home_score), int(r.away_score))
            out.append({"key": (date, r.home_team, r.away_team),
                        "rps": evaluate.rps([float(p[0]), float(p[1]), float(p[2])], ao),
                        "had_xg": (r.home_team in seen or r.away_team in seen)})
        # rating update — goals, or xG-blended performance when this pass is xG-ON
        if r.home_score > r.away_score:
            sa = 1.0
        elif r.home_score < r.away_score:
            sa = 0.0
        else:
            sa = 0.5
        gmult = _g_multiplier(r.home_score - r.away_score)
        if has_xg and xgw > 0:
            xgh, xga = getattr(r, "xg_home", np.nan), getattr(r, "xg_away", np.nan)
            if xgh == xgh and xga == xga:
                m = float(xgh) - float(xga)
                sa = xgw * (0.5 + 0.5 * np.tanh(m / TAU)) + (1 - xgw) * sa
                gmult = _g_multiplier(round(m))
        exp = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        k = _k_base(getattr(r, "tournament", "")) * gmult
        ratings[r.home_team] = ra + k * (sa - exp)
        ratings[r.away_team] = rb + k * (exp - sa)
        if (date, frozenset((r.home_team, r.away_team))) in xg_games:
            seen.add(r.home_team)
            seen.add(r.away_team)
    return pd.DataFrame(out)


def main():
    full = data.load_results(os.path.join(PROJ, "data", "raw", "results.csv"))
    xgf = os.path.join(PROJ, "deploy", "xg.json")
    xg = json.load(open(xgf)) if os.path.exists(xgf) else {}
    n_rows, lut = attach_xg(full, xg)
    print(f"xG entries: {len(xg)} · matched to {n_rows} history rows")

    base = E.fit(full, xg_weight=0.0)          # gap->1X2 calibration, reused for both passes
    theta, s = base.theta, base.s

    on = walk(full, 1.0, theta, s, set(lut)).set_index("key")
    off = walk(full, 0.0, theta, s, set(lut)).set_index("key")
    m = off.join(on, lsuffix="_off", rsuffix="_on")
    wc = m  # all WC games (matchday-1 games are identical and wash out)
    sub = m[m["had_xg_off"]]
    print(f"\nWC games scored: {len(wc)}  (of which had prior xG info: {len(sub)})")
    if not len(sub):
        print("No WC game yet was predicted using xG — need a team's 2nd/3rd group game.")
        return 0

    diff = sub["rps_off"] - sub["rps_on"]      # +ve => xG-ON lower RPS => xG better
    print(f"  Elo RPS  xG-OFF : {sub['rps_off'].mean():.4f}")
    print(f"  Elo RPS  xG-ON  : {sub['rps_on'].mean():.4f}")
    print(f"  delta (off-on)  : {diff.mean():+.4f}  -> {'xG better' if diff.mean() > 0 else 'xG worse' if diff.mean() < 0 else 'no change'}")
    better = int((diff > 1e-9).sum())
    worse = int((diff < -1e-9).sum())
    same = int((diff.abs() <= 1e-9).sum())
    print(f"  per-game        : xG better {better}, worse {worse}, identical {same}")
    if len(diff) > 1:
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"  significance    : mean delta {diff.mean():+.4f} ± {se:.4f} (1 SE) -> |t| = {abs(diff.mean() / se) if se else 0:.2f}")
        print("                    (|t| under ~2 = within noise, can't claim an effect)")
    print("\nNB: Elo is ~40% of the Ensemble, so any Ensemble-level effect is smaller still.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
