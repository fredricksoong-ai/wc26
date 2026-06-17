#!/usr/bin/env python3
"""Kill-switch: does xG out-predict plain goals on past World Cups?

Runs on StatsBomb's free open data (no scraping, no API key). Manual GitHub
Actions job — trigger it, then read the log. Two tests:
  1. Repeatability — does a team's prior-match xG predict its NEXT-match goals
     better than its prior-match goals? (the classic "xG is more signal" test)
  2. Outcome RPS — build the same crude strength model from goals-rates vs
     xG-rates and compare out-of-sample 1X2 RPS.
If xG clearly wins both, integrating live xG is justified; if not, we stop.
"""
import sys
import numpy as np
import pandas as pd


def main() -> int:
    try:
        from statsbombpy import sb
    except Exception as e:
        print("statsbombpy import failed:", e)
        return 1

    comps = sb.competitions()
    intl = comps[comps["competition_name"] == "FIFA World Cup"]   # bounded + directly relevant
    if intl.empty:
        print("No FIFA World Cup in open data. Available:")
        print(comps[["competition_id", "season_id", "competition_name", "season_name"]].to_string(index=False))
        return 1
    print("Competitions used:")
    print(intl[["competition_id", "season_id", "competition_name", "season_name"]].to_string(index=False))

    mparts = []
    for r in intl.itertuples():
        try:
            mparts.append(sb.matches(competition_id=r.competition_id, season_id=r.season_id))
        except Exception as e:
            print(f"  matches fail {r.season_name}: {e}")
    matches = pd.concat(mparts, ignore_index=True)
    print(f"\n{len(matches)} matches found. Fetching events for xG (this is the slow part)...")

    recs = []
    for i, m in enumerate(matches.itertuples(), 1):
        try:
            ev = sb.events(match_id=m.match_id)
            if "shot_statsbomb_xg" not in ev.columns:
                continue
            sh = ev[ev["type"] == "Shot"]
            recs.append({
                "date": m.match_date, "home": m.home_team, "away": m.away_team,
                "gh": int(m.home_score), "ga": int(m.away_score),
                "xgh": float(sh[sh["team"] == m.home_team]["shot_statsbomb_xg"].sum()),
                "xga": float(sh[sh["team"] == m.away_team]["shot_statsbomb_xg"].sum()),
            })
        except Exception:
            pass
        if i % 25 == 0:
            print(f"  {i}/{len(matches)}", flush=True)

    df = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
    print(f"\n{len(df)} matches with xG.\n")
    if len(df) < 30:
        print("Too few matches with xG to conclude.")
        return 0

    # long form: one row per team-match
    home_long = df[["date", "home", "gh", "xgh"]].rename(columns={"home": "team", "gh": "gf", "xgh": "xgf"})
    away_long = df[["date", "away", "ga", "xga"]].rename(columns={"away": "team", "ga": "gf", "xga": "xgf"})
    long = pd.concat([home_long, away_long]).sort_values("date")

    # --- Test 1: predict NEXT-match goals from prior goals vs prior xG ---
    pr = []
    for _, g in long.groupby("team"):
        g = g.reset_index(drop=True)
        for k in range(len(g) - 1):
            pr.append((g["gf"].iloc[k], g["xgf"].iloc[k], g["gf"].iloc[k + 1]))
    pairs = pd.DataFrame(pr, columns=["g_t", "xg_t", "g_next"])
    if len(pairs) > 20:
        def mae(x, y):
            b = np.polyfit(x, y, 1)
            return float(np.mean(np.abs(y - np.polyval(b, x))))
        cg, cx = pairs["g_t"].corr(pairs["g_next"]), pairs["xg_t"].corr(pairs["g_next"])
        mg, mx = mae(pairs["g_t"], pairs["g_next"]), mae(pairs["xg_t"], pairs["g_next"])
        print(f"=== Test 1: predict a team's NEXT-match goals  (n pairs = {len(pairs)}) ===")
        print(f"  from prior GOALS : corr {cg:+.3f} | MAE {mg:.3f}")
        print(f"  from prior xG    : corr {cx:+.3f} | MAE {mx:.3f}")
        print(f"  -> xG the better predictor? {'YES' if (cx > cg and mx < mg) else 'mixed / no'}\n")

    # --- Test 2: 1X2 RPS, goals-rates vs xG-rates (same crude model, different input) ---
    try:
        from scipy.stats import poisson
        cut = int(len(df) * 0.6)
        tr, te = df.iloc[:cut], df.iloc[cut:]

        def rates(frame, fh, fa):
            t = pd.concat([
                frame.rename(columns={"home": "t", "away": "o", fh: "f", fa: "a"})[["t", "o", "f", "a"]],
                frame.rename(columns={"away": "t", "home": "o", fa: "f", fh: "a"})[["t", "o", "f", "a"]],
            ])
            return t.groupby("t")["f"].mean(), t.groupby("t")["a"].mean(), t["f"].mean()

        def rps(fh, fa):
            att, dfn, lg = rates(tr, fh, fa)
            P, Y = [], []
            for m in te.itertuples():
                if m.home not in att.index or m.away not in att.index:
                    continue
                lh = max(0.1, att[m.home] * dfn[m.away] / lg)
                la = max(0.1, att[m.away] * dfn[m.home] / lg)
                ph, pa = poisson.pmf(np.arange(11), lh), poisson.pmf(np.arange(11), la)
                M = np.outer(ph, pa)
                h, d, a = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
                s = h + d + a
                P.append([h / s, d / s, a / s])
                Y.append(0 if m.gh > m.ga else 2 if m.gh < m.ga else 1)
            P, Y = np.array(P), np.array(Y)
            cp, co = np.cumsum(P, 1), np.cumsum(np.eye(3)[Y], 1)
            return len(Y), float((np.sum((cp - co) ** 2, 1) / 2).mean())

        ng, rg = rps("gh", "ga")
        _, rx = rps("xgh", "xga")
        print(f"=== Test 2: out-of-sample 1X2 RPS (test n = {ng}) ===")
        print(f"  goals-rates RPS {rg:.4f}")
        print(f"  xG-rates    RPS {rx:.4f}")
        print(f"  -> xG better? {'YES' if rx < rg else 'NO'}")
    except Exception as e:
        print("Test 2 failed:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
