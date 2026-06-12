# Architecture & Infrastructure

The whole thing is **key-free** and splits into three layers: **data → model → view.**
Same shape as your FPL project (scheduled job → JSON → static dashboard), minus the API secret.

```
                        ┌──────────────────────── DATA (public, no key) ───────────────────────┐
   martj42/international_results ──results.csv──┐          openfootball/worldcup.json ──2026/worldcup.json──┐
   (history + live results, ~daily)             │          (fixture list, ~daily)                          │
                                                ▼                                                          ▼
                                       scripts/update_data.py  ──────────────►  data/raw/*.csv , worldcup2026.json
                                                                                          │
                        ┌──────────────────────── MODEL (Python) ──────────────────────────┐ │
                        │  src/data.py        load / filter / weight                        │ │
                        │  src/poisson.py     rung 1  (attack/defence GLM)                   │◄┘
                        │  src/dixon_coles.py rung 2  (low-score fix + time decay)           │
                        │  src/elo.py         rung 3  (ratings prior + calibrated 1X2)       │
                        │  src/evaluate.py    RPS / Brier / log-loss                         │
                        │  src/fixtures.py    map fixtures → teams, predict                  │
                        └───────────────────────────────┬───────────────────────────────────┘
                                                        ▼
                                          scripts/run_predictions.py
                                       (fits DC+Elo ensemble, predicts)
                                                        ▼
                                            deploy/predictions.json   ◄── the single hand-off file
                                                        ▼
                        ┌──────────────────────── VIEW (static, phone) ─────────────────────┐
                        │  deploy/index.html   fetches predictions.json, renders fixtures,   │
                        │                      win-prob bars, power ratings, model RPS       │
                        └───────────────────────────────────────────────────────────────────┘

   AUTOMATION (pick one or both):
     • Local:  launchd  scripts/com.fredricksoong.wc-data-pull.plist   → keeps data/raw fresh for the notebooks
     • Hosted: GitHub Action .github/workflows/predict.yml             → re-runs data+model, commits predictions.json
                                                        ▼
                                      Cloudflare Pages (watches the repo)
                                       auto-deploys deploy/  → https://your-site.pages.dev  (open on phone)
```

## The contract between layers

Everything downstream depends on **one file**: `deploy/predictions.json`. Its shape:

```jsonc
{
  "generated_utc": "2026-06-12 16:31 UTC",
  "model": "Dixon–Coles (ξ=0.001) + Elo ensemble (w_DC=0.6), neutral venues",
  "metrics": { "ensemble_rps": 0.1675, "baseline_rps": 0.2282, "n_test": 1518 },
  "n_fixtures": 8,
  "fixtures": [
    { "date":"2026-06-11", "group":"Group A", "team1":"Mexico", "team2":"South Africa",
      "resolved":true, "played":false,
      "p_home":0.666, "p_draw":0.228, "p_away":0.105, "pred_score":"1-0" }
  ],
  "ratings": [ { "team":"Spain", "elo":2211 } ]
}
```

The dashboard reads only this. Swap the model internals freely — as long as `run_predictions.py`
keeps emitting this shape, the view never changes.

## Why no API key (vs FPL)

FPL needed a secret for the AI-insights call. Here every source is public-domain GitHub data,
so the GitHub Action needs **no secrets** and the dashboard fetches a static JSON. Simpler and
nothing to leak.

---

# What YOU need to do

### A. Just want it on your phone (hosted) — the FPL path
1. **Make a GitHub repo** and push this folder to it.
   `deploy/predictions.json` is already generated and committed-ready.
2. **Connect Cloudflare Pages** to the repo (same as FPL):
   - New Pages project → pick the repo → **Build command:** *(none)* → **Output directory:** `deploy`.
   - Deploy. You get `https://<project>.pages.dev`. Open it on your phone, add to Home Screen.
3. **Turn on the Action:** it's at `.github/workflows/predict.yml`. In the repo → Actions tab → enable.
   It runs twice daily, refreshes data + predictions, commits the JSON, Cloudflare redeploys. Done.

### B. Just want fresh data locally for the notebooks
```bash
cp scripts/com.fredricksoong.wc-data-pull.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.fredricksoong.wc-data-pull.plist
launchctl start com.fredricksoong.wc-data-pull        # test it now
```
Then work through `notebooks/01→02→03`.

### C. Run the whole thing once by hand (no automation)
```bash
python scripts/update_data.py        # pull latest results + fixtures
python scripts/run_predictions.py     # writes deploy/predictions.json
open deploy/index.html                # view locally in a browser
```

### One thing to watch
If a fixture team name from openfootball doesn't match the model's team names,
`run_predictions.py` prints `WARN unmatched fixture team names: [...]`.
Add those to `ALIASES` in `src/fixtures.py` (e.g. `"Türkiye": "Turkey"`). That's the only
manual upkeep, and it tells you exactly what to add.
