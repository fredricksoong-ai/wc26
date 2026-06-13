# World Cup 2026 Score Predictor

A first prediction-modelling project. Goal: **understand the modelling**, not just call a black box.
We climb a deliberate learning ladder, measuring at every rung so "did it get better?" is always answerable.

## The learning ladder

| Rung | Model | What it teaches | Status |
|------|-------|-----------------|--------|
| 1 | **Poisson** | attack/defence strengths → goals as Poisson → full scoreline matrix → 1X2 probs | ← start here |
| 2 | **Dixon–Coles** | low-score correlation fix + time-decay weighting | next |
| 3 | **Strength priors** | blend Elo / FIFA ranking; neutral-venue handling | next |
| 4 | **ML (optional)** | gradient boosting on engineered features — only if it beats the baseline | tested — loses (0.175 vs 0.170), not shipped |

Scope: **per-match predictions first** (win/draw/loss + most-likely scores). A Monte-Carlo tournament sim comes after the per-match model is trustworthy.

## How we know it's working (evaluation)

Backtest on past tournaments (2018/2022 WC, recent Euros) with **probabilistic** scores, not accuracy:

- **RPS** (Ranked Probability Score) — the standard for 1X2 football outcomes. Lower is better.
- **Brier** / **log-loss** — calibration.

Always compared against two benchmarks:
- **(a)** bookmaker odds → implied probabilities (API-Football `/odds`).
- **(b)** API-Football's own `/predictions` endpoint.

If the model can't beat the bookmaker baseline, the rung isn't done.

## Repo layout

```
World Cup 2026 Predictor/
├── README.md
├── API-Football Notes.md      # endpoints, limits, signup checklist
├── requirements.txt
├── .env.example               # copy to .env, add your key (never commit .env)
├── .gitignore
├── data/
│   ├── raw/                    # downloaded datasets (gitignored)
│   └── processed/              # cleaned, model-ready (gitignored)
├── src/
│   ├── data.py                # load & clean historical results
│   ├── poisson.py             # rung 1 — the Poisson model (read this — it's the core)
│   ├── dixon_coles.py         # rung 2 — low-score correction + time decay
│   ├── fixtures.py            # load openfootball WC2026 schedule, predict on it
│   ├── evaluate.py            # RPS, Brier, log-loss
│   └── api_client.py          # (optional, unused) API-Football wrapper
├── scripts/
│   ├── update_data.py         # daily key-free data pull
│   ├── update_data.sh         # launchd wrapper
│   └── com.fredricksoong.wc-data-pull.plist
├── notebooks/
│   ├── 01_poisson_baseline.ipynb   # rung 1
│   └── 02_dixon_coles.ipynb        # rung 2
└── deploy/                    # (later) GH Action + static dashboard, FPL pattern
    └── README.md
```

## Setup

```bash
cd "World Cup 2026 Predictor"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your API-Football key into .env
jupyter lab notebooks/01_poisson_baseline.ipynb
```

## Data sources — fully key-free

No API key needed. Everything comes from public-domain GitHub repos, pulled daily.

| Data | Source | Notes |
|------|--------|-------|
| Training history **and** live WC2026 results | [martj42/international_results](https://github.com/martj42/international_results) `results.csv` | 1872–present, ~49k matches. Updated ~daily upstream. |
| WC2026 fixture list (schedule, groups, venues) | [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) `2026/worldcup.json` | No key. Updated ~daily upstream (by hand). |
| Strength ratings (rung 3) | [eloratings.net](https://www.eloratings.net), FIFA ranking | for the Elo prior. |
| (optional) bookmaker-odds benchmark | [The Odds API](https://the-odds-api.com) free tier | only if you want the market comparison. |

**We dropped API-Football entirely** — its only unique offering was its own `/predictions` benchmark, which we don't need. The real benchmark is the bookmaker (optional) and the real "is it improving?" signal is the RPS backtest.

### Daily auto-update

`scripts/update_data.py` downloads all the above into `data/raw/` (atomic writes, no key). Two ways to run it daily:

**Option A — launchd (matches your stock-pipeline setup):**
```bash
cp "scripts/com.fredricksoong.wc-data-pull.plist" ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.fredricksoong.wc-data-pull.plist
launchctl start com.fredricksoong.wc-data-pull   # run once now to test
```
Runs at 8:30am daily (after the 8:10 stock pipeline). Logs to `data/raw/update.log`. Uses the Anaconda python at `/opt/anaconda3/bin/python3`.

**Option B — run on demand:** `python scripts/update_data.py` whenever you want fresh data.

During the tournament the daily pull keeps both results (for scoring) and the bracket (as it fills in) current.

## Deployment (later, mirrors FPL architecture)

Model script in a repo → GitHub Action runs on a schedule → writes `predictions.json` → static dashboard on Cloudflare Pages reads it and shows fixtures, predicted scores, win probs. **API key stays a GitHub secret** (server-side), same as the FPL AI-insights pattern.

## Next actions

- [ ] Install the daily pull (launchd Option A above), or just run `python scripts/update_data.py` once.
- [x] ~~Rung 1 Poisson~~ and ~~rung 2 Dixon–Coles~~ — done, both backtested.
- [ ] **Rung 3: Elo / FIFA-ranking prior** — the big one for the WC (group teams have thin recent data; a prior stabilises them). Next `src/` module.
- [ ] Wire `src/fixtures.py` predictions into a `predictions.json` for the dashboard.

No API key required at any step.
