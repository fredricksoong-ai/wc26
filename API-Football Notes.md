# API-Football — Endpoints & Limits Notes

Source: [WC2026 guide](https://www.api-football.com/news/post/fifa-world-cup-2026-guide-to-using-data-with-api-sports) · [Pricing](https://www.api-football.com/pricing) · [Docs v3.9.3](https://www.api-football.com/documentation-v3)
Captured: 2026-06-12. Scope decision: **per-match predictions first**, Monte-Carlo tournament sim later.

---

## TL;DR for the predictor

- Base URL: `https://v3.football.api-sports.io` — auth header `x-apisports-key: YOUR_KEY`.
- Two constants for everything WC: **`league=1`**, **`season=2026`**.
- The schedule is **already live** (all 104 matches appear as they're confirmed). You can build now.
- Free tier = **100 requests/day**, all endpoints included, but **limited available seasons** — this is the catch that matters for training data (see below).

---

## Pricing & free-tier limits

| Plan | Price | Requests/day | Endpoints | Seasons |
|------|-------|--------------|-----------|---------|
| Free | $0 | **100/day** | All | **Limited** (recent only) |
| Pro | $19/mo | 7,500/day | All | All |
| Ultra | $29/mo | 75,000/day | All | All |
| Mega | $39/mo | 150,000/day | All | All |

Notes:
- Hard quota — when you hit the daily cap the API returns an error instead of billing extra. No overage charges.
- Prepaid, no auto-renew; lapses back to free on expiry.
- **Per-minute rate limit is not stated on the pricing page — confirm in the dashboard after signup.** (Free tier typically caps requests/minute too; check before writing any polling loop.)
- **Season restriction is the real free-tier constraint.** All endpoints are technically available on free, but historical seasons are gated. For a Poisson model trained on years of international results this likely means free won't give you deep history → plan to **supplement with Kaggle "International football results 1872–present" + eloratings.net**, and use API-Football mainly for the *live* WC2026 fixtures/results/odds/predictions. Verify exactly which seasons free exposes once the key is active.

### Does the free tier cover the build?
For per-match WC2026 prediction the request budget is tiny: schedule (1 call), teams (1), standings (1), plus per-fixture predictions/odds/H2H on match days. 100/day is plenty **if** training history comes from Kaggle/Elo rather than hammering the API for past seasons. If you later want the API's own historical fixtures for many seasons as training data, you'll need Pro ($19).

---

## Endpoints that matter for this project

All take `league=1&season=2026` unless noted. ★ = core to the predictor.

### Setup / reference
- `GET /leagues?id=1&season=2026` — returns a `coverage` object listing which data types exist for the WC. Check this first; `predictions`, `odds`, `injuries`, `standings` etc. each flagged true/false. Availability still varies match-to-match early on.
- ★ `GET /fixtures?league=1&season=2026` — the schedule, all 104 matches. Each has `fixture.id` (save these — needed by almost every other call), date, UTC time, venue, status. Matches added as tournament progresses.
- ★ `GET /teams?league=1&season=2026` — the 48 teams with `team.id`, name, code, country, logo, home venue. (Save the id→name map.)
- `GET /fixtures/rounds?league=1&season=2026` — stage/round names ("Group Stage - 1", "Round of 32", "Quarter-finals"…). Add `current=true` for the active round only. Useful for the dashboard nav and for filtering knockout vs group.

### Results & live data (model inputs + outcomes for backtest)
- ★ `GET /fixtures?id=FIXTURE_ID` — full match detail; embeds events, lineups, statistics, players.
- Batch up to 20 fixtures: `GET /fixtures?ids=ID1-ID2-ID3`.
- Live: `GET /fixtures?live=all` or filter `status=1H-HT-2H-ET-P-BT-LIVE`. `/fixtures` + `/fixtures/events` refresh every 15s — set cache accordingly for live polling.
- ★ `GET /standings?league=1&season=2026` — all 12 group tables: played, W/D/L, GF, GA, GD, points, form. (Goals for/against per team = direct Poisson input once games are played.)
- `GET /fixtures/headtohead?h2h=TEAM_A-TEAM_B` — full H2H history between two nations (e.g. `h2h=10-3`). Useful as a feature / sanity check.

### Prediction benchmarks (compare your model against these)
- ★ `GET /predictions?fixture=FIXTURE_ID` — API's own model: predicted winner, estimated score, home/draw/away probabilities, based on form + H2H + history. **This is benchmark (b) for evaluation.**
- ★ `GET /odds?fixture=FIXTURE_ID` — pre-match odds (only retrievable within **7 days** of the fixture). Convert to implied probabilities → **benchmark (a), the bookmaker baseline.**
- `GET /odds/live?fixture=FIXTURE_ID` — in-play odds (not needed for pre-match prediction).

### Context features (optional, later)
- `GET /injuries?league=1&season=2026` — injuries/suspensions.
- `GET /fixtures/players?fixture=FIXTURE_ID` — per-player 0–10 ratings per match.
- `GET /players?league=1&season=2026&page=1` — squads/profiles (paged).
- `GET /coachs?team=TEAM_ID` — head coach info.

---

## How this maps to the per-match Poisson plan

1. **Train** attack/defence strengths on historical international results — pull from **Kaggle 1872–present + Elo** (avoids the free-tier season gate), not the API.
2. **Predict** each WC2026 fixture: get the fixture list + team ids from the API (`/fixtures`, `/teams`), run the model → scoreline matrix → 1X2 + most-likely scores.
3. **Benchmark** each prediction against `/odds` (implied probs) and `/predictions` (API's model).
4. **Backtest** on past tournaments using results you already have locally (Kaggle), scoring with RPS / Brier / log-loss.
5. Live results for scoring real WC2026 predictions come from `/fixtures` + `/standings` as matches finish.

Request budget under this plan stays well within 100/day → **free tier is enough to start.** Upgrade to Pro ($19) only if you decide to pull multi-season history *from the API* or need a higher per-minute rate for a polling dashboard.

---

## Signup checklist (you do this)

1. Register: https://dashboard.api-football.com/register (no card needed for free).
2. Copy the API key from the dashboard. **Store it as a GitHub secret**, never in the repo — same pattern as the FPL AI-insights key.
3. In the dashboard: note the **per-minute rate limit** and **exactly which seasons** the free plan exposes (the two things the public pages don't state).
4. Smoke test: `GET /leagues?id=1&season=2026` and read the `coverage` object to confirm `predictions` + `odds` are true for the WC.

---

## Open items carried forward

- Confirm free-tier per-minute rate limit + season coverage (dashboard only).
- Decide training-data source: Kaggle/Elo (free, recommended) vs API multi-season (needs Pro).
- Next session step 2: pull a historical results set, build the basic Poisson model in a notebook, sanity-check on known matches; then add an RPS backtest.
