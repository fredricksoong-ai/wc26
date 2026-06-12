# Deployment (later)

Mirrors the FPL architecture. Don't build this until the per-match model is trustworthy.

```
model script  ──►  GitHub Action (scheduled)  ──►  predictions.json  ──►  Cloudflare Pages (static dashboard)
```

- **Model script:** runs `src/` to produce `predictions.json` (fixtures, predicted scores, win/draw/loss probs).
- **GitHub Action:** runs the script on a cron (e.g. daily during the tournament). `API_FOOTBALL_KEY` is a **GitHub secret**, injected as an env var — never in the repo. Same pattern as the FPL AI-insights key.
- **Dashboard:** static page on Cloudflare Pages that fetches `predictions.json` and renders it. No server, no exposed key.

Checklist when we get here:
- [ ] `scripts/run_predictions.py` writing `predictions.json`
- [ ] `.github/workflows/predict.yml` with cron + `API_FOOTBALL_KEY` secret
- [ ] static dashboard reading the published JSON
