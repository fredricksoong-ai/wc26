#!/usr/bin/env python3
"""Pull the latest open data into data/raw/ — no API key required.

Sources (both public domain, both updated ~once a day by hand upstream):
  * martj42/international_results  -> results.csv (+ shootouts, goalscorers, former_names)
      training history AND live WC2026 results as they're played.
  * openfootball/worldcup.json     -> 2026/worldcup.json
      the WC2026 fixture list (schedule, groups, venues) to predict on.

Measured match xG (deploy/xg.json) is NOT fetched here — RealGM's tracker blocks
the Actions runner's IP, so it's folded in from a Cowork session (scripts/fold_xg.py).

Run daily (see scripts/update_data.sh + the launchd plist). Idempotent:
downloads to a temp file then atomically replaces, so a failed fetch never
corrupts the existing copy.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

import requests

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ)
RAW_DIR = os.path.join(PROJ, "data", "raw")

MARTJ42 = "https://raw.githubusercontent.com/martj42/international_results/master"
OPENFOOTBALL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master"

FILES = {
    "results.csv": f"{MARTJ42}/results.csv",
    "shootouts.csv": f"{MARTJ42}/shootouts.csv",
    "goalscorers.csv": f"{MARTJ42}/goalscorers.csv",
    "former_names.csv": f"{MARTJ42}/former_names.csv",
    "worldcup2026.json": f"{OPENFOOTBALL}/2026/worldcup.json",
}
# Only these two actually drive the model + fixtures. A transient failure on any other
# file (GitHub rate-limits happen) must NOT sink the deploy — the atomic download keeps
# the previous copy, so we warn and carry on.
CRITICAL = {"results.csv", "worldcup2026.json"}
_UA = {"User-Agent": "wc26-updater (github-actions)"}

# Odds are fetched on a fixed schedule — 6pm and 10pm Singapore time — plus a 12h
# safety floor. The pipeline carries deploy/odds.json forward from the live site
# each run (the bot never commits), so we can tell when it was last fetched and
# only spend a credit on the first run after each target window passes.
SGT = timezone(timedelta(hours=8))
ODDS_TARGETS_SGT = (18, 22)          # 6pm and 10pm Singapore time
ODDS_FLOOR_H = 24                    # only refetch off-schedule if a whole day was missed
ODDS_FILE = os.path.join(PROJ, "deploy", "odds.json")


def _last_odds_fetch(path: str):
    """Datetime of the last odds fetch from the carried-forward file, or None."""
    try:
        meta = json.load(open(path))
        return datetime.strptime(meta["fetched_utc"], "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _should_fetch_odds(path: str):
    """(should_fetch, reason). Forced by a manual refresh; else 6pm/10pm SGT or stale."""
    if os.environ.get("FORCE_ODDS"):            # the ↻ button (manual workflow_dispatch)
        return True, "manual refresh — forced"
    now = datetime.now(timezone.utc)
    last = _last_odds_fetch(path)
    if last is None:
        return True, "no prior snapshot"
    age_h = (now - last).total_seconds() / 3600.0
    if age_h >= ODDS_FLOOR_H:
        return True, f"{age_h:.1f}h old (>{ODDS_FLOOR_H}h floor)"
    now_sgt = now.astimezone(SGT)
    passed = []
    for h in ODDS_TARGETS_SGT:                       # most recent 6pm/10pm SGT that has passed
        t = now_sgt.replace(hour=h, minute=0, second=0, microsecond=0)
        passed += [t, t - timedelta(days=1)]
    last_target = max(t for t in passed if t <= now_sgt)
    if last.astimezone(SGT) < last_target:
        return True, f"scheduled {last_target.strftime('%H:%M')} SGT window"
    return False, f"{age_h:.1f}h old, already fetched since the last window"


def download(url: str, dest: str, timeout: int = 60) -> int:
    """Stream `url` to `dest` atomically. Returns bytes written. Retries transient
    errors (GitHub raw occasionally 429s / 5xx) with a short backoff."""
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout, headers=_UA)
            resp.raise_for_status()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))     # 2s, 4s
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(resp.content)
        os.replace(tmp, dest)  # atomic on the same filesystem
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return len(resp.content)


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{stamp}] updating data in {RAW_DIR}")
    failures = []
    for name, url in FILES.items():
        dest = os.path.join(RAW_DIR, name)
        try:
            n = download(url, dest)
            print(f"  ok   {name:22} {n:>10,} bytes")
        except Exception as e:  # keep going; one bad source shouldn't sink the rest
            failures.append(name)
            print(f"  FAIL {name:22} {e}")

    # bookmaker odds (optional): only if a key is configured. Fail-soft — a missing
    # or rate-limited odds feed must never sink the run; the model works without it.
    key = os.environ.get("ODDS_API_KEY")
    if key:
        do_fetch, why = _should_fetch_odds(ODDS_FILE)
        if not do_fetch:
            print(f"  odds.json reused ({why}) — no credit spent")
        else:
            try:
                from src import odds as O
                events, remaining = O.fetch_odds(key)
                parsed = O.parse(events)
                payload = {"fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                           "credits_remaining": int(remaining) if str(remaining or "").isdigit() else None,
                           "matches": parsed}
                os.makedirs(os.path.dirname(ODDS_FILE), exist_ok=True)
                with open(ODDS_FILE, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"  ok   odds.json [{why}]  {len(parsed):>3} matches priced | {remaining} credits left")
            except Exception as e:
                print(f"  WARN odds skipped: {e}")
    else:
        print("  (no ODDS_API_KEY set — skipping bookmaker odds)")

    # quick sanity line on the headline file
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(RAW_DIR, "results.csv"), parse_dates=["date"])
        print(f"  results.csv now: {len(df):,} matches, latest {df['date'].max().date()}")
    except Exception as e:
        print(f"  (could not summarise results.csv: {e})")

    crit = [f for f in failures if f in CRITICAL]
    if crit:
        print(f"DONE — CRITICAL failure(s) after retries: {crit}")
        return 1
    if failures:  # non-critical: kept the prior copy, deploy proceeds
        print(f"DONE — {len(failures)} non-critical failure(s) (kept prior copy): {failures}")
        return 0
    print("DONE — all sources updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
