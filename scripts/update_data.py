#!/usr/bin/env python3
"""Pull the latest open data into data/raw/ — no API key required.

Sources (both public domain, both updated ~once a day by hand upstream):
  * martj42/international_results  -> results.csv (+ shootouts, goalscorers, former_names)
      training history AND live WC2026 results as they're played.
  * openfootball/worldcup.json     -> 2026/worldcup.json
      the WC2026 fixture list (schedule, groups, venues) to predict on.

Run daily (see scripts/update_data.sh + the launchd plist). Idempotent:
downloads to a temp file then atomically replaces, so a failed fetch never
corrupts the existing copy.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone

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


def download(url: str, dest: str, timeout: int = 60) -> int:
    """Stream `url` to `dest` atomically. Returns bytes written."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
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
        try:
            from src import odds as O
            events, remaining = O.fetch_odds(key)
            parsed = O.parse(events)
            with open(os.path.join(RAW_DIR, "odds.json"), "w") as f:
                json.dump(parsed, f, indent=2)
            print(f"  ok   odds.json            {len(parsed):>3} matches priced | {remaining} credits left")
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

    if failures:
        print(f"DONE with {len(failures)} failure(s): {failures}")
        return 1
    print("DONE — all sources updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
