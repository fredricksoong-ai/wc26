#!/usr/bin/env python3
"""Pull the latest open data into data/raw/ — no API key required.

Sources (both public domain, both updated ~once a day by hand upstream):
  * martj42/international_results  -> results.csv (+ shootouts, goalscorers, former_names)
      training history AND live WC2026 results as they're played.
  * openfootball/worldcup.json     -> 2026/worldcup.json
      the WC2026 fixture list (schedule, groups, venues) to predict on.
  * RealGM xG tracker (optional)   -> deploy/xg.json
      measured match xG, scraped fail-soft; feeds the dashboard's xG page.

Run daily (see scripts/update_data.sh + the launchd plist). Idempotent:
downloads to a temp file then atomically replaces, so a failed fetch never
corrupts the existing copy.
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import sys
import tempfile
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

# Odds are fetched on a fixed schedule — 6pm and 10pm Singapore time — plus a 12h
# safety floor. The pipeline carries deploy/odds.json forward from the live site
# each run (the bot never commits), so we can tell when it was last fetched and
# only spend a credit on the first run after each target window passes.
SGT = timezone(timedelta(hours=8))
ODDS_TARGETS_SGT = (18, 22)          # 6pm and 10pm Singapore time
ODDS_FLOOR_H = 24                    # only refetch off-schedule if a whole day was missed
ODDS_FILE = os.path.join(PROJ, "deploy", "odds.json")

# Measured match xG, scraped fail-soft from RealGM's hand-maintained WC2026 tracker
# (clean static HTML, no key, no anti-bot). Feeds deploy/xg.json -> the xG page.
XG_URL = ("https://soccer.realgm.com/analysis/559/"
          "2026-FIFA-World-Cup-xG-Tracker-Results-Expected-Goals-Of-Every-Match")
XG_FILE = os.path.join(PROJ, "deploy", "xg.json")
_UA = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")}


def _xg_side(s):
    """('Team', xg) from one side of a RealGM line, tolerating its format quirks."""
    s = s.replace("**", "")
    m = re.search(r"\d+\.\d+", s)
    if not m:
        return None, None
    name = re.sub(r"\d+\.?\d*", "", s)        # drop xG + goal numbers
    name = re.sub(r"\(.*?\)", "", name)       # drop (goals) / (Group X)
    name = name.replace("xG", "")
    name = re.sub(r"\s+", " ", name).strip(" :)(")
    return name, float(m.group())


def _parse_xg(text: str):
    """Tag-stripped RealGM text -> {frozenset(pair): (date, {team: xg})}, names spec-renamed.
    A pair plays at most once all tournament, so the unordered pair is a safe match key —
    robust to RealGM dating a game a day off from our fixtures."""
    from src.fixtures import fold
    spec = {fold("Türkiye"): "Turkey", fold("Côte d'Ivoire"): "Ivory Coast",
            fold("Czechia"): "Czech Republic", fold("Congo DR"): "DR Congo"}
    ren = lambda n: spec.get(fold(n), n)
    out = {}
    for line in text.splitlines():
        md = re.search(r"\b(June|July)\s+(\d{1,2})\b", line)
        if not md or " vs. " not in line:
            continue
        date = f"2026-{6 if md.group(1)=='June' else 7:02d}-{int(md.group(2)):02d}"
        left, right = line.split(" vs. ", 1)
        n1, x1 = _xg_side(re.split(r":\s+", left)[-1])     # drop the 'June DD (Group X): ' prefix
        n2, x2 = _xg_side(right)
        if x1 is None or x2 is None or not n1 or not n2:
            continue
        n1, n2 = ren(n1), ren(n2)
        out[frozenset((n1, n2))] = (date, {n1: x1, n2: x2})
    return out


def _merge_xg(xg: dict, lut: dict):
    """Merge a pair-keyed lut into xg in place. Existing games are matched by PAIR (so a
    date-label mismatch updates in place instead of duplicating); new ones are appended."""
    from src.fixtures import fold
    canon = {t for k in xg for t in k.split("|")[1:]}
    foldidx = {fold(t): t for t in canon}
    norm = lambda n: n if n in canon else foldidx.get(fold(n), n)
    lutn = {}                                              # snap names onto our exact spellings
    for pair, (date, rec) in lut.items():
        nrec = {norm(t): v for t, v in rec.items()}
        lutn[frozenset(nrec)] = (date, nrec)
    updated = added = 0
    have = set()
    for key in list(xg):
        d, t1, t2 = key.split("|")
        pair = frozenset((t1, t2)); have.add(pair)
        rec = lutn.get(pair, (None, {}))[1]
        if t1 in rec and t2 in rec:
            nv = [round(rec[t1], 2), round(rec[t2], 2)]
            if nv != xg[key]:
                xg[key] = nv
                updated += 1
    for pair, (d, rec) in lutn.items():
        if pair in have:
            continue
        (a, x_a), (b, x_b) = list(rec.items())
        xg[f"{d}|{a}|{b}"] = [round(x_a, 2), round(x_b, 2)]
        added += 1
    return updated, added


def update_xg_from_realgm(xg_path: str):
    """Refresh the committed xg.json from RealGM. Returns (updated, added). Raises on
    fetch/parse failure — the caller keeps it fail-soft so a bad scrape never sinks the run."""
    resp = requests.get(XG_URL, headers=_UA, timeout=30)
    resp.raise_for_status()
    text = re.sub(r"(?i)<br\s*/?>", "\n", resp.text)
    text = re.sub(r"(?i)</(p|div|li|tr|h2|h3)>", "\n", text)
    text = _html.unescape(re.sub(r"<[^>]+>", "", text))
    lut = _parse_xg(text)
    if not lut:
        raise ValueError("parsed 0 matches — RealGM layout may have changed")
    xg = json.load(open(xg_path)) if os.path.exists(xg_path) else {}
    updated, added = _merge_xg(xg, lut)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(xg_path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(xg, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, xg_path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return updated, added


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

    # measured match xG (optional, fail-soft): refresh deploy/xg.json from RealGM's
    # tracker. A bad scrape (site down, layout change) leaves the existing file as-is.
    try:
        up, add = update_xg_from_realgm(XG_FILE)
        print(f"  ok   xg.json   RealGM xG: {up} updated, {add} added")
    except Exception as e:
        print(f"  WARN xg skipped (kept existing xg.json): {e}")

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
