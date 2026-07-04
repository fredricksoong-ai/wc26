#!/usr/bin/env python3
"""Fold RealGM xG into deploy/xg.json — the manual half of xG collection.

RealGM's tracker blocks the GitHub Actions runner's IP (403), so this can't run in
CI. Instead it's run from a Cowork session: Claude fetches the tracker page with its
web tool, saves the text to a file, and runs this to parse + merge it in.

    python scripts/fold_xg.py /tmp/realgm.txt      # or pipe the page text on stdin

Matching is by team PAIR (a pair plays at most once all tournament), so it's robust
to RealGM dating a game a day off from our fixtures or listing home/away reversed —
existing games update in place, genuinely new ones are appended. Never deletes.
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import sys
import tempfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ)
XG_FILE = os.path.join(PROJ, "deploy", "xg.json")


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
    """RealGM page text -> {frozenset(pair): (date, {team: xg})}, names spec-renamed."""
    from src.fixtures import fold
    spec = {fold("Türkiye"): "Turkey", fold("Côte d'Ivoire"): "Ivory Coast",
            fold("Czechia"): "Czech Republic", fold("Congo DR"): "DR Congo",
            fold("Bosnia-Herzegovina"): "Bosnia and Herzegovina"}   # RealGM drops the 'and' in KO rounds
    ren = lambda n: spec.get(fold(n), n)
    out = {}
    for line in text.splitlines():
        md = re.search(r"\b(June|July)\s+(\d{1,2})\b", line)
        if not md or " vs. " not in line:
            continue
        date = f"2026-{6 if md.group(1)=='June' else 7:02d}-{int(md.group(2)):02d}"
        left, right = line.split(" vs. ", 1)
        n1, x1 = _xg_side(re.split(r":\s+", left)[-1])     # drop the 'June DD (…): ' prefix
        n2, x2 = _xg_side(right)
        if x1 is None or x2 is None or not n1 or not n2:
            continue
        n1, n2 = ren(n1), ren(n2)
        out[frozenset((n1, n2))] = (date, {n1: x1, n2: x2})
    return out


def _merge_xg(xg: dict, lut: dict):
    """Merge a pair-keyed lut into xg in place. Existing games matched by PAIR (a date-label
    mismatch updates in place, no duplicate); genuinely new ones are appended."""
    from src.fixtures import fold
    canon = {t for k in xg for t in k.split("|")[1:]}
    foldidx = {fold(t): t for t in canon}
    norm = lambda n: n if n in canon else foldidx.get(fold(n), n)
    lutn = {}
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


def main():
    raw = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    text = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    text = re.sub(r"(?i)</(p|div|li|tr|h2|h3)>", "\n", text)
    text = _html.unescape(re.sub(r"<[^>]+>", "", text))       # harmless on markdown too
    lut = _parse_xg(text)
    if not lut:
        print("parsed 0 matches — check the input text"); return 1
    xg = json.load(open(XG_FILE)) if os.path.exists(XG_FILE) else {}
    updated, added = _merge_xg(xg, lut)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(XG_FILE))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(xg, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, XG_FILE)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"parsed {len(lut)} games | updated {updated} | added {added} | total {len(xg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
