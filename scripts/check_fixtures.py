#!/usr/bin/env python3
"""Sanity-check the fixture list so you're never trusting bad data.

Run after scripts/update_data.py. It asserts the schedule looks like a real
World Cup and tells you exactly what (if anything) is still unresolved:

    104 matches · 12 groups (A-L) · dates 2026-06-11 → 2026-07-19

Exit code 0 if everything checks out, 1 if something's off — so the GitHub
Action fails loudly rather than publishing a broken schedule.
"""
from __future__ import annotations

import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ)
from src import fixtures as fx  # noqa: E402

EXPECT_MATCHES = 104
EXPECT_GROUPS = {f"Group {c}" for c in "ABCDEFGHIJKL"}
EXPECT_START, EXPECT_END = "2026-06-11", "2026-07-19"


def main() -> int:
    path = os.path.join(PROJ, "data", "raw", "worldcup2026.json")
    if not os.path.exists(path):
        print("FAIL: no worldcup2026.json — run scripts/update_data.py first.")
        return 1

    df = fx.load_fixtures(path)
    problems = []

    n = len(df)
    print(f"matches: {n}  (expect {EXPECT_MATCHES})")
    if n != EXPECT_MATCHES:
        problems.append(f"expected {EXPECT_MATCHES} matches, found {n}")

    groups = set(df["group"].dropna())
    missing = EXPECT_GROUPS - groups
    print(f"groups : {len(EXPECT_GROUPS - missing)}/12 present")
    if missing:
        problems.append(f"missing groups: {sorted(missing)}")

    if not df.empty:
        lo, hi = df["date"].min().date(), df["date"].max().date()
        print(f"dates  : {lo} → {hi}  (expect {EXPECT_START} → {EXPECT_END})")
        if str(lo) != EXPECT_START or str(hi) != EXPECT_END:
            problems.append(f"date span {lo}→{hi} != {EXPECT_START}→{EXPECT_END}")

    resolved = int(df["resolved"].sum())
    played = int(df["played"].sum())
    print(f"resolved: {resolved}/{n} fixtures have both real teams")
    print(f"played  : {played} have a score so far")

    # list the still-unresolved slots (informational — knockouts are expected)
    unresolved = df[~df["resolved"]]
    if len(unresolved):
        print(f"\nstill unresolved ({len(unresolved)} — knockout brackets are normal):")
        for r in unresolved.head(20).itertuples():
            print(f"  {r.round:16} {r.team1}  vs  {r.team2}")

    print()
    if problems:
        print("CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("CHECK OK — schedule looks like a valid World Cup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
