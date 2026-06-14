"""Bookmaker odds from The Odds API -> de-vigged market probabilities.

The market (especially a sharp, overround-removed consensus) is the strongest
single 1X2 baseline in football. We pull h2h (1X2) odds for the World Cup, turn
each bookmaker's prices into implied probabilities, remove the overround (the
bookmaker's margin) by normalising, then average across bookmakers for a
consensus. Team names are normalised to the model's spelling.

Cost: one request = 1 credit (1 market x 1 region) and returns every open match.
Free tier is 500 credits/month — comfortably enough. No historical endpoint.
"""
from __future__ import annotations

import statistics as st

import requests

from .fixtures import normalize_team

API = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/"


def fetch_odds(api_key: str, regions: str = "eu", timeout: int = 30):
    """Return (events, credits_remaining). Raises on HTTP error."""
    r = requests.get(API, params={"regions": regions, "markets": "h2h",
                                   "oddsFormat": "decimal", "apiKey": api_key}, timeout=timeout)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def _devig(home_p, draw_p, away_p):
    """Decimal prices -> overround-removed probabilities (home, draw, away)."""
    inv = [1.0 / p for p in (home_p, draw_p, away_p) if p and p > 1]
    if len(inv) != 3:
        return None
    s = sum(inv)
    return [v / s for v in inv]


def consensus(event: dict):
    """Average de-vigged 1X2 probs across all bookmakers for one event."""
    home, away = event.get("home_team"), event.get("away_team")
    hs, ds, as_ = [], [], []
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            price = {o.get("name"): o.get("price") for o in mkt.get("outcomes", [])}
            if home in price and away in price and "Draw" in price:
                dv = _devig(price[home], price["Draw"], price[away])
                if dv:
                    hs.append(dv[0]); ds.append(dv[1]); as_.append(dv[2])
    if not hs:
        return None
    return {"p_home": round(st.mean(hs), 4), "p_draw": round(st.mean(ds), 4),
            "p_away": round(st.mean(as_), 4), "books": len(hs)}


def parse(events: list) -> list:
    """Events -> [{home, away, commence, p_home, p_draw, p_away, books}] (model names)."""
    out = []
    for e in events:
        c = consensus(e)
        if not c:
            continue
        out.append({"home": normalize_team(e.get("home_team", "")),
                    "away": normalize_team(e.get("away_team", "")),
                    "commence": e.get("commence_time"), **c})
    return out
