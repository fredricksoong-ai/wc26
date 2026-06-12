"""Thin API-Football wrapper for the WC2026 endpoints we need.

Auth: header `x-apisports-key`. WC2026 is league=1, season=2026.
Free tier = 100 requests/day, so cache aggressively and don't poll in a loop.
See API-Football Notes.md for the full endpoint list and limits.

Usage:
    from src.api_client import ApiFootball
    api = ApiFootball()                      # reads API_FOOTBALL_KEY from .env
    fixtures = api.fixtures()                # all 104 WC2026 matches
    preds = api.predictions(fixture_id)      # API's own model (benchmark b)
    odds = api.odds(fixture_id)              # pre-match odds (benchmark a)
"""
from __future__ import annotations

import os
from typing import Any

import requests

WC_LEAGUE = 1
WC_SEASON = 2026


class ApiFootball:
    def __init__(self, key: str | None = None, base_url: str | None = None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.key = key or os.getenv("API_FOOTBALL_KEY")
        self.base_url = (base_url or os.getenv("API_FOOTBALL_BASE_URL")
                         or "https://v3.football.api-sports.io")
        if not self.key:
            raise RuntimeError(
                "No API key. Copy .env.example to .env and set API_FOOTBALL_KEY, "
                "or pass key=... explicitly."
            )

    def _get(self, path: str, **params) -> dict[str, Any]:
        resp = requests.get(
            f"{self.base_url}/{path.lstrip('/')}",
            headers={"x-apisports-key": self.key},
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            raise RuntimeError(f"API error on /{path}: {data['errors']}")
        return data

    # --- reference -----------------------------------------------------------
    def coverage(self) -> dict:
        """leagues?id=1&season=2026 -> inspect the `coverage` object."""
        return self._get("leagues", id=WC_LEAGUE, season=WC_SEASON)

    def teams(self) -> list[dict]:
        return self._get("teams", league=WC_LEAGUE, season=WC_SEASON)["response"]

    def fixtures(self, **extra) -> list[dict]:
        return self._get("fixtures", league=WC_LEAGUE, season=WC_SEASON, **extra)["response"]

    def standings(self) -> list[dict]:
        return self._get("standings", league=WC_LEAGUE, season=WC_SEASON)["response"]

    def head_to_head(self, team_a: int, team_b: int) -> list[dict]:
        return self._get("fixtures/headtohead", h2h=f"{team_a}-{team_b}")["response"]

    # --- benchmarks ----------------------------------------------------------
    def predictions(self, fixture_id: int) -> list[dict]:
        """API's own model output -- evaluation benchmark (b)."""
        return self._get("predictions", fixture=fixture_id)["response"]

    def odds(self, fixture_id: int) -> list[dict]:
        """Pre-match odds (only within 7 days of kickoff) -- benchmark (a)."""
        return self._get("odds", fixture=fixture_id)["response"]
