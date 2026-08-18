"""
Live sports scores via TheSportsDB's free tier (key "3" is the shared
public test key; swap in your own free key from thesportsdb.com for
higher rate limits).
"""
import requests

from app.config import get_settings

settings = get_settings()
BASE = f"https://www.thesportsdb.com/api/v1/json/{settings.SPORTSDB_API_KEY}"


def get_recent_results(team_name: str) -> list[dict]:
    resp = requests.get(f"{BASE}/searchteams.php", params={"t": team_name}, timeout=10)
    resp.raise_for_status()
    teams = resp.json().get("teams") or []
    if not teams:
        return []

    team_id = teams[0]["idTeam"]
    resp = requests.get(f"{BASE}/eventslast.php", params={"id": team_id}, timeout=10)
    resp.raise_for_status()
    events = resp.json().get("results") or []

    return [
        {
            "text": (
                f"{e['strEvent']}: {e.get('intHomeScore', '?')}-"
                f"{e.get('intAwayScore', '?')} on {e['dateEvent']}"
            ),
            "source": "TheSportsDB",
        }
        for e in events
    ]
