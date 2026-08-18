"""
Fast tests that don't require live API keys — verify the connectors fail
soft (return empty results) instead of crashing when unconfigured.
"""
from app.connectors import web_search


def test_web_search_returns_empty_without_key(monkeypatch):
    monkeypatch.setattr("app.connectors.web_search.settings.TAVILY_API_KEY", "")
    results = web_search.search_web("test query")
    assert results == []
