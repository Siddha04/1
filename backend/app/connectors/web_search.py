"""
Live web search via Tavily — built for LLM/RAG use cases, so it returns
clean, citation-ready snippets instead of raw HTML. Free key at tavily.com.
"""
import requests

from app.config import get_settings

settings = get_settings()


def search_web(query: str, max_results: int = 5) -> list[dict]:
    if not settings.TAVILY_API_KEY:
        return []

    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": settings.TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    return [
        {
            "text": item.get("content", ""),
            "source": item.get("url", ""),
            "title": item.get("title", ""),
            "published": item.get("published_date"),
        }
        for item in data.get("results", [])
    ]
