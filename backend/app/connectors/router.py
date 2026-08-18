"""
Small intent router: decides which live connector(s) a query needs, on
top of (not instead of) the vector-store retrieval that always runs. This
uses keyword rules to avoid an extra model call on every request — swap
in an LLM-based classifier later once you have a fine-tuned model handy.
"""
import re

from app.connectors import finance, sports, web_search

TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
CRYPTO_WORDS = {"bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol"}
SPORTS_WORDS = {"score", "match", "game", "fixture", "vs", "won", "lost", "standings"}


def gather_live_context(query: str) -> list[dict]:
    """Returns a list of {text, source} snippets from whichever live
    connectors seem relevant to this query. Always safe to call — each
    connector fails soft (returns []) if it can't find anything."""
    q_lower = query.lower()
    results: list[dict] = []

    if any(w in q_lower for w in CRYPTO_WORDS):
        coin = "bitcoin" if "bitcoin" in q_lower or "btc" in q_lower else "ethereum"
        try:
            results.append(finance.get_crypto_price(coin))
        except Exception:
            pass

    tickers = TICKER_RE.findall(query)
    if tickers and any(w in q_lower for w in ("stock", "share", "price", "$")):
        try:
            results.append(finance.get_stock_quote(tickers[0]))
        except Exception:
            pass

    if any(w in q_lower for w in SPORTS_WORDS):
        candidate = " ".join(w for w in query.split() if w[:1].isupper())
        if candidate:
            try:
                results.extend(sports.get_recent_results(candidate)[:3])
            except Exception:
                pass

    # General web search always runs as a fallback for "current events"
    # style questions the other connectors don't cover.
    try:
        results.extend(web_search.search_web(query, max_results=4))
    except Exception:
        pass

    return results
