"""
Scheduled refresh job, run by .github/workflows/ingest.yml on a cron.
Invoke from the repo root: PYTHONPATH=backend python -m app.ingestion.crawler

Pulls fresh results for a watchlist of topics and re-indexes them, so the
vector store never goes seriously stale for the things you actually care
about.
"""
import datetime as dt
import json
import pathlib

from app.connectors import web_search
from app.ingestion.indexer import index_documents

WATCHLIST_FILE = pathlib.Path(__file__).parent / "watchlist.json"


def load_watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        return json.loads(WATCHLIST_FILE.read_text())
    return ["AI industry news", "global markets today"]


def run():
    topics = load_watchlist()
    fetched_at = dt.datetime.utcnow().isoformat()
    documents = []

    for topic in topics:
        for item in web_search.search_web(topic, max_results=5):
            documents.append(
                {"text": item["text"], "source": item["source"], "fetched_at": fetched_at}
            )

    count = index_documents(documents)
    print(
        f"[ingest] indexed {count} chunks from {len(documents)} docs "
        f"across {len(topics)} watched topics"
    )


if __name__ == "__main__":
    run()
