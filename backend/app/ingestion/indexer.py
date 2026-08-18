from app.core.vector_store import get_vector_store
from app.ingestion.chunker import chunk_text


def index_documents(documents: list[dict]) -> int:
    """documents: [{"text": ..., "source": ..., "fetched_at": ...}, ...]"""
    store = get_vector_store()
    all_chunks, all_meta = [], []

    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append(chunk)
            all_meta.append(
                {
                    "source": doc.get("source", "unknown"),
                    "fetched_at": doc.get("fetched_at", ""),
                }
            )

    store.upsert(all_chunks, all_meta)
    return len(all_chunks)
