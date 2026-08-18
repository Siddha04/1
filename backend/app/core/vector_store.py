"""
Thin wrapper around a persistent ChromaDB collection. This is the
"knowledge base" that the ingestion pipeline refreshes and the RAG
pipeline queries.
"""
import uuid
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.core.embeddings import get_embedding_model

settings = get_settings()


class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = get_embedding_model()

    def upsert(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """Embed + store a batch of chunks. Called by the ingestion pipeline."""
        if not documents:
            return
        ids = [str(uuid.uuid4()) for _ in documents]
        vectors = self._embedder.embed(documents)
        self._collection.upsert(
            ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas
        )

    def query(self, text: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Return the top_k most relevant chunks for a query string."""
        top_k = top_k or settings.TOP_K_RESULTS
        vector = self._embedder.embed_one(text)
        result = self._collection.query(query_embeddings=[vector], n_results=top_k)

        docs = result["documents"][0] if result["documents"] else []
        metas = result["metadatas"][0] if result["metadatas"] else []
        dists = result["distances"][0] if result["distances"] else []

        return [
            {"text": doc, "metadata": meta, "score": 1 - dist}
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def get_recent(self, limit: int = 200) -> List[str]:
        """Return up to `limit` stored chunks — used by the training
        pipeline to build fresh fine-tuning examples."""
        result = self._collection.get(limit=limit)
        return result.get("documents", [])

    def count(self) -> int:
        return self._collection.count()


_store: "VectorStore | None" = None


def get_vector_store() -> "VectorStore":
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
