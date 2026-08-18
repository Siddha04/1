"""
Wraps a sentence-transformers model so the rest of the app never touches
the underlying library directly (makes it trivial to swap models later).
"""
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import get_settings

settings = get_settings()


class EmbeddingModel:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
