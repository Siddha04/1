from fastapi import APIRouter

from app.core.vector_store import get_vector_store

router = APIRouter()


@router.get("/health")
def health():
    store = get_vector_store()
    return {"status": "ok", "knowledge_base_chunks": store.count()}
