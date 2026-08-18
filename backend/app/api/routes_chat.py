from fastapi import APIRouter
from pydantic import BaseModel

from app.core.rag_pipeline import answer_query

router = APIRouter()


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = answer_query(req.query)
    return ChatResponse(**result)
