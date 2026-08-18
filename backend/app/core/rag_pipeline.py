"""
Ties everything together: pull live data + retrieve from the vector store,
merge into a single context block, then ask the LLM to answer grounded
in it.
"""
from app.config import get_settings
from app.connectors.router import gather_live_context
from app.core.llm_engine import generate
from app.core.vector_store import get_vector_store

settings = get_settings()


def _build_context(snippets: list[dict]) -> tuple[str, list[dict]]:
    context, sources, total_chars = [], [], 0
    for s in snippets:
        text = s["text"].strip()
        if not text or total_chars > settings.MAX_CONTEXT_CHARS:
            continue
        context.append(f"[Source: {s.get('source', 'knowledge base')}] {text}")
        sources.append(s)
        total_chars += len(text)
    return "\n\n".join(context), sources


def answer_query(query: str) -> dict:
    store = get_vector_store()

    live_snippets = gather_live_context(query)
    kb_hits = store.query(query)
    kb_snippets = [
        {"text": h["text"], "source": h["metadata"].get("source", "knowledge base")}
        for h in kb_hits
    ]

    context, sources = _build_context(live_snippets + kb_snippets)

    if not context:
        answer = (
            "I couldn't find any current information on that — the live "
            "connectors and knowledge base both came back empty. Try "
            "rephrasing, or check that API keys are configured."
        )
    else:
        answer = generate(query, context)

    return {"answer": answer, "sources": sources}
