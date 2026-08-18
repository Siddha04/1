"""Simple, dependency-free text chunker (character-based with overlap)."""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
