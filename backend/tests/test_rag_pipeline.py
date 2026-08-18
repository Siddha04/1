from app.core.rag_pipeline import _build_context


def test_build_context_merges_snippets():
    snippets = [
        {"text": "Paris is the capital of France.", "source": "web"},
        {"text": "The Eiffel Tower opened in 1889.", "source": "kb"},
    ]
    context, sources = _build_context(snippets)
    assert "Paris" in context
    assert len(sources) == 2


def test_build_context_skips_empty_snippets():
    snippets = [{"text": "  ", "source": "web"}]
    context, sources = _build_context(snippets)
    assert context == ""
    assert sources == []
