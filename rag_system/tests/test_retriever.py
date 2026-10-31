"""
tests/test_retriever.py
-----------------------
Tests for semantic retrieval and relevance gating.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.retrieval.retriever import RetrievalResult, RetrievedChunk, Retriever


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chroma_result(docs, metas, distances, ids=None):
    """Build a mock ChromaDB query result."""
    if ids is None:
        ids = [f"id_{i}" for i in range(len(docs))]
    return {
        "ids": [ids],
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }


def _mock_store(query_return_value, count: int = 5):
    store = MagicMock()
    store.count.return_value = count
    store.query.return_value = query_return_value
    return store


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRetrievedChunk:
    def test_source_property(self) -> None:
        chunk = RetrievedChunk(
            content="text",
            metadata={"source": "login.md", "page": "2"},
            distance=0.3,
        )
        assert chunk.source == "login.md"
        assert chunk.page == "2"

    def test_unknown_source_fallback(self) -> None:
        chunk = RetrievedChunk(content="text", metadata={}, distance=0.5)
        assert chunk.source == "unknown"


class TestRetrievalResult:
    def test_context_text_contains_chunks(self) -> None:
        chunks = [
            RetrievedChunk("content A", {"source": "a.md"}, 0.2),
            RetrievedChunk("content B", {"source": "b.md"}, 0.4),
        ]
        result = RetrievalResult(chunks=chunks, query="test", found=True)
        assert "content A" in result.context_text
        assert "content B" in result.context_text

    def test_source_citations_deduplicated(self) -> None:
        chunks = [
            RetrievedChunk("text1", {"source": "a.md"}, 0.1),
            RetrievedChunk("text2", {"source": "a.md"}, 0.2),
            RetrievedChunk("text3", {"source": "b.md"}, 0.3),
        ]
        result = RetrievalResult(chunks=chunks, query="test", found=True)
        citations = result.source_citations
        assert len(citations) == 2  # deduplicated
        assert any("a.md" in c for c in citations)
        assert any("b.md" in c for c in citations)


class TestRetriever:
    def test_returns_chunks_within_threshold(self) -> None:
        mock_result = _make_chroma_result(
            docs=["relevant content about passwords"],
            metas=[{"source": "login.md"}],
            distances=[0.5],
        )
        store = _mock_store(mock_result)
        retriever = Retriever(store=store, top_k=3, threshold=1.0)
        result = retriever.retrieve("password reset")
        assert result.found is True
        assert len(result.chunks) == 1

    def test_filters_chunks_above_threshold(self) -> None:
        mock_result = _make_chroma_result(
            docs=["irrelevant content"],
            metas=[{"source": "random.md"}],
            distances=[1.8],  # above threshold of 1.2
        )
        store = _mock_store(mock_result)
        retriever = Retriever(store=store, top_k=3, threshold=1.2)
        result = retriever.retrieve("password reset")
        assert result.found is False
        assert len(result.chunks) == 0

    def test_empty_query_returns_not_found(self) -> None:
        store = _mock_store({})
        retriever = Retriever(store=store)
        result = retriever.retrieve("")
        assert result.found is False

    def test_empty_collection_returns_not_found(self) -> None:
        store = MagicMock()
        store.query.side_effect = ValueError("Collection is empty")
        retriever = Retriever(store=store)
        result = retriever.retrieve("any question")
        assert result.found is False

    def test_mixed_results_only_relevant_returned(self) -> None:
        mock_result = _make_chroma_result(
            docs=["relevant doc", "irrelevant doc"],
            metas=[{"source": "a.md"}, {"source": "b.md"}],
            distances=[0.4, 1.5],
            ids=["id_0", "id_1"],
        )
        store = _mock_store(mock_result)
        retriever = Retriever(store=store, top_k=5, threshold=1.0)
        result = retriever.retrieve("something")
        assert result.found is True
        assert len(result.chunks) == 1
        assert result.chunks[0].source == "a.md"
