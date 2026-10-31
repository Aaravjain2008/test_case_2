"""
tests/test_splitter.py
-----------------------
Tests for text splitting and chunk metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.ingestion.splitter import build_splitter, split_documents


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_doc(content: str, source: str = "test.md") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "file_path": f"/data/{source}", "doc_type": "markdown", "file_hash": "abc123", "page": 1},
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildSplitter:
    def test_returns_splitter(self) -> None:
        splitter = build_splitter(chunk_size=200, chunk_overlap=20)
        assert splitter is not None

    def test_overlap_must_be_less_than_size(self) -> None:
        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            build_splitter(chunk_size=100, chunk_overlap=100)


class TestSplitDocuments:
    def test_splits_long_document(self) -> None:
        long_text = "Word " * 500  # 2500 chars
        doc = _make_doc(long_text)
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_preserves_metadata(self) -> None:
        doc = _make_doc("Some content about passwords and logins.")
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.md"
            assert "chunk_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunk_id_is_stable(self) -> None:
        doc = _make_doc("Stable content.")
        chunks1 = split_documents([doc], chunk_size=200, chunk_overlap=20)
        chunks2 = split_documents([doc], chunk_size=200, chunk_overlap=20)
        assert [c.metadata["chunk_id"] for c in chunks1] == [
            c.metadata["chunk_id"] for c in chunks2
        ]

    def test_chunk_id_changes_if_content_changes(self) -> None:
        doc1 = _make_doc("Version one content.")
        doc2 = _make_doc("Version two content — different.")
        chunks1 = split_documents([doc1], chunk_size=200, chunk_overlap=20)
        chunks2 = split_documents([doc2], chunk_size=200, chunk_overlap=20)
        ids1 = {c.metadata["chunk_id"] for c in chunks1}
        ids2 = {c.metadata["chunk_id"] for c in chunks2}
        assert ids1.isdisjoint(ids2)

    def test_empty_document_list_returns_empty(self) -> None:
        assert split_documents([]) == []

    def test_empty_content_document_is_skipped(self) -> None:
        doc = _make_doc("   ")
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
        assert chunks == []

    def test_multiple_documents(self) -> None:
        docs = [_make_doc(f"Document {i} content " * 20, f"doc{i}.md") for i in range(3)]
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)
        sources = {c.metadata["source"] for c in chunks}
        assert sources == {"doc0.md", "doc1.md", "doc2.md"}
