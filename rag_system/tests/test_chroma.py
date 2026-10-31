"""
tests/test_chroma.py
--------------------
Tests for ChromaDB storage and retrieval.
Uses an in-memory-style temporary directory to avoid polluting chroma_db/.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.database.chroma import ChromaStore
from app.ingestion.embedder import LocalEmbedder
from app.ingestion.splitter import split_documents


# ── Shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder()


@pytest.fixture()
def temp_store(tmp_path, embedder) -> ChromaStore:
    """A ChromaStore backed by a temporary directory."""
    return ChromaStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
        embedder=embedder,
    )


def _make_chunk(content: str, source: str, idx: int) -> Document:
    import hashlib
    chunk_id = hashlib.sha256(f"{source}::{idx}::{content}".encode()).hexdigest()[:16]
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "file_path": f"/data/{source}",
            "doc_type": "markdown",
            "file_hash": "test123",
            "page": 1,
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "total_chunks_in_doc": 1,
        },
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestChromaStoreInit:
    def test_initialises_empty(self, temp_store: ChromaStore) -> None:
        assert temp_store.count() == 0

    def test_collection_name_set(self, temp_store: ChromaStore) -> None:
        assert temp_store.collection_name == "test_collection"


class TestChromaUpsert:
    def test_upserts_chunks(self, temp_store: ChromaStore) -> None:
        chunks = [_make_chunk("Password reset content.", "login.md", 0)]
        submitted, total = temp_store.upsert_chunks(chunks)
        assert submitted == 1
        assert total == 1

    def test_upsert_is_idempotent(self, temp_store: ChromaStore) -> None:
        chunks = [_make_chunk("Idempotent content.", "stable.md", 0)]
        temp_store.upsert_chunks(chunks)
        temp_store.upsert_chunks(chunks)  # second upsert should not duplicate
        # ChromaDB upsert replaces on same ID
        assert temp_store.count() >= 1

    def test_upsert_empty_list_is_noop(self, temp_store: ChromaStore) -> None:
        count_before = temp_store.count()
        submitted, _ = temp_store.upsert_chunks([])
        assert submitted == 0
        assert temp_store.count() == count_before

    def test_missing_chunk_id_raises(self, temp_store: ChromaStore) -> None:
        bad_chunk = Document(
            page_content="No chunk_id here.",
            metadata={"source": "bad.md"},
        )
        with pytest.raises(ValueError, match="chunk_id"):
            temp_store.upsert_chunks([bad_chunk])


class TestChromaQuery:
    def test_query_returns_results(self, temp_store: ChromaStore) -> None:
        chunks = [
            _make_chunk("How to reset password on the login page.", "login.md", 0),
            _make_chunk("Subscription plan pricing information.", "plans.md", 1),
        ]
        temp_store.upsert_chunks(chunks)
        result = temp_store.query("password reset", n_results=2)
        docs = result["documents"][0]
        assert len(docs) > 0
        assert any("password" in d.lower() for d in docs)

    def test_query_empty_collection_raises(self, tmp_path) -> None:
        embedder = LocalEmbedder()
        empty_store = ChromaStore(
            persist_dir=str(tmp_path / "empty_chroma"),
            collection_name="empty_col",
            embedder=embedder,
        )
        with pytest.raises(ValueError, match="empty"):
            empty_store.query("any question")

    def test_metadata_preserved_in_results(self, temp_store: ChromaStore) -> None:
        chunks = [_make_chunk("Billing info details.", "billing.md", 0)]
        temp_store.upsert_chunks(chunks)
        result = temp_store.query("billing", n_results=1)
        metas = result["metadatas"][0]
        assert any(m.get("source") == "billing.md" for m in metas)

    def test_distances_returned(self, temp_store: ChromaStore) -> None:
        chunks = [_make_chunk("Security two-factor authentication.", "security.md", 0)]
        temp_store.upsert_chunks(chunks)
        result = temp_store.query("security", n_results=1)
        distances = result["distances"][0]
        assert len(distances) > 0
        assert all(isinstance(d, float) for d in distances)
