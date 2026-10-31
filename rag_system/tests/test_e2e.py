"""
tests/test_e2e.py
-----------------
End-to-end integration test for the entire RAG pipeline.
Uses a mock LLM and temporary directory to test the flow:
Document -> Chunk -> Embed -> Store -> Query -> Retrieve -> LLM -> Answer
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.database.chroma import ChromaStore
from app.ingestion.embedder import LocalEmbedder
from app.llm.client import BaseLLMClient
from app.pipeline import IngestionPipeline, QueryPipeline


# ── Mock LLM Client ───────────────────────────────────────────────────────────

class DummyLLMClient(BaseLLMClient):
    """A dummy LLM client that returns a pre-configured response."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_prompt = ""

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response_text


# ── Test Suite ────────────────────────────────────────────────────────────────

def test_full_pipeline_flow(tmp_path: Path) -> None:
    # 1. Setup temporary source data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    doc_content = (
        "# Server Installation Guide\n\n"
        "To install the server, run the setup.sh script. "
        "The default installation port is 8080. "
        "Verify the status using systemctl status app."
    )
    (data_dir / "install.md").write_text(doc_content, encoding="utf-8")

    # 2. Setup database persist directory
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    # 3. Instantiate components
    embedder = LocalEmbedder()
    store = ChromaStore(
        persist_dir=str(persist_dir),
        collection_name="test_e2e_collection",
        embedder=embedder,
    )

    # 4. Run Ingestion Pipeline
    ingest_pipeline = IngestionPipeline(data_dir=str(data_dir), store=store)
    summary = ingest_pipeline.run(reset=True)

    assert summary["documents"] == 1
    assert summary["chunks"] == 1
    assert summary["stored"] == 1

    # 5. Setup Query Pipeline with Dummy LLM
    expected_answer = "Install the server by running the setup.sh script."
    mock_llm = DummyLLMClient(expected_answer)
    query_pipeline = QueryPipeline(
        store=store,
        llm=mock_llm,
        top_k=2,
        threshold=1.2,
    )

    # 6. Run query that matches
    question = "How do I install the server?"
    answer = query_pipeline.run(question)

    # Verify LLM was prompted with retrieved context
    assert "Server Installation Guide" in mock_llm.last_prompt
    assert "setup.sh" in mock_llm.last_prompt
    assert question in mock_llm.last_prompt
    
    # Verify final response includes the LLM answer and the source citations
    assert expected_answer in answer
    assert "Sources:" in answer
    assert "install.md" in answer


def test_pipeline_relevance_gate(tmp_path: Path) -> None:
    # 1. Setup temporary source data with unrelated info
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "info.md").write_text("# Office Hours\nWe are open from 9 AM to 5 PM.", encoding="utf-8")

    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    embedder = LocalEmbedder()
    store = ChromaStore(
        persist_dir=str(persist_dir),
        collection_name="test_e2e_gate",
        embedder=embedder,
    )

    # Ingest
    IngestionPipeline(data_dir=str(data_dir), store=store).run(reset=True)

    # Query Pipeline with a low threshold or query completely unrelated
    mock_llm = DummyLLMClient("Should not be called")
    query_pipeline = QueryPipeline(
        store=store,
        llm=mock_llm,
        top_k=2,
        threshold=0.5, # very strict threshold
    )

    # Query something completely different
    question = "How do I unlock an account?"
    answer = query_pipeline.run(question)

    # The LLM should not be called due to relevance gating
    assert mock_llm.last_prompt == ""
    assert "The information is not available in the provided knowledge base." in answer
