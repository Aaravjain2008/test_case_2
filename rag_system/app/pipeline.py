"""
app/pipeline.py
---------------
Orchestrates the two major pipelines:

IngestionPipeline
    data_dir → load → split → embed → upsert (ChromaDB)

QueryPipeline
    question → embed → retrieve → gate → prompt → LLM → answer
"""

from __future__ import annotations

import os
from typing import Optional

from app import get_logger
from app.database.chroma import ChromaStore
from app.ingestion.embedder import LocalEmbedder
from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.llm.client import BaseLLMClient, get_llm_client
from app.prompts.grounding_prompt import (
    NOT_FOUND_RESPONSE,
    build_prompt,
    format_answer_with_sources,
)
from app.retrieval.retriever import RetrievalResult, Retriever

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class IngestionPipeline:
    """Loads, chunks, embeds, and stores documents into ChromaDB.

    Designed for large datasets:
    - Documents are discovered from disk without loading them all at once.
    - Chunks are upserted by stable ID, so unchanged docs are skipped.
    - Embedding happens in configurable batches.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        store: Optional[ChromaStore] = None,
    ) -> None:
        """Initialise the ingestion pipeline.

        Args:
            data_dir: Path to the document library.  Defaults to ``DATA_DIR``
                      env var or ``./data``.
            store:    ChromaStore instance.  Created automatically if not supplied.
        """
        self.data_dir: str = data_dir or os.getenv("DATA_DIR", "./data")
        self.store: ChromaStore = store or ChromaStore()

    def run(self, reset: bool = False) -> dict:
        """Execute the full ingestion pipeline.

        Args:
            reset: If ``True``, wipe the collection before ingesting.  Use
                   this when you want to rebuild from scratch.

        Returns:
            A summary dict with keys: documents, chunks, stored.

        Raises:
            FileNotFoundError: If data_dir does not exist.
            RuntimeError:      If embedding or ChromaDB fails.
        """
        if reset:
            logger.warning("Resetting ChromaDB collection before ingestion.")
            self.store.reset_collection()

        # Step 1: Load
        logger.info("── Step 1/3: Loading documents from '%s' …", self.data_dir)
        documents = load_documents(self.data_dir)
        if not documents:
            logger.warning("No documents found. Ingestion complete (nothing to do).")
            return {"documents": 0, "chunks": 0, "stored": 0}

        # Step 2: Split
        logger.info("── Step 2/3: Splitting documents into chunks …")
        chunks = split_documents(documents)
        if not chunks:
            logger.warning("No chunks produced. Ingestion complete.")
            return {"documents": len(documents), "chunks": 0, "stored": 0}

        # Step 3: Upsert into ChromaDB (embedding happens inside the store)
        logger.info("── Step 3/3: Embedding and storing %d chunk(s) …", len(chunks))
        submitted, total = self.store.upsert_chunks(chunks)

        summary = {
            "documents": len(documents),
            "chunks": len(chunks),
            "stored": total,
        }
        logger.info(
            "Ingestion complete — %d doc(s) → %d chunk(s) → %d in store.",
            summary["documents"],
            summary["chunks"],
            summary["stored"],
        )
        return summary


# ══════════════════════════════════════════════════════════════════════════════
# Query Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class QueryPipeline:
    """Retrieves context and generates grounded answers for user questions.

    Pipeline:
        question → semantic search → relevance gate → prompt → LLM → answer
    """

    def __init__(
        self,
        store: Optional[ChromaStore] = None,
        llm: Optional[BaseLLMClient] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """Initialise the query pipeline.

        Args:
            store:     ChromaStore.  Created automatically if not supplied.
            llm:       LLM client.  Created via factory if not supplied.
            top_k:     Retrieval count.  Defaults to ``TOP_K`` env var.
            threshold: Relevance threshold.  Defaults to
                       ``RELEVANCE_THRESHOLD`` env var.
        """
        self.store: ChromaStore = store or ChromaStore()
        self.llm: BaseLLMClient = llm or get_llm_client()
        self.retriever = Retriever(
            store=self.store,
            top_k=top_k,
            threshold=threshold,
        )

    def run(self, question: str) -> str:
        """Answer *question* from the knowledge base.

        Args:
            question: Natural-language question from the user.

        Returns:
            A grounded answer string (with source citations), or a
            clear "not found" message if no relevant context exists.
        """
        if not question or not question.strip():
            return "Please enter a question."

        logger.info("Query: '%s'", question)

        # Step 1: Retrieve
        result: RetrievalResult = self.retriever.retrieve(question)

        # Step 2: Relevance gate — do NOT call LLM if no relevant context found
        if not result.found:
            logger.info("No relevant context found — returning not-found response.")
            return NOT_FOUND_RESPONSE

        # Step 3: Build grounded prompt
        prompt = build_prompt(result)

        # Step 4: Call LLM
        try:
            raw_answer = self.llm.complete(prompt)
        except RuntimeError as exc:
            logger.error("LLM call failed: %s", exc)
            return (
                "An error occurred while contacting the language model.\n"
                f"Details: {exc}"
            )

        # Step 5: Append source citations
        final_answer = format_answer_with_sources(raw_answer, result)

        return final_answer
