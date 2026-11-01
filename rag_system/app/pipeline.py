"""
app/pipeline.py
---------------
Updated to use new async Retriever API.
QueryPipeline.run() is now async — callers must await it.
"""

from __future__ import annotations

import asyncio
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


class IngestionPipeline:
    """Loads, chunks, embeds, and stores documents into ChromaDB."""

    def __init__(self, data_dir: Optional[str] = None, store: Optional[ChromaStore] = None) -> None:
        self.data_dir: str = data_dir or os.getenv("DATA_DIR", "./data")
        self.store: ChromaStore = store or ChromaStore()

    def run(self, reset: bool = False) -> dict:
        if reset:
            logger.warning("Resetting ChromaDB collection before ingestion.")
            self.store.reset_collection()

        logger.info("── Step 1/3: Loading documents from '%s' …", self.data_dir)
        documents = load_documents(self.data_dir)
        if not documents:
            return {"documents": 0, "chunks": 0, "stored": 0}

        logger.info("── Step 2/3: Splitting documents into chunks …")
        chunks = split_documents(documents)
        if not chunks:
            return {"documents": len(documents), "chunks": 0, "stored": 0}

        logger.info("── Step 3/3: Embedding and storing %d chunk(s) …", len(chunks))
        submitted, total = self.store.upsert_chunks(chunks)

        return {"documents": len(documents), "chunks": len(chunks), "stored": total}


class QueryPipeline:
    """Async query pipeline using the new Retriever API.

    BREAKING: run() is now a coroutine — you must await it:
        answer = await pipeline.run(question)
    """

    def __init__(
        self,
        store: Optional[ChromaStore] = None,
        llm: Optional[BaseLLMClient] = None,
        embedder: Optional[LocalEmbedder] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.store: ChromaStore = store or ChromaStore()
        self.llm: BaseLLMClient = llm or get_llm_client()
        self.embedder = embedder or LocalEmbedder()
        # NOTE: embedder is now required by Retriever
        self.retriever = Retriever(
            store=self.store,
            embedder=self.embedder,
            top_k=top_k,
            threshold=threshold,
        )

    async def run(self, question: str) -> str:
        """Answer question — must be awaited by caller."""
        if not question or not question.strip():
            return "Please enter a question."

        logger.info("Query: '%s'", question)

        result: RetrievalResult = await self.retriever.retrieve(question)

        # 'found' removed — use chunk count
        if not result.chunks:
            logger.info("No relevant context found — returning not-found response.")
            return NOT_FOUND_RESPONSE

        prompt = build_prompt(result)

        try:
            raw_answer = await asyncio.to_thread(self.llm.complete, prompt)
        except RuntimeError as exc:
            logger.error("LLM call failed: %s", exc)
            return f"An error occurred while contacting the language model.\nDetails: {exc}"

        return format_answer_with_sources(raw_answer, result)
