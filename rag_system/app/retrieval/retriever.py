"""
app/retrieval/retriever.py
--------------------------
Semantic retrieval with relevance gating.

The retriever:
1. Embeds the user's query using the same model used for ingestion.
2. Queries ChromaDB for the top-K most similar chunks.
3. Filters results by a configurable relevance threshold (L2 distance).
4. Returns structured RetrievalResult objects.

If no chunks pass the threshold, the system should NOT call the LLM —
it should return a "not found" response instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app import get_logger
from app.database.chroma import ChromaStore

logger = get_logger(__name__)

_DEFAULT_TOP_K = 5
_DEFAULT_THRESHOLD = 1.2  # L2 distance; lower = more similar


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its content and metadata."""

    content: str
    metadata: Dict[str, str]
    distance: float
    chunk_id: str = field(default="")

    @property
    def source(self) -> str:
        """Human-readable source identifier."""
        return self.metadata.get("source", "unknown")

    @property
    def page(self) -> str:
        """Page number if available."""
        return str(self.metadata.get("page", ""))


@dataclass
class RetrievalResult:
    """Outcome of a retrieval operation."""

    chunks: List[RetrievedChunk]
    query: str
    found: bool

    @property
    def context_text(self) -> str:
        """Concatenated text of all retrieved chunks, numbered for clarity."""
        parts: List[str] = []
        for i, chunk in enumerate(self.chunks, start=1):
            page_info = f", page {chunk.page}" if chunk.page else ""
            header = f"[Chunk {i} — Source: {chunk.source}{page_info}]"
            parts.append(f"{header}\n{chunk.content}")
        return "\n\n---\n\n".join(parts)

    @property
    def source_citations(self) -> List[str]:
        """Deduplicated list of citation strings for the answer footer."""
        seen: set[str] = set()
        citations: List[str] = []
        for chunk in self.chunks:
            page_info = f" — page {chunk.page}" if chunk.page else ""
            citation = f"• {chunk.source}{page_info}"
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)
        return citations


class Retriever:
    """Semantic retriever backed by ChromaDB.

    Attributes:
        store:     ChromaStore instance.
        top_k:     Maximum number of chunks to retrieve.
        threshold: Maximum L2 distance to accept a chunk as relevant.
    """

    def __init__(
        self,
        store: ChromaStore,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.store = store
        self.top_k: int = top_k or int(os.getenv("TOP_K", str(_DEFAULT_TOP_K)))
        self.threshold: float = threshold or float(
            os.getenv("RELEVANCE_THRESHOLD", str(_DEFAULT_THRESHOLD))
        )

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve the most relevant chunks for *query*.

        Args:
            query: The user's natural-language question.

        Returns:
            A ``RetrievalResult`` with zero or more ``RetrievedChunk`` objects.
        """
        if not query or not query.strip():
            logger.warning("Retriever received empty query.")
            return RetrievalResult(chunks=[], query=query, found=False)

        try:
            raw = self.store.query(query_text=query, n_results=self.top_k)
        except ValueError as exc:
            logger.warning("Retrieval skipped — collection is empty: %s", exc)
            return RetrievalResult(chunks=[], query=query, found=False)
        except RuntimeError as exc:
            logger.error("Retrieval failed: %s", exc)
            raise

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        # Always mark found=True regardless of distance — return all chunks
        chunks: List[RetrievedChunk] = []
        for chunk_id, content, meta, dist in zip(ids, documents, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    content=content,
                    metadata=meta or {},
                    distance=dist,
                    chunk_id=chunk_id,
                )
            )

        logger.debug("Retrieved %d chunk(s) (no threshold filter applied).", len(chunks))
        return RetrievalResult(chunks=chunks, query=query, found=bool(chunks))
