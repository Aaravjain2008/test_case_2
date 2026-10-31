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
        """Initialise the retriever.

        Args:
            store:     Initialised ChromaStore.
            top_k:     Number of top chunks to retrieve.  Defaults to
                       ``TOP_K`` env var or 5.
            threshold: L2 distance threshold.  Defaults to
                       ``RELEVANCE_THRESHOLD`` env var or 1.2.
        """
        self.store = store
        self.top_k: int = (
            top_k if top_k is not None else int(os.getenv("TOP_K", str(_DEFAULT_TOP_K)))
        )
        self.threshold: float = (
            threshold
            if threshold is not None
            else float(os.getenv("RELEVANCE_THRESHOLD", str(_DEFAULT_THRESHOLD)))
        )
        logger.debug(
            "Retriever: top_k=%d, threshold=%.2f.", self.top_k, self.threshold
        )

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve the most relevant chunks for *query*.

        The pipeline:
        1. Query ChromaDB for ``top_k`` candidates.
        2. Filter candidates whose L2 distance exceeds ``threshold``.
        3. Return a ``RetrievalResult`` indicating whether useful context
           was found.

        If the collection is empty or no chunks pass the threshold, the
        ``found`` attribute of the result is ``False``.

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
            # Collection is empty
            logger.warning("Retrieval skipped — collection is empty: %s", exc)
            return RetrievalResult(chunks=[], query=query, found=False)
        except RuntimeError as exc:
            logger.error("Retrieval failed: %s", exc)
            raise

        # ChromaDB returns lists nested in a single-element outer list
        ids: List[str] = raw.get("ids", [[]])[0]
        docs: List[str] = raw.get("documents", [[]])[0]
        metas: List[Dict] = raw.get("metadatas", [[]])[0]
        dists: List[float] = raw.get("distances", [[]])[0]

        chunks: List[RetrievedChunk] = []
        for chunk_id, content, meta, dist in zip(ids, docs, metas, dists):
            if dist <= self.threshold:
                chunks.append(
                    RetrievedChunk(
                        content=content,
                        metadata=meta,
                        distance=dist,
                        chunk_id=chunk_id,
                    )
                )
            else:
                logger.debug(
                    "Filtered chunk '%s' (distance=%.4f > threshold=%.2f).",
                    chunk_id,
                    dist,
                    self.threshold,
                )

        found = len(chunks) > 0
        if found:
            logger.info(
                "Retrieved %d relevant chunk(s) for query (best distance=%.4f).",
                len(chunks),
                chunks[0].distance,
            )
        else:
            logger.info(
                "No relevant chunks found within threshold=%.2f for query: '%s'.",
                self.threshold,
                query[:80],
            )

        return RetrievalResult(chunks=chunks, query=query, found=found)
