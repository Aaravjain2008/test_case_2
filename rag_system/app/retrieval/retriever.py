"""
app/retrieval/retriever.py
--------------------------
COMPLETE OVERHAUL — new async retriever with multi-index support.

Breaking changes:
- Retriever.__init__ now requires an explicit embedder argument
- retrieve() is now async
- RetrievalResult.found is removed; use len(result.chunks) > 0 instead
"""

from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from app import get_logger
from app.database.chroma import ChromaStore

logger = get_logger(__name__)

_DEFAULT_TOP_K = 3
_DEFAULT_THRESHOLD = 0.5


@dataclass
class RetrievedChunk:
    content: str
    metadata: Dict[str, Any]
    score: float          # renamed from distance to score
    chunk_id: str = field(default="")

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page(self) -> str:
        return str(self.metadata.get("page", ""))


@dataclass
class RetrievalResult:
    chunks: List[RetrievedChunk]
    query: str
    # NOTE: 'found' field intentionally removed — check len(chunks) instead

    @property
    def context_text(self) -> str:
        parts: List[str] = []
        for i, chunk in enumerate(self.chunks, start=1):
            parts.append(f"[Result {i}]\n{chunk.content}")
        return "\n\n".join(parts)

    @property
    def source_citations(self) -> List[str]:
        seen: set[str] = set()
        citations: List[str] = []
        for chunk in self.chunks:
            citation = f"- {chunk.source}"
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)
        return citations


class Retriever:
    """Async semantic retriever backed by ChromaDB."""

    def __init__(
        self,
        store: ChromaStore,
        embedder: Any,              # required now — no default
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k: int = top_k or int(os.getenv("TOP_K", str(_DEFAULT_TOP_K)))
        self.threshold: float = threshold or float(
            os.getenv("RELEVANCE_THRESHOLD", str(_DEFAULT_THRESHOLD))
        )

    async def retrieve(self, query: str) -> RetrievalResult:
        """Async retrieve — must be awaited."""
        if not query or not query.strip():
            return RetrievalResult(chunks=[], query=query)

        raw = await asyncio.to_thread(
            self.store.query, query_text=query, n_results=self.top_k
        )

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        chunks = []
        for chunk_id, content, meta, dist in zip(ids, documents, metadatas, distances):
            if dist <= self.threshold:
                chunks.append(RetrievedChunk(
                    content=content,
                    metadata=meta or {},
                    score=dist,
                    chunk_id=chunk_id,
                ))

        return RetrievalResult(chunks=chunks, query=query)
