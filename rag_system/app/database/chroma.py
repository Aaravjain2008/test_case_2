"""
app/database/chroma.py
----------------------
ChromaDB vector store management.

Key design decisions:
- Uses PersistentClient so the database survives process restarts.
- Documents are upserted by chunk_id (hash of source + index + content).
  This means re-running ingestion on unchanged files is a no-op.
- Stores full metadata alongside every chunk for source attribution.
- The collection name is configurable via CHROMA_COLLECTION env var.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from app import get_logger
from app.ingestion.embedder import LocalEmbedder

logger = get_logger(__name__)

_DEFAULT_PERSIST_DIR = "./chroma_db"
_DEFAULT_COLLECTION = "rag_knowledge_base"


class ChromaStore:
    """Persistent ChromaDB vector store with upsert-based ingestion.

    Attributes:
        persist_dir:     Filesystem path where ChromaDB persists data.
        collection_name: Name of the ChromaDB collection.
        embedder:        LocalEmbedder instance used to embed queries.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedder: Optional[LocalEmbedder] = None,
    ) -> None:
        """Initialise the ChromaDB client and open/create the collection.

        Args:
            persist_dir:     Path for on-disk persistence.  Defaults to
                             ``CHROMA_PERSIST_DIR`` env var.
            collection_name: ChromaDB collection name.  Defaults to
                             ``CHROMA_COLLECTION`` env var.
            embedder:        Embedding function instance.  Created automatically
                             if not supplied.

        Raises:
            RuntimeError: If ChromaDB cannot initialise at the given path.
        """
        self.persist_dir: str = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", _DEFAULT_PERSIST_DIR
        )
        self.collection_name: str = collection_name or os.getenv(
            "CHROMA_COLLECTION", _DEFAULT_COLLECTION
        )
        self.embedder: LocalEmbedder = embedder or LocalEmbedder()

        logger.info(
            "Initialising ChromaDB: path='%s', collection='%s'.",
            self.persist_dir,
            self.collection_name,
        )
        try:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ChromaDB at '{self.persist_dir}': {exc}"
            ) from exc

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedder,  # type: ignore[arg-type]
            metadata={"hnsw:space": "l2"},     # L2 distance for relevance gating
        )
        logger.info(
            "Collection '%s' ready — %d existing document(s).",
            self.collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_chunks(
        self, chunks: List[Any], batch_size: int = 100
    ) -> Tuple[int, int]:
        """Upsert document chunks into ChromaDB.

        Uses the ``chunk_id`` field from metadata as the document ID so
        that re-ingesting unchanged chunks is a no-op.

        Args:
            chunks:     List of LangChain Document objects (post-splitting).
            batch_size: Number of chunks to upsert per call (controls memory).

        Returns:
            Tuple of (total_chunks_submitted, total_chunks_in_collection).

        Raises:
            ValueError: If ``chunk_id`` is missing from a chunk's metadata.
            RuntimeError: If ChromaDB upsert fails.
        """
        if not chunks:
            logger.warning("upsert_chunks called with empty list.")
            return 0, self._collection.count()

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = chunk.metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError(
                    f"Chunk is missing 'chunk_id' in metadata: {chunk.metadata}"
                )
            # ChromaDB metadata values must be str | int | float | bool
            safe_meta: Dict[str, Any] = {
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in chunk.metadata.items()
            }
            ids.append(chunk_id)
            texts.append(chunk.page_content)
            metadatas.append(safe_meta)

        submitted = 0
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            batch_texts = texts[start : start + batch_size]
            batch_meta = metadatas[start : start + batch_size]
            try:
                self._collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_meta,
                )
                submitted += len(batch_ids)
                logger.debug("Upserted batch of %d chunk(s).", len(batch_ids))
            except Exception as exc:
                raise RuntimeError(
                    f"ChromaDB upsert failed on batch starting at index {start}: {exc}"
                ) from exc

        total = self._collection.count()
        logger.info(
            "Upsert complete — %d chunk(s) submitted; %d total in collection.",
            submitted,
            total,
        )
        return submitted, total

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """Perform semantic similarity search against the collection.

        Args:
            query_text: The user's natural-language question.
            n_results:  Maximum number of results to return.

        Returns:
            Raw ChromaDB query result dict with keys:
            ``ids``, ``documents``, ``metadatas``, ``distances``.

        Raises:
            RuntimeError: If ChromaDB query fails.
            ValueError:   If the collection is empty.
        """
        if self._collection.count() == 0:
            raise ValueError(
                "ChromaDB collection is empty. Run ingestion first:\n"
                "  python main.py ingest"
            )

        n_results = min(n_results, self._collection.count())
        try:
            result = self._collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RuntimeError(f"ChromaDB query failed: {exc}") from exc

        logger.debug(
            "Query returned %d result(s) for: '%s'.",
            len(result.get("ids", [[]])[0]),
            query_text[:80],
        )
        return result

    def count(self) -> int:
        """Return the number of documents currently stored in the collection."""
        return self._collection.count()

    def reset_collection(self) -> None:
        """Delete and recreate the collection (destructive — use with caution)."""
        logger.warning("Resetting collection '%s'.", self.collection_name)
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedder,  # type: ignore[arg-type]
            metadata={"hnsw:space": "l2"},
        )
        logger.info("Collection '%s' has been reset.", self.collection_name)
