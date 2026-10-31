"""
app/ingestion/embedder.py
-------------------------
Embedding model wrapper using sentence-transformers.

- Fully local: no API key or network call required.
- Model name is configurable via EMBEDDING_MODEL env var.
- Batch-aware: processes chunks in batches to avoid OOM on large datasets.
- The embedding function is compatible with ChromaDB's embedding_function API.
"""

from __future__ import annotations

import os
from typing import List

from app import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_BATCH_SIZE = 64


class LocalEmbedder:
    """Wraps a sentence-transformers model as a ChromaDB-compatible embedding function.

    Usage:
        embedder = LocalEmbedder()
        vectors = embedder(["text one", "text two"])

    The instance is callable, which matches ChromaDB's ``embedding_function`` interface.
    """

    def __init__(self, model_name: str | None = None, batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        """Initialise and load the embedding model.

        Args:
            model_name:  HuggingFace model identifier.  Defaults to
                         ``EMBEDDING_MODEL`` env var or ``all-MiniLM-L6-v2``.
            batch_size:  Number of texts to embed per batch.
        """
        self.model_name: str = model_name or os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL)
        self.batch_size = batch_size
        self._model = None  # lazy-loaded on first call

    def _load_model(self) -> None:
        """Load the sentence-transformers model (deferred to avoid startup cost)."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model '%s' …", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded.")

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        """Embed a list of texts.  Matches ChromaDB's EmbeddingFunction signature.

        Args:
            input: List of text strings to embed.

        Returns:
            List of embedding vectors (list of floats).

        Raises:
            RuntimeError: If the model fails to encode.
            ValueError:   If *input* is empty.
        """
        if not input:
            return []

        if self._model is None:
            self._load_model()

        results: List[List[float]] = []
        for start in range(0, len(input), self.batch_size):
            batch = input[start : start + self.batch_size]
            try:
                embeddings = self._model.encode(  # type: ignore[union-attr]
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Embedding model '{self.model_name}' failed on batch "
                    f"starting at index {start}: {exc}"
                ) from exc
            results.extend(emb.tolist() for emb in embeddings)

        logger.debug("Embedded %d text(s) using '%s'.", len(input), self.model_name)
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Convenience wrapper so the embedder can also be used as a LangChain
        ``Embeddings`` interface for retrieval.

        Args:
            text: The query string.

        Returns:
            Embedding vector as a list of floats.
        """
        vectors = self([ text])
        return vectors[0]
