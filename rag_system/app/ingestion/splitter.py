"""
app/ingestion/splitter.py
-------------------------
Text chunking using LangChain's RecursiveCharacterTextSplitter.

Chunk size defaults are chosen for markdown knowledge-base documents:
- chunk_size=512   captures a complete section without losing context
- chunk_overlap=64 prevents facts that straddle chunk boundaries from being missed
- separators are markdown-aware: headings are preferred split points so each
  chunk stays topically coherent

All configuration is read from environment variables so it can be changed
without touching code.
"""

from __future__ import annotations

import hashlib
import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import get_logger

logger = get_logger(__name__)

# Markdown-aware separators (ordered from largest to smallest semantic unit)
_MARKDOWN_SEPARATORS: List[str] = [
    "\n## ",   # h2 heading — major section break
    "\n### ",  # h3 heading
    "\n#### ", # h4 heading
    "\n\n",    # paragraph break
    "\n",      # line break
    " ",       # word boundary (last resort)
    "",        # character boundary (absolute last resort)
]


def _chunk_id(source: str, chunk_index: int, content: str) -> str:
    """Generate a stable, unique ID for a chunk.

    The ID is derived from the source filename, chunk index, and a short
    content hash so re-ingesting a changed document produces different IDs
    (forcing re-embedding) while an unchanged document keeps the same IDs
    (allowing ChromaDB to skip re-embedding via upsert).
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
    return f"{source}::chunk_{chunk_index:04d}::{content_hash}"


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Build and return a configured RecursiveCharacterTextSplitter.

    Values are read from environment variables if not passed directly:
      CHUNK_SIZE    (default: 512)
      CHUNK_OVERLAP (default: 64)

    Args:
        chunk_size:    Maximum characters per chunk.  Overrides env var.
        chunk_overlap: Overlap characters between adjacent chunks.

    Returns:
        A configured RecursiveCharacterTextSplitter instance.
    """
    size = chunk_size or int(os.getenv("CHUNK_SIZE", "512"))
    overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "64"))

    if overlap >= size:
        raise ValueError(
            f"CHUNK_OVERLAP ({overlap}) must be less than CHUNK_SIZE ({size})."
        )

    logger.debug("Splitter: chunk_size=%d, chunk_overlap=%d.", size, overlap)
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=_MARKDOWN_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """Split a list of Documents into chunks and attach chunk metadata.

    Each output chunk receives:
      - ``chunk_id``    — stable unique identifier
      - ``chunk_index`` — zero-based position within the source document
      - All original metadata from the parent document

    Args:
        documents:    List of LangChain Documents to split.
        chunk_size:   Optional override for chunk size.
        chunk_overlap: Optional override for chunk overlap.

    Returns:
        List of chunk Documents with populated metadata.

    Raises:
        ValueError: If documents list is empty or contains invalid entries.
    """
    if not documents:
        logger.warning("split_documents received an empty document list.")
        return []

    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: List[Document] = []

    for doc in documents:
        if not doc.page_content or not doc.page_content.strip():
            logger.warning(
                "Skipping empty document: %s", doc.metadata.get("source", "unknown")
            )
            continue

        raw_chunks = splitter.split_documents([doc])

        source = doc.metadata.get("source", "unknown")
        for idx, chunk in enumerate(raw_chunks):
            chunk.metadata["chunk_id"] = _chunk_id(source, idx, chunk.page_content)
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks_in_doc"] = len(raw_chunks)
            chunks.append(chunk)

        logger.debug("'%s' → %d chunk(s).", source, len(raw_chunks))

    logger.info(
        "Splitting complete: %d document(s) → %d chunk(s).",
        len(documents),
        len(chunks),
    )
    return chunks
