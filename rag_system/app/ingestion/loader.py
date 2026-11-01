"""
app/ingestion/loader.py
-----------------------
Document discovery and loading.

Supports: .md, .txt, .pdf files
Attaches rich metadata to every loaded document.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app import get_logger

logger = get_logger(__name__)

# File extensions and their human-readable type labels
_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".md": "markdown",
    ".txt": "text",
    ".pdf": "pdf",
}


def _file_hash(path: Path) -> str:
    """Return a short SHA-256 hex digest for the file at *path*.

    Used to detect whether a document has changed since the last ingestion
    so we can skip unchanged files.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def _file_size_kb(path: Path) -> float:
    """Return the file size in kilobytes."""
    return path.stat().st_size / 1024


def _load_text_file(path: Path) -> List[Document]:
    """Load a plain-text or markdown file as a single Document."""
    t_start = time.monotonic()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read %s: %s — skipping.", path, exc)
        return []

    if not text.strip():
        logger.warning("File %s is empty — skipping.", path)
        return []

    elapsed = time.monotonic() - t_start
    size_kb = _file_size_kb(path)
    logger.debug(
        "Loaded '%s' (%.1f KB) in %.3fs.", path.name, size_kb, elapsed
    )

    doc = Document(
        page_content=text,
        metadata={
            "source": path.name,
            "file_path": str(path.resolve()),
            "doc_type": _SUPPORTED_EXTENSIONS.get(path.suffix.lower(), "unknown"),
            "file_hash": _file_hash(path),
            "file_size_kb": round(size_kb, 2),
            "page": 0,  # NOTE: using 0-based page index for text files
        },
    )
    return [doc]


def _load_pdf_file(path: Path) -> List[Document]:
    """Load a PDF file page-by-page using PyPDFLoader.

    Falls back gracefully if pypdf is not installed.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader  # type: ignore
    except ImportError:
        logger.warning(
            "PyPDFLoader not available (install pypdf). Skipping %s.", path
        )
        return []

    try:
        loader = PyPDFLoader(str(path))
        pages: List[Document] = loader.load()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load PDF %s: %s — skipping.", path, exc)
        return []

    file_hash = _file_hash(path)
    size_kb = _file_size_kb(path)
    for doc in pages:
        doc.metadata.update(
            {
                "source": path.name,
                "file_path": str(path.resolve()),
                "doc_type": "pdf",
                "file_hash": file_hash,
                "file_size_kb": round(size_kb, 2),
            }
        )
    return pages


def discover_documents(data_dir: str) -> List[Path]:
    """Return a sorted list of supported document paths inside *data_dir*.

    Args:
        data_dir: Path to the directory containing source documents.

    Returns:
        Sorted list of Path objects for each supported file found.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
        NotADirectoryError: If *data_dir* is a file, not a directory.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Data directory not found: {root.resolve()}\n"
            "Set DATA_DIR in your .env file or pass --data-dir on the CLI."
        )
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got a file: {root.resolve()}")

    paths: List[Path] = []
    for ext in _SUPPORTED_EXTENSIONS:
        paths.extend(root.rglob(f"*{ext}"))

    paths.sort()
    total_size = sum(p.stat().st_size for p in paths) / 1024
    logger.info(
        "Discovered %d document(s) in %s (total %.1f KB).",
        len(paths),
        root.resolve(),
        total_size,
    )
    return paths


def load_documents(data_dir: str) -> List[Document]:
    """Discover and load all supported documents from *data_dir*.

    Args:
        data_dir: Path to the document library directory.

    Returns:
        List of LangChain Documents, each with populated metadata.
    """
    paths = discover_documents(data_dir)
    if not paths:
        logger.warning("No supported documents found in %s.", data_dir)
        return []

    documents: List[Document] = []
    failed = 0
    for path in paths:
        ext = path.suffix.lower()
        if ext == ".pdf":
            docs = _load_pdf_file(path)
        else:
            docs = _load_text_file(path)

        if docs:
            logger.info("Loaded '%s' → %d page(s).", path.name, len(docs))
            documents.extend(docs)
        else:
            logger.warning("No content extracted from '%s'.", path.name)
            failed += 1

    logger.info(
        "Total documents loaded: %d (%d failed).", len(documents), failed
    )
    return documents
