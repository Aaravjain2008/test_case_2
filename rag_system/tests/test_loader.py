"""
tests/test_loader.py
--------------------
Tests for document discovery and loading.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.ingestion.loader import discover_documents, load_documents


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def temp_data_dir(tmp_path: Path) -> Path:
    """A temporary directory with sample documents."""
    (tmp_path / "doc1.md").write_text("# Title\nSome content.", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Plain text content.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("a,b,c", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def empty_data_dir(tmp_path: Path) -> Path:
    """A temporary directory with no supported files."""
    (tmp_path / "ignored.csv").write_text("a,b,c", encoding="utf-8")
    return tmp_path


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDiscoverDocuments:
    def test_finds_supported_files(self, temp_data_dir: Path) -> None:
        paths = discover_documents(str(temp_data_dir))
        names = {p.name for p in paths}
        assert "doc1.md" in names
        assert "doc2.txt" in names

    def test_ignores_unsupported_files(self, temp_data_dir: Path) -> None:
        paths = discover_documents(str(temp_data_dir))
        names = {p.name for p in paths}
        assert "ignored.csv" not in names

    def test_raises_if_dir_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            discover_documents("/nonexistent/path/that/does/not/exist")

    def test_raises_if_path_is_file(self, tmp_path: Path) -> None:
        f = tmp_path / "afile.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            discover_documents(str(f))

    def test_empty_dir_returns_empty_list(self, empty_data_dir: Path) -> None:
        paths = discover_documents(str(empty_data_dir))
        assert paths == []


class TestLoadDocuments:
    def test_loads_markdown_with_metadata(self, temp_data_dir: Path) -> None:
        docs = load_documents(str(temp_data_dir))
        assert len(docs) >= 1
        sources = {d.metadata["source"] for d in docs}
        assert "doc1.md" in sources

    def test_metadata_fields_present(self, temp_data_dir: Path) -> None:
        docs = load_documents(str(temp_data_dir))
        for doc in docs:
            assert "source" in doc.metadata
            assert "file_path" in doc.metadata
            assert "doc_type" in doc.metadata
            assert "file_hash" in doc.metadata

    def test_empty_dir_returns_empty_list(self, empty_data_dir: Path) -> None:
        docs = load_documents(str(empty_data_dir))
        assert docs == []

    def test_skips_empty_files(self, tmp_path: Path) -> None:
        (tmp_path / "empty.md").write_text("   \n  ", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert all(d.metadata["source"] != "empty.md" for d in docs)

    def test_doc_type_markdown(self, temp_data_dir: Path) -> None:
        docs = load_documents(str(temp_data_dir))
        md_docs = [d for d in docs if d.metadata["source"] == "doc1.md"]
        assert len(md_docs) == 1
        assert md_docs[0].metadata["doc_type"] == "markdown"
