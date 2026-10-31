"""
tests/test_config_defaults.py
-----------------------------
Explicit zero values must not silently fall back to defaults.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.ingestion.splitter import build_splitter
from app.retrieval.retriever import Retriever


class TestSplitterConfig:
    def test_zero_overlap_is_respected(self) -> None:
        splitter = build_splitter(chunk_size=200, chunk_overlap=0)
        assert splitter._chunk_overlap == 0

    def test_env_values_used_when_args_missing(self, monkeypatch) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "300")
        monkeypatch.setenv("CHUNK_OVERLAP", "30")
        splitter = build_splitter()
        assert splitter._chunk_size == 300
        assert splitter._chunk_overlap == 30

    def test_overlap_must_be_smaller_than_size(self) -> None:
        with pytest.raises(ValueError):
            build_splitter(chunk_size=100, chunk_overlap=100)


class TestRetrieverConfig:
    def test_zero_threshold_is_respected(self) -> None:
        retriever = Retriever(store=MagicMock(), top_k=3, threshold=0.0)
        assert retriever.threshold == 0.0

    def test_defaults_when_nothing_given(self, monkeypatch) -> None:
        monkeypatch.delenv("TOP_K", raising=False)
        monkeypatch.delenv("RELEVANCE_THRESHOLD", raising=False)
        retriever = Retriever(store=MagicMock())
        assert retriever.top_k == 5
        assert retriever.threshold == 1.2

    def test_exact_match_passes_zero_threshold(self) -> None:
        store = MagicMock()
        store.query.return_value = {
            "ids": [["a"]],
            "documents": [["exact"]],
            "metadatas": [[{"source": "a.md"}]],
            "distances": [[0.0]],
        }
        result = Retriever(store=store, top_k=1, threshold=0.0).retrieve("q")
        assert result.found is True
