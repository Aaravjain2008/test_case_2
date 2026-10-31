"""
tests/test_llm_prompt.py
------------------------
Tests for the anti-hallucination prompt construction and grounding rules.
"""

from __future__ import annotations

import pytest

from app.prompts.grounding_prompt import (
    SYSTEM_PROMPT,
    build_prompt,
    format_answer_with_sources,
)
from app.retrieval.retriever import RetrievalResult, RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_result() -> RetrievalResult:
    chunks = [
        RetrievedChunk(
            content="To reset your password, click Forgot Password on the login page.",
            metadata={"source": "account_login.md", "page": "1"},
            distance=0.1,
        )
    ]
    return RetrievalResult(chunks=chunks, query="How do I reset password?", found=True)


@pytest.fixture()
def empty_result() -> RetrievalResult:
    return RetrievalResult(chunks=[], query="How to fly a plane?", found=False)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPromptConstruction:
    def test_build_prompt_includes_system_prompt(self, sample_result: RetrievalResult) -> None:
        prompt = build_prompt(sample_result)
        assert SYSTEM_PROMPT in prompt

    def test_build_prompt_includes_context(self, sample_result: RetrievalResult) -> None:
        prompt = build_prompt(sample_result)
        assert "To reset your password" in prompt
        assert "account_login.md" in prompt
        assert "page 1" in prompt

    def test_build_prompt_includes_query(self, sample_result: RetrievalResult) -> None:
        prompt = build_prompt(sample_result)
        assert "How do I reset password?" in prompt

    def test_build_prompt_raises_on_not_found(self, empty_result: RetrievalResult) -> None:
        with pytest.raises(ValueError, match="no chunks"):
            build_prompt(empty_result)


class TestSourceFormatting:
    def test_appends_sources_if_missing(self, sample_result: RetrievalResult) -> None:
        raw_answer = "Click forgot password on the page."
        final_answer = format_answer_with_sources(raw_answer, sample_result)
        assert "Sources:" in final_answer
        assert "account_login.md — page 1" in final_answer

    def test_does_not_duplicate_sources_if_already_present(self, sample_result: RetrievalResult) -> None:
        raw_answer = "Click forgot password.\n\nSources:\n• account_login.md — page 1"
        final_answer = format_answer_with_sources(raw_answer, sample_result)
        # Verify the sources block is not duplicated
        assert final_answer.lower().count("sources:") == 1
