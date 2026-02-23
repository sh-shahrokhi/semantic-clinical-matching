"""Tests for the LLM reranker (Stage 2)."""

from __future__ import annotations

import pytest

from app.reranker.llm_reranker import LLMReranker, RankedCandidate
from app.reranker.prompts import build_reranker_prompt


class TestPrompts:
    def test_build_reranker_prompt(self, sample_job_text):
        """Test that the reranker prompt includes job text and candidate info."""
        candidates = [
            {"resume_id": "resume_001", "text": "Some resume text"},
            {"resume_id": "resume_002", "text": "Another resume text"},
        ]
        prompt = build_reranker_prompt(sample_job_text, candidates)

        assert "ICU Registered Nurse" in prompt
        assert "resume_001" in prompt
        assert "resume_002" in prompt
        assert "Some resume text" in prompt

    def test_prompt_contains_separator(self, sample_job_text):
        """Test that candidates are separated in the prompt."""
        candidates = [
            {"resume_id": "r1", "text": "Text 1"},
            {"resume_id": "r2", "text": "Text 2"},
        ]
        prompt = build_reranker_prompt(sample_job_text, candidates)
        assert "---" in prompt


class TestLLMRerankerParsing:
    def test_parse_valid_response(self, sample_llm_response):
        """Test parsing a well-formed LLM JSON response."""
        results = LLMReranker._parse_response(sample_llm_response)

        assert len(results) == 2
        assert all(isinstance(r, RankedCandidate) for r in results)

        pass_candidate = results[0]
        assert pass_candidate.resume_id == "resume_001"
        assert pass_candidate.status == "PASS"
        assert pass_candidate.rank == 1
        assert "ICU experience" in pass_candidate.skill_overlaps
        assert len(pass_candidate.missing_criteria) == 0

        fail_candidate = results[1]
        assert fail_candidate.status == "FAIL"
        assert fail_candidate.rank is None
        assert "CARNA registration" in fail_candidate.missing_criteria

    def test_parse_response_with_markdown_fences(self, sample_llm_response):
        """Test parsing when LLM wraps response in markdown code fences."""
        fenced = f"```json\n{sample_llm_response}\n```"
        results = LLMReranker._parse_response(fenced)
        assert len(results) == 2

    def test_parse_invalid_json_raises(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="not valid JSON"):
            LLMReranker._parse_response("this is not json at all")

    def test_parse_non_array_raises(self):
        """Test that a JSON object (not array) raises ValueError."""
        with pytest.raises(ValueError, match="must be a JSON array"):
            LLMReranker._parse_response('{"result": "not an array"}')

    def test_parse_empty_array(self):
        """Test parsing an empty JSON array."""
        results = LLMReranker._parse_response("[]")
        assert results == []
