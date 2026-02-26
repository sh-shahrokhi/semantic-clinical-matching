"""Tests for the LLM reranker (Stage 2)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from app.reranker.llm_reranker import (
    LLMReranker,
    RankedCandidate,
)
from app.reranker.prompts import (
    SINGLE_CANDIDATE_SYSTEM_PROMPT,
    SINGLE_CANDIDATE_SYSTEM_PROMPT_NONCLINICAL,
    build_reranker_prompt,
    build_single_candidate_prompt,
)


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

    def test_build_single_candidate_prompt(self, sample_job_text):
        """Test single candidate prompt includes all fields."""
        prompt = build_single_candidate_prompt(
            sample_job_text, "resume_001", "Some resume text",
        )
        assert "ICU Registered Nurse" in prompt
        assert "resume_001" in prompt
        assert "Some resume text" in prompt


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

    def test_parse_single_response_object(self):
        """Test parsing a single candidate JSON object."""
        json_str = (
            '{"resume_id":"r1","status":"PASS",'
            '"skill_overlaps":["X"],"missing_criteria":[],'
            '"reasoning":"Good."}'
        )
        result = LLMReranker._parse_single_response(json_str, "r1")
        assert result.resume_id == "r1"
        assert result.status == "PASS"
        assert result.rank is None
        assert result.skill_overlaps == ["X"]

    def test_parse_single_response_array_fallback(self):
        """Test that a one-element array is accepted by _parse_single_response."""
        json_str = (
            '[{"resume_id":"r1","status":"FAIL",'
            '"skill_overlaps":[],"missing_criteria":["license"],'
            '"reasoning":"Missing."}]'
        )
        result = LLMReranker._parse_single_response(json_str, "r1")
        assert result.status == "FAIL"

    def test_parse_single_response_invalid_json_raises(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="not valid JSON"):
            LLMReranker._parse_single_response("garbage", "r1")

    def test_parse_single_response_pass_with_missing_criteria_becomes_fail(self):
        """PASS status is overridden when required criteria are missing."""
        json_str = (
            '{"resume_id":"r1","status":"PASS",'
            '"skill_overlaps":["nursing"],"missing_criteria":["license"],'
            '"reasoning":"Missing required license."}'
        )
        result = LLMReranker._parse_single_response(json_str, "r1")
        assert result.status == "FAIL"

    def test_parse_single_response_normalizes_scalar_fields(self):
        """Scalar/list-mismatched fields are normalized to expected types."""
        json_str = (
            '{"resume_id":"r1","status":"pass",'
            '"skill_overlaps":"nursing","missing_criteria":"","reasoning":123}'
        )
        result = LLMReranker._parse_single_response(json_str, "r1")
        assert result.status == "PASS"
        assert result.skill_overlaps == ["nursing"]
        assert result.missing_criteria == []
        assert result.reasoning == "123"


class TestLLMRerankerRuntime:
    def _make_single_response(self, resume_id, status="PASS"):
        """Helper to create a mock single-candidate LLM response."""
        return MagicMock(
            text=(
                f'{{"resume_id":"{resume_id}","status":"{status}",'
                f'"skill_overlaps":["nursing"],"missing_criteria":[],'
                f'"reasoning":"Evaluated."}}'
            )
        )

    def test_rerank_single_candidate_pass(self):
        reranker = LLMReranker()
        reranker.llm = MagicMock()
        reranker.llm.complete.return_value = self._make_single_response("r1", "PASS")

        results = asyncio.run(reranker.rerank(
            "Job text",
            [{"resume_id": "r1", "text": "Resume text"}],
        ))

        assert len(results) == 1
        assert results[0].resume_id == "r1"
        assert results[0].status == "PASS"
        assert results[0].rank == 1

    def test_role_router_detects_nonclinical_sales_job(self):
        job_text = (
            "Medical Sales Representative role. "
            "Territory account management and business development required."
        )
        assert LLMReranker._is_nonclinical_role(job_text) is True
        assert (
            LLMReranker._select_single_candidate_system_prompt(job_text)
            == SINGLE_CANDIDATE_SYSTEM_PROMPT_NONCLINICAL
        )

    def test_role_router_keeps_clinical_policy_for_nursing_job(self):
        job_text = (
            "ICU Registered Nurse role. Active RN license and ACLS required. "
            "3+ years bedside patient care experience."
        )
        assert LLMReranker._is_nonclinical_role(job_text) is False
        assert (
            LLMReranker._select_single_candidate_system_prompt(job_text)
            == SINGLE_CANDIDATE_SYSTEM_PROMPT
        )

    def test_rerank_inlines_system_prompt_into_completion_prompt(self):
        reranker = LLMReranker()
        reranker.llm = MagicMock()
        reranker.llm.complete.return_value = self._make_single_response("r1", "PASS")

        asyncio.run(reranker.rerank(
            "Job text",
            [{"resume_id": "r1", "text": "Resume text"}],
        ))

        kwargs = reranker.llm.complete.call_args.kwargs
        assert "system_prompt" not in kwargs
        assert SINGLE_CANDIDATE_SYSTEM_PROMPT.splitlines()[0] in kwargs["prompt"]

    def test_rerank_uses_nonclinical_prompt_for_sales_job(self):
        reranker = LLMReranker()
        reranker.llm = MagicMock()
        reranker.llm.complete.return_value = self._make_single_response("r1", "PASS")

        asyncio.run(reranker.rerank(
            "Medical Sales Representative role with territory account ownership.",
            [{"resume_id": "r1", "text": "Resume text"}],
        ))

        kwargs = reranker.llm.complete.call_args.kwargs
        assert SINGLE_CANDIDATE_SYSTEM_PROMPT_NONCLINICAL.splitlines()[0] in kwargs["prompt"]

    def test_rerank_multiple_candidates_triggers_ranking(self):
        reranker = LLMReranker()
        reranker.llm = MagicMock()
        # First two calls: per-candidate evaluation (both PASS)
        # Third call: ranking
        reranker.llm.complete.side_effect = [
            self._make_single_response("r1", "PASS"),
            self._make_single_response("r2", "PASS"),
            MagicMock(
                text='[{"resume_id":"r1","rank":2},{"resume_id":"r2","rank":1}]'
            ),
        ]

        results = asyncio.run(reranker.rerank(
            "Job text",
            [
                {"resume_id": "r1", "text": "Resume 1"},
                {"resume_id": "r2", "text": "Resume 2"},
            ],
        ))

        passing = [r for r in results if r.status == "PASS"]
        assert len(passing) == 2
        assert passing[0].resume_id == "r2"
        assert passing[0].rank == 1
        assert passing[1].resume_id == "r1"
        assert passing[1].rank == 2

    def test_rerank_mixed_pass_fail(self):
        reranker = LLMReranker()
        reranker.llm = MagicMock()
        reranker.llm.complete.side_effect = [
            self._make_single_response("r1", "PASS"),
            MagicMock(
                text=(
                    '{"resume_id":"r2","status":"FAIL",'
                    '"skill_overlaps":[],"missing_criteria":["license"],'
                    '"reasoning":"Missing license."}'
                )
            ),
        ]

        results = asyncio.run(reranker.rerank(
            "Job text",
            [
                {"resume_id": "r1", "text": "Resume 1"},
                {"resume_id": "r2", "text": "Resume 2"},
            ],
        ))

        assert len(results) == 2
        assert results[0].status == "PASS"
        assert results[0].rank == 1
        assert results[1].status == "FAIL"

    def test_rerank_retries_and_recovers_json(self):
        reranker = LLMReranker()
        reranker.llm = MagicMock()
        reranker.llm.complete.side_effect = [
            # First call: malformed output
            MagicMock(text="Output was malformed"),
            # Repair call: valid JSON
            MagicMock(
                text=(
                    '{"resume_id":"r1","status":"PASS",'
                    '"skill_overlaps":[],"missing_criteria":[],"reasoning":"ok"}'
                )
            ),
        ]

        results = asyncio.run(reranker.rerank(
            "Job text",
            [{"resume_id": "r1", "text": "Resume text"}],
        ))

        assert len(results) == 1
        assert results[0].resume_id == "r1"
        assert reranker.llm.complete.call_count == 2

    def test_rerank_empty_response_creates_fail(self):
        reranker = LLMReranker()
        reranker.llm = MagicMock()
        reranker.llm.complete.return_value = MagicMock(text="   ")

        results = asyncio.run(reranker.rerank(
            "Job text",
            [{"resume_id": "r1", "text": "Resume text"}],
        ))

        # Exception is caught and turned into a FAIL result
        assert len(results) == 1
        assert results[0].status == "FAIL"
        assert "evaluation_error" in results[0].missing_criteria

    def test_rerank_ranking_failure_falls_back(self):
        """When ranking call fails, candidates get insertion-order ranks."""
        reranker = LLMReranker()
        reranker.llm = MagicMock()
        reranker.llm.complete.side_effect = [
            self._make_single_response("r1", "PASS"),
            self._make_single_response("r2", "PASS"),
            # Ranking call fails with invalid JSON
            MagicMock(text="not json"),
        ]

        results = asyncio.run(reranker.rerank(
            "Job text",
            [
                {"resume_id": "r1", "text": "Resume 1"},
                {"resume_id": "r2", "text": "Resume 2"},
            ],
        ))

        passing = [r for r in results if r.status == "PASS"]
        assert len(passing) == 2
        assert passing[0].rank == 1
        assert passing[1].rank == 2
