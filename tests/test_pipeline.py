"""Integration tests for the full matching pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.pipeline import MatchingPipeline


class TestMatchingPipeline:
    def test_ingest_and_match(
        self, test_settings, sample_resume_text,
        sample_resume_text_2, sample_job_text,
    ):
        """Test full pipeline: ingest resumes, then match a job posting."""
        # Create sample resume files
        resumes_dir = test_settings.resumes_path
        resumes_dir.mkdir(parents=True)
        (resumes_dir / "resume_001.txt").write_text(sample_resume_text)
        (resumes_dir / "resume_002.txt").write_text(sample_resume_text_2)

        mock_embeddings = [
            np.random.default_rng(42).random(768).tolist(),
            np.random.default_rng(43).random(768).tolist(),
        ]
        query_embedding = np.random.default_rng(42).random(768).tolist()

        llm_response_json = """[
          {"resume_id": "resume_001", "rank": 1, "status": "PASS",
           "skill_overlaps": ["ICU experience"], "missing_criteria": [],
           "reasoning": "Strong match."},
          {"resume_id": "resume_002", "rank": null, "status": "FAIL",
           "skill_overlaps": [], "missing_criteria": ["CARNA registration"],
           "reasoning": "Wrong specialty and province."}
        ]"""

        pipeline = MatchingPipeline(test_settings)

        # Replace Pydantic models with plain mocks
        mock_embed = MagicMock()
        mock_embed.get_text_embedding_batch.return_value = mock_embeddings
        mock_embed.get_text_embedding.return_value = query_embedding
        pipeline.retriever.embed_model = mock_embed

        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(
            text=llm_response_json,
        )
        pipeline.reranker.llm = mock_llm

        count = pipeline.ingest_resumes()
        assert count == 2

        result = pipeline.match(sample_job_text, top_k=2)

        assert len(result.retrieval_results) == 2
        assert len(result.ranked_candidates) == 2
        assert result.ranked_candidates[0].status == "PASS"
        assert result.ranked_candidates[1].status == "FAIL"

    def test_match_without_index_raises(
        self, test_settings, sample_job_text,
    ):
        """Test that matching without an index raises RuntimeError."""
        pipeline = MatchingPipeline(test_settings)
        with pytest.raises(RuntimeError, match="No FAISS index available"):
            pipeline.match(sample_job_text)
