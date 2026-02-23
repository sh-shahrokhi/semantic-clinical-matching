"""API endpoint tests using FastAPI TestClient."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import _get_pipeline, app  # noqa: I001


@pytest.fixture
def client(test_settings, tmp_path):
    """Create a TestClient with test settings and mocked pipeline."""
    # Create sample resume files
    resumes_dir = test_settings.resumes_path
    resumes_dir.mkdir(parents=True)
    (resumes_dir / "resume_001.txt").write_text(
        "Sample resume content for testing"
    )

    # Override the settings and pipeline globals
    import app.main as main_module

    main_module._settings = test_settings
    main_module._pipeline = None

    with TestClient(app) as c:
        yield c

    # Cleanup
    main_module._settings = None
    main_module._pipeline = None


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


class TestIngestEndpoint:
    def test_ingest_resumes(self, client, test_settings):
        mock_embeddings = [
            np.random.default_rng(42).random(768).tolist(),
        ]

        pipeline = _get_pipeline()

        # Replace Pydantic embed_model with a plain mock
        mock_embed = MagicMock()
        mock_embed.get_text_embedding_batch.return_value = (
            mock_embeddings
        )
        pipeline.retriever.embed_model = mock_embed

        resp = client.post("/resumes/ingest")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert "Successfully ingested" in data["message"]

    def test_ingest_nonexistent_dir(self, client):
        resp = client.post(
            "/resumes/ingest",
            json={"resumes_dir": "/nonexistent/path"},
        )
        assert resp.status_code == 404


class TestMatchEndpoint:
    def test_match_without_index(self, client):
        resp = client.post(
            "/match", json={"job_text": "ICU Nurse needed"},
        )
        assert resp.status_code == 400

    def test_match_with_index(self, client, test_settings):
        mock_embeddings = [
            np.random.default_rng(42).random(768).tolist(),
        ]
        query_embedding = np.random.default_rng(42).random(768).tolist()

        llm_response = """[
          {"resume_id": "resume_001", "rank": 1, "status": "PASS",
           "skill_overlaps": ["nursing"], "missing_criteria": [],
           "reasoning": "Good match."}
        ]"""

        pipeline = _get_pipeline()

        # Replace Pydantic models with plain mocks
        mock_embed = MagicMock()
        mock_embed.get_text_embedding_batch.return_value = (
            mock_embeddings
        )
        mock_embed.get_text_embedding.return_value = query_embedding
        pipeline.retriever.embed_model = mock_embed

        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(
            text=llm_response,
        )
        pipeline.reranker.llm = mock_llm

        # First ingest
        client.post("/resumes/ingest")
        # Then match
        resp = client.post(
            "/match",
            json={"job_text": "ICU Nurse needed in Calgary"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["retrieval_results"]) == 1
        assert len(data["ranked_candidates"]) == 1
        assert data["ranked_candidates"][0]["status"] == "PASS"
