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


class TestModelsEndpoint:
    def test_models(self, client, monkeypatch):
        import app.main as main_module

        monkeypatch.setattr(
            main_module,
            "_list_ollama_models",
            lambda _: ["llama3.2:latest", "mistral:latest"],
        )

        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models"] == ["llama3.2:latest", "mistral:latest"]
        assert data["default_model"] == "llama3"

    def test_models_ollama_unavailable(self, client, monkeypatch):
        import app.main as main_module

        def _raise(_base_url: str) -> list[str]:
            raise RuntimeError("Cannot reach Ollama")

        monkeypatch.setattr(main_module, "_list_ollama_models", _raise)

        resp = client.get("/models")
        assert resp.status_code == 503
        assert "Cannot reach Ollama" in resp.json()["detail"]

    def test_models_filters_embedding_models(self, client, monkeypatch):
        import app.main as main_module

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "models": [
                {"name": "nomic-embed-text:latest"},
                {"name": "llama3.2:latest"},
                {"model": "mistral:7b"},
            ]
        }

        monkeypatch.setattr(
            main_module.httpx, "get", lambda _url, timeout: mock_resp
        )

        resp = client.get("/models")
        assert resp.status_code == 200
        assert resp.json()["models"] == ["llama3.2:latest", "mistral:7b"]


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

        llm_pass = MagicMock(
            text=(
                '{"resume_id":"resume_001","status":"PASS",'
                '"skill_overlaps":["nursing"],"missing_criteria":[],'
                '"reasoning":"Good match."}'
            ),
        )

        pipeline = _get_pipeline()

        # Replace Pydantic models with plain mocks
        mock_embed = MagicMock()
        mock_embed.get_text_embedding_batch.return_value = (
            mock_embeddings
        )
        mock_embed.get_text_embedding.return_value = query_embedding
        pipeline.retriever.embed_model = mock_embed

        mock_llm = MagicMock()
        mock_llm.complete.return_value = llm_pass
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

    def test_match_with_selected_model(self, client, test_settings):
        mock_embeddings = [
            np.random.default_rng(42).random(768).tolist(),
        ]
        query_embedding = np.random.default_rng(42).random(768).tolist()

        llm_pass = MagicMock(
            text=(
                '{"resume_id":"resume_001","status":"PASS",'
                '"skill_overlaps":["nursing"],"missing_criteria":[],'
                '"reasoning":"Good match."}'
            ),
        )

        pipeline = _get_pipeline()

        mock_embed = MagicMock()
        mock_embed.get_text_embedding_batch.return_value = (
            mock_embeddings
        )
        mock_embed.get_text_embedding.return_value = query_embedding
        pipeline.retriever.embed_model = mock_embed

        override_llm = MagicMock()
        override_llm.complete.return_value = llm_pass
        pipeline.reranker._build_client = MagicMock(
            return_value=override_llm
        )

        client.post("/resumes/ingest")
        resp = client.post(
            "/match",
            json={
                "job_text": "ICU Nurse needed in Calgary",
                "llm_model": "mistral:latest",
            },
        )

        assert resp.status_code == 200
        pipeline.reranker._build_client.assert_called_once_with(
            "mistral:latest"
        )

    def test_match_empty_llm_response_returns_fail_candidate(
        self, client, test_settings
    ):
        mock_embeddings = [
            np.random.default_rng(42).random(768).tolist(),
        ]
        query_embedding = np.random.default_rng(42).random(768).tolist()

        pipeline = _get_pipeline()

        mock_embed = MagicMock()
        mock_embed.get_text_embedding_batch.return_value = mock_embeddings
        mock_embed.get_text_embedding.return_value = query_embedding
        pipeline.retriever.embed_model = mock_embed

        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(text=" ")
        pipeline.reranker.llm = mock_llm

        client.post("/resumes/ingest")
        resp = client.post(
            "/match",
            json={"job_text": "ICU Nurse needed in Calgary"},
        )

        # Empty responses are now caught and turned into FAIL candidates
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["ranked_candidates"]) == 1
        assert data["ranked_candidates"][0]["status"] == "FAIL"
        assert "evaluation_error" in data["ranked_candidates"][0]["missing_criteria"]

