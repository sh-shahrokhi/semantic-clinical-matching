"""Tests for the FAISS retriever (Stage 1)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from llama_index.core.schema import Document

from app.retriever.faiss_retriever import FAISSRetriever, RetrievalResult


@pytest.fixture
def mock_embeddings():
    """Return deterministic mock embeddings for testing."""
    return [
        np.random.default_rng(42).random(768).tolist(),
        np.random.default_rng(43).random(768).tolist(),
        np.random.default_rng(44).random(768).tolist(),
    ]


@pytest.fixture
def sample_documents(sample_resume_text, sample_resume_text_2):
    """Create sample LlamaIndex Documents."""
    return [
        Document(
            text=sample_resume_text,
            metadata={"resume_id": "resume_001"},
        ),
        Document(
            text=sample_resume_text_2,
            metadata={"resume_id": "resume_002"},
        ),
    ]


class TestFAISSRetriever:
    def test_build_index_and_query(
        self, sample_documents, mock_embeddings,
    ):
        """Test building an index and querying it returns results."""
        retriever = FAISSRetriever()

        # Replace Pydantic embed_model with a plain mock
        mock_embed = MagicMock()
        mock_embed.get_text_embedding_batch.return_value = (
            mock_embeddings[:2]
        )
        mock_embed.get_text_embedding.return_value = mock_embeddings[0]
        retriever.embed_model = mock_embed

        retriever.build_index(sample_documents)
        results = retriever.query("ICU nurse Alberta", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Sorted by score descending
        assert results[0].score >= results[1].score
        assert results[0].resume_id in {"resume_001", "resume_002"}

    def test_query_without_index_raises(self):
        """Test querying without building an index raises RuntimeError."""
        retriever = FAISSRetriever()
        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.query("ICU nurse")

    def test_save_and_load_index(
        self, tmp_path, sample_documents, mock_embeddings,
    ):
        """Test saving and loading a FAISS index."""
        retriever = FAISSRetriever()

        mock_embed = MagicMock()
        mock_embed.get_text_embedding_batch.return_value = (
            mock_embeddings[:2]
        )
        retriever.embed_model = mock_embed

        retriever.build_index(sample_documents)
        retriever.save_index(tmp_path / "test_index")

        # Load into a new retriever
        retriever2 = FAISSRetriever()
        retriever2.load_index(tmp_path / "test_index")

        assert retriever2.index is not None
        assert len(retriever2.documents) == 2
        assert retriever2.documents[0].metadata["resume_id"] == "resume_001"

    def test_load_nonexistent_index_raises(self, tmp_path):
        """Test loading from a nonexistent path raises FileNotFoundError."""
        retriever = FAISSRetriever()
        with pytest.raises(FileNotFoundError):
            retriever.load_index(tmp_path / "nonexistent")

    def test_save_without_index_raises(self, tmp_path):
        """Test saving without an index raises RuntimeError."""
        retriever = FAISSRetriever()
        with pytest.raises(RuntimeError, match="No index to save"):
            retriever.save_index(tmp_path / "test_index")
