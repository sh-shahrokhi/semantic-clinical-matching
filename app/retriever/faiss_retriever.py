"""Stage 1 — FAISS vector retrieval for candidate resumes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding


@dataclass
class RetrievalResult:
    """A single retrieval result with candidate metadata and similarity score."""

    resume_id: str
    text: str
    score: float
    metadata: dict


class FAISSRetriever:
    """Embed resumes and retrieve top-N candidates for a job posting using FAISS.

    Args:
        embedding_model: Name of the Ollama embedding model.
        ollama_base_url: Ollama server URL.
        dimension: Embedding vector dimension.
    """

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        dimension: int = 768,
    ) -> None:
        self.embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=ollama_base_url,
        )
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[Document] = []

    def build_index(self, documents: list[Document]) -> None:
        """Embed all documents and build a FAISS inner-product index (cosine sim on L2-normed vecs).

        Args:
            documents: LlamaIndex Documents to index.
        """
        self.documents = documents
        texts = [doc.text for doc in documents]

        # Get embeddings from Ollama
        embeddings = self.embed_model.get_text_embedding_batch(texts)
        vectors = np.array(embeddings, dtype=np.float32)

        # L2-normalize so inner product == cosine similarity
        faiss.normalize_L2(vectors)

        # Build index
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def query(self, job_text: str, top_k: int = 20) -> list[RetrievalResult]:
        """Retrieve most similar resumes for a given job posting.

        Args:
            job_text: Job posting text to match against.
            top_k: Number of top candidates to return.

        Returns:
            List of RetrievalResult ordered by descending similarity.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_embedding = self.embed_model.get_text_embedding(job_text)
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, min(top_k, len(self.documents)))

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append(
                RetrievalResult(
                    resume_id=doc.metadata.get("resume_id", f"doc_{idx}"),
                    text=doc.text,
                    score=float(score),
                    metadata=doc.metadata,
                )
            )
        return results

    def save_index(self, path: str | Path) -> None:
        """Persist the FAISS index and document metadata to disk.

        Args:
            path: Directory to save the index files.
        """
        if self.index is None:
            raise RuntimeError("No index to save.")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_path / "index.faiss"))

        # Save document metadata
        meta = [
            {
                "resume_id": doc.metadata.get("resume_id", ""),
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for doc in self.documents
        ]
        (save_path / "documents.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load_index(self, path: str | Path) -> None:
        """Load a previously saved FAISS index and document metadata.

        Args:
            path: Directory containing saved index files.
        """
        load_path = Path(path)
        index_file = load_path / "index.faiss"
        docs_file = load_path / "documents.json"

        if not index_file.exists() or not docs_file.exists():
            raise FileNotFoundError(f"Index files not found in {load_path}")

        self.index = faiss.read_index(str(index_file))

        meta = json.loads(docs_file.read_text(encoding="utf-8"))
        self.documents = [
            Document(text=m["text"], metadata=m.get("metadata", {})) for m in meta
        ]
