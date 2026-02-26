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

    @staticmethod
    def _normalize_market(market: str | None) -> str | None:
        if market is None:
            return None
        normalized = market.strip().upper()
        if normalized in {"UK", "US"}:
            return normalized
        return None

    @staticmethod
    def _infer_market_from_text(text: str) -> str:
        low = f" {text.lower()} "

        uk_tokens = (
            " uk ",
            " united kingdom ",
            " nhs ",
            " england ",
            " scotland ",
            " wales ",
            " northern ireland ",
            " london ",
            " manchester ",
            " birmingham ",
            " glasgow ",
            " oxford ",
            " leeds ",
            " bristol ",
        )
        us_tokens = (
            " usa ",
            " u.s.a ",
            " united states ",
            " washington, dc ",
            ", va ",
            ", md ",
            ", tx ",
            ", ca ",
            ", ny ",
            " fort belvoir ",
            " virginia ",
            " maryland ",
            " california ",
        )

        uk_score = sum(token in low for token in uk_tokens)
        us_score = sum(token in low for token in us_tokens)

        if uk_score > us_score and uk_score > 0:
            return "UK"
        if us_score > uk_score and us_score > 0:
            return "US"
        return "UNKNOWN"

    @classmethod
    def _infer_document_market(cls, doc: Document) -> str:
        metadata = doc.metadata or {}
        for key in ("market", "country", "location", "region"):
            value = metadata.get(key)
            if isinstance(value, str):
                normalized = cls._normalize_market(value)
                if normalized is not None:
                    return normalized
                inferred = cls._infer_market_from_text(value)
                if inferred != "UNKNOWN":
                    return inferred

        inferred = cls._infer_market_from_text(doc.text)
        metadata["market"] = inferred
        doc.metadata = metadata
        return inferred

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

    def query(
        self,
        job_text: str,
        top_k: int = 20,
        market: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve most similar resumes for a given job posting.

        Args:
            job_text: Job posting text to match against.
            top_k: Number of top candidates to return.
            market: Optional market filter (``UK`` or ``US``).

        Returns:
            List of RetrievalResult ordered by descending similarity.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        requested_market = self._normalize_market(market)

        query_embedding = self.embed_model.get_text_embedding(job_text)
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        if requested_market is None:
            search_k = min(top_k, len(self.documents))
        else:
            # Metadata filters are applied after vector search, so fetch all.
            search_k = len(self.documents)

        scores, indices = self.index.search(query_vec, search_k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            doc_market = self._infer_document_market(doc)
            if requested_market is not None and doc_market != requested_market:
                continue
            results.append(
                RetrievalResult(
                    resume_id=doc.metadata.get("resume_id", f"doc_{idx}"),
                    text=doc.text,
                    score=float(score),
                    metadata=doc.metadata,
                )
            )
            if len(results) >= top_k:
                break
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
