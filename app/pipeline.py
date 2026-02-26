"""Two-stage pipeline orchestrator: Retrieval → Reranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings
from app.ingestion.resume_loader import load_resumes
from app.reranker.llm_reranker import LLMReranker, RankedCandidate
from app.retriever.faiss_retriever import FAISSRetriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full pipeline output including both stages."""

    retrieval_results: list[RetrievalResult]
    ranked_candidates: list[RankedCandidate]


class MatchingPipeline:
    """Orchestrates the two-stage clinical credential matching pipeline.

    Stage 1: FAISS retrieval (cosine similarity) to get top-N candidates.
    Stage 2: LLM reranking with recruiter-level analysis.

    Args:
        settings: Application settings.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.retriever = FAISSRetriever(
            embedding_model=settings.embedding_model,
            ollama_base_url=settings.ollama_base_url,
            dimension=settings.embedding_dimension,
        )
        self.reranker = LLMReranker(
            llm_model=settings.llm_model,
            ollama_base_url=settings.ollama_base_url,
            request_timeout=settings.llm_request_timeout,
            max_concurrency=settings.llm_max_concurrency,
        )
        self._index_built = False

    def ingest_resumes(self, resumes_dir: str | None = None) -> int:
        """Load and index resumes from disk.

        Args:
            resumes_dir: Path to resumes directory. Defaults to settings value.

        Returns:
            Number of resumes indexed.
        """
        directory = resumes_dir or self.settings.resumes_dir
        documents = load_resumes(directory)
        logger.info("Loaded %d resumes from %s", len(documents), directory)

        self.retriever.build_index(documents)
        self._index_built = True
        logger.info("FAISS index built with %d documents", len(documents))

        # Persist index to disk
        self.retriever.save_index(self.settings.faiss_index_path)
        logger.info("FAISS index saved to %s", self.settings.faiss_index_path)

        return len(documents)

    def load_index(self) -> None:
        """Load a previously saved FAISS index."""
        self.retriever.load_index(self.settings.faiss_index_path)
        self._index_built = True
        logger.info("FAISS index loaded from %s", self.settings.faiss_index_path)

    async def match(
        self,
        job_text: str,
        top_k: int | None = None,
        llm_model: str | None = None,
        market: str | None = None,
    ) -> PipelineResult:
        """Run the full two-stage matching pipeline.

        Args:
            job_text: Job posting text to match against.
            top_k: Number of candidates to retrieve in Stage 1. Defaults to settings value.
            llm_model: Optional model override for Stage 2 reranking.
            market: Optional market filter (``UK`` or ``US``) for Stage 1 retrieval.

        Returns:
            PipelineResult with retrieval and ranking results.

        Raises:
            RuntimeError: If the index has not been built or loaded.
        """
        if not self._index_built:
            # Try to load a saved index
            try:
                self.load_index()
            except FileNotFoundError:
                raise RuntimeError(
                    "No FAISS index available. Call ingest_resumes() or load_index() first."
                )

        k = top_k or self.settings.top_k

        # Stage 1: Retrieval
        logger.info("Stage 1: Retrieving top-%d candidates via FAISS", k)
        retrieval_results = self.retriever.query(job_text, top_k=k, market=market)
        logger.info("Stage 1 complete: %d candidates retrieved", len(retrieval_results))

        # Stage 2: LLM Reranking
        candidates = [
            {"resume_id": r.resume_id, "text": r.text} for r in retrieval_results
        ]
        logger.info("Stage 2: Sending %d candidates to LLM for reranking", len(candidates))
        ranked = await self.reranker.rerank(job_text, candidates, llm_model=llm_model)
        logger.info("Stage 2 complete: %d candidates evaluated", len(ranked))

        return PipelineResult(
            retrieval_results=retrieval_results,
            ranked_candidates=ranked,
        )
