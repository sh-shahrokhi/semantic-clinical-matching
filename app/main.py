"""FastAPI application — REST API for the clinical credential matcher."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from app.config import Settings, get_settings
from app.models import (
    CandidateResult,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MatchRequest,
    MatchResponse,
    RetrievalResultSchema,
)
from app.pipeline import MatchingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Semantic Clinical Matching",
    description=(
        "Two-stage NLP pipeline for matching medical professionals"
        " to clinical job postings."
    ),
    version="0.1.0",
)

# --- Application State ---

_settings: Settings | None = None
_pipeline: MatchingPipeline | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def _get_pipeline() -> MatchingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MatchingPipeline(_get_settings())
    return _pipeline


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/resumes/ingest", response_model=IngestResponse)
async def ingest_resumes(request: IngestRequest | None = None) -> IngestResponse:
    """Ingest resumes into the FAISS index.

    Reads text files from the configured (or specified) resumes directory,
    embeds them, and builds a FAISS index.
    """
    try:
        pipeline = _get_pipeline()
        resumes_dir = request.resumes_dir if request else None
        count = pipeline.ingest_resumes(resumes_dir)
        return IngestResponse(
            message=f"Successfully ingested {count} resumes",
            count=count,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error during ingestion")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match", response_model=MatchResponse)
async def match(request: MatchRequest) -> MatchResponse:
    """Match a job posting against indexed resumes.

    Runs the two-stage pipeline:
    1. FAISS retrieval for top-N candidates.
    2. LLM reranking with recruiter-level analysis.
    """
    try:
        pipeline = _get_pipeline()
        result = pipeline.match(request.job_text, request.top_k)

        return MatchResponse(
            retrieval_results=[
                RetrievalResultSchema(
                    resume_id=r.resume_id,
                    score=r.score,
                    text=r.text,
                )
                for r in result.retrieval_results
            ],
            ranked_candidates=[
                CandidateResult(
                    resume_id=c.resume_id,
                    rank=c.rank,
                    status=c.status,
                    skill_overlaps=c.skill_overlaps,
                    missing_criteria=c.missing_criteria,
                    reasoning=c.reasoning,
                )
                for c in result.ranked_candidates
            ],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error during matching")
        raise HTTPException(status_code=500, detail=str(e))
