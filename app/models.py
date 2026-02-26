"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field  # noqa: I001

# --- Request Schemas ---


class MatchRequest(BaseModel):
    """Request body for the /match endpoint."""

    job_text: str = Field(..., description="Full text of the job posting to match against")
    top_k: int | None = Field(None, description="Number of candidates for Stage 1 retrieval")
    market: str | None = Field(
        None,
        description=(
            "Optional candidate market filter for Stage 1 retrieval. "
            "Supported values: UK or US."
        ),
    )
    llm_model: str | None = Field(
        None,
        description=(
            "Ollama LLM model to use for Stage 2 reranking. "
            "Uses configured default if omitted."
        ),
    )


class IngestRequest(BaseModel):
    """Request body for the /resumes/ingest endpoint."""

    resumes_dir: str | None = Field(
        None,
        description="Path to resumes directory. Uses default from settings if omitted.",
    )


# --- Response Schemas ---


class RetrievalResultSchema(BaseModel):
    """A single Stage 1 retrieval result."""

    resume_id: str
    score: float
    text: str


class CandidateResult(BaseModel):
    """A single Stage 2 ranked candidate result."""

    resume_id: str
    rank: int | None
    status: str
    skill_overlaps: list[str]
    missing_criteria: list[str]
    reasoning: str


class MatchResponse(BaseModel):
    """Response body for the /match endpoint."""

    retrieval_results: list[RetrievalResultSchema]
    ranked_candidates: list[CandidateResult]


class IngestResponse(BaseModel):
    """Response body for the /resumes/ingest endpoint."""

    message: str
    count: int


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    version: str


class ModelsResponse(BaseModel):
    """Response body for the /models endpoint."""

    models: list[str]
    default_model: str
