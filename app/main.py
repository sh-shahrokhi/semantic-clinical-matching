"""FastAPI application — REST API for the clinical credential matcher."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.models import (
    CandidateResult,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MatchRequest,
    MatchResponse,
    ModelsResponse,
    RetrievalResultSchema,
)
from app.pipeline import MatchingPipeline
from app.reranker.llm_reranker import LLMResponseError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="Semantic Clinical Matching",
    description=(
        "Two-stage NLP pipeline for matching medical professionals"
        " to clinical job postings."
    ),
    version="0.1.0",
)

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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


def _list_ollama_models(base_url: str) -> list[str]:
    """Return installed Ollama model names from /api/tags."""
    url = f"{base_url.rstrip('/')}/api/tags"

    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
    except httpx.RequestError as e:
        raise RuntimeError(f"Cannot reach Ollama at {base_url}: {e}") from e
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        raise RuntimeError(f"Ollama returned HTTP {status_code} for {url}") from e

    payload = response.json()
    raw_models = payload.get("models", [])
    if not isinstance(raw_models, list):
        raise RuntimeError("Invalid response from Ollama: 'models' must be a list")

    model_names: list[str] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_name = item.get("model") or item.get("name")
        if not isinstance(model_name, str) or not model_name:
            continue
        model_name_lc = model_name.lower()
        if "embed" in model_name_lc:
            continue
        if model_name_lc.endswith("-embedding"):
            continue
        details = item.get("details", {})
        families = []
        if isinstance(details, dict):
            family = details.get("family")
            if isinstance(family, str):
                families.append(family.lower())
            families_value = details.get("families")
            if isinstance(families_value, list):
                families.extend(
                    fam.lower() for fam in families_value if isinstance(fam, str)
                )
        if any("bert" in family for family in families):
            continue
        if any("embed" in family for family in families):
            continue
        if any("clip" in family for family in families):
            continue
        if isinstance(model_name, str) and model_name:
            model_names.append(model_name)

    return sorted(set(model_names))


# --- Endpoints ---


@app.get("/", include_in_schema=False)
async def root():
    """Serve the web UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """Return installed Ollama model names for UI model selection."""
    settings = _get_settings()

    try:
        models = _list_ollama_models(settings.ollama_base_url)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ModelsResponse(models=models, default_model=settings.llm_model)


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
        result = await pipeline.match(
            request.job_text,
            request.top_k,
            llm_model=request.llm_model,
        )

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
    except LLMResponseError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM returned invalid JSON: {e}",
        )
    except Exception as e:
        logger.exception("Error during matching")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/random-job")
async def get_random_job():
    """Returns a random job posting from the processed dataset."""
    jobs_dir = Path("data/processed/jobs")
    if not jobs_dir.exists():
        raise HTTPException(status_code=404, detail="Jobs directory not found")
    
    job_files = list(jobs_dir.glob("*.txt"))
    if not job_files:
        raise HTTPException(status_code=404, detail="No job postings found")
        
    random_job_file = random.choice(job_files)
    text = random_job_file.read_text(encoding="utf-8")
    
    return {"text": text}
