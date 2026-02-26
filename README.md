# Semantic Clinical Matching

A FastAPI application for resume-to-job matching using a two-stage pipeline:

1. semantic retrieval with FAISS + embeddings
2. LLM-based recruiter evaluation with structured PASS/FAIL reasoning

The project is optimized for local inference with Ollama and includes a UK-focused demo workflow.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [UK Demo Workflow](#uk-demo-workflow)
- [API](#api)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Current Scope](#current-scope)

## Features

- Two-stage matching pipeline (retrieval then reranking)
- FAISS cosine-similarity search over resume embeddings
- Structured LLM output per candidate:
  - `status` (`PASS`/`FAIL`)
  - `skill_overlaps`
  - `missing_criteria`
  - `reasoning`
- Role-aware reranker policy:
  - strict clinical evaluation for clinical-care jobs
  - non-clinical policy for healthcare sales/admin/ops jobs
- Optional market-aware retrieval filter (`UK`/`US`)
- REST API + lightweight web UI
- Test coverage for retriever, reranker, pipeline, and API

## Architecture

```text
FastAPI
  ├─ POST /resumes/ingest
  ├─ POST /match
  ├─ GET  /models
  ├─ GET  /health
  └─ GET  /random-job

MatchingPipeline
  ├─ Stage 1: FAISSRetriever (Ollama embeddings)
  └─ Stage 2: LLMReranker (Ollama LLM)
```

## Quick Start

### 1) Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)

### 2) Install dependencies

```bash
git clone https://github.com/sh-shahrokhi/semantic-clinical-matching.git
cd semantic-clinical-matching
uv sync --dev
```

### 3) Pull required Ollama models

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

You can use a different generation model at request time via `llm_model` in `/match`.

### 4) Prepare data

Option A: build processed data from raw datasets

```bash
uv run python scripts/prepare_data.py --max-resumes 500 --max-jobs 200
```

Option B: build UK-focused demo subset from existing processed text

```bash
uv run python scripts/build_demo_uk_dataset.py --max-jobs 200 --max-resumes 500 --max-queries 10
```

### 5) Start the API

```bash
uv run uvicorn app.main:app --reload
```

- API docs: `http://127.0.0.1:8000/docs`
- UI: `http://127.0.0.1:8000`

### 6) Ingest resumes and run matching

Ingest:

```bash
curl -X POST "http://127.0.0.1:8000/resumes/ingest" \
  -H "Content-Type: application/json" \
  -d '{"resumes_dir":"data/processed/resumes"}'
```

Match:

```bash
curl -X POST "http://127.0.0.1:8000/match" \
  -H "Content-Type: application/json" \
  -d '{
    "job_text":"ICU Registered Nurse role in Alberta. Active license, BLS, ACLS, and 3+ years ICU experience required.",
    "top_k":5,
    "market":"UK",
    "llm_model":"llama3"
  }'
```

## UK Demo Workflow

For a weekend demo with UK-focused filtering:

1. Build UK subset:

```bash
uv run python scripts/build_demo_uk_dataset.py --max-jobs 200 --max-resumes 500 --max-queries 10
```

2. Ingest only UK demo resumes:

```bash
curl -X POST "http://127.0.0.1:8000/resumes/ingest" \
  -H "Content-Type: application/json" \
  -d '{"resumes_dir":"data/demo_uk/resumes"}'
```

3. Use one query from `data/demo_uk/demo_queries_uk.json` and call `/match` with:
- `"market": "UK"`

## API

### Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | service health |
| `GET` | `/models` | list installed non-embedding Ollama models |
| `POST` | `/resumes/ingest` | build/update FAISS index from resume text files |
| `POST` | `/match` | run retrieval + LLM reranking |
| `GET` | `/random-job` | return random processed job text |

### `POST /match` request

```json
{
  "job_text": "...",
  "top_k": 10,
  "market": "UK",
  "llm_model": "llama3"
}
```

- `top_k` optional, defaults to `SCM_TOP_K`
- `market` optional, supported values: `UK`, `US`
- `llm_model` optional model override for reranking stage

### `POST /match` response shape

```json
{
  "retrieval_results": [
    {
      "resume_id": "resume_00042",
      "score": 0.87,
      "text": "..."
    }
  ],
  "ranked_candidates": [
    {
      "resume_id": "resume_00042",
      "rank": 1,
      "status": "PASS",
      "skill_overlaps": ["..."],
      "missing_criteria": [],
      "reasoning": "..."
    }
  ]
}
```

## Configuration

All settings use the `SCM_` prefix.

| Variable | Default | Description |
|---|---|---|
| `SCM_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `SCM_EMBEDDING_MODEL` | `nomic-embed-text` | embedding model |
| `SCM_LLM_MODEL` | `llama3` | default reranker model |
| `SCM_TOP_K` | `20` | default retrieval cutoff |
| `SCM_EMBEDDING_DIMENSION` | `768` | embedding size |
| `SCM_RESUMES_DIR` | `data/processed/resumes` | default resume directory |
| `SCM_JOBS_DIR` | `data/processed/jobs` | default jobs directory |
| `SCM_FAISS_INDEX_PATH` | `data/faiss_index` | FAISS persistence path |
| `SCM_LLM_REQUEST_TIMEOUT` | `120.0` | LLM timeout in seconds |
| `SCM_LLM_MAX_CONCURRENCY` | `3` | max concurrent candidate evaluations |

## Development

Run tests:

```bash
uv run pytest -q
```

Run lint:

```bash
uv run ruff check app tests
```

Notes:
- tests mock model calls; live Ollama is not required for the test suite
- API smoke tests against real models require Ollama running locally

## Troubleshooting

### `Failed to connect to Ollama`

- Ensure Ollama is installed and running
- Check `SCM_OLLAMA_BASE_URL`
- Confirm required models are pulled

### Empty retrieval results

- Ensure `/resumes/ingest` succeeded
- Confirm resume directory has `.txt` files
- If using `market`, verify resume corpus includes that market

### Reranker outputs mostly `FAIL`

- Validate job/resume market alignment (e.g., UK jobs with UK resumes)
- Use the role-appropriate corpus (clinical-care vs non-clinical healthcare)

## Project Structure

```text
app/
  main.py
  config.py
  models.py
  pipeline.py
  ingestion/
  retriever/
  reranker/
scripts/
  prepare_data.py
  build_demo_uk_dataset.py
static/
tests/
```

## Current Scope

This project is built for local/demo operation and iterative experimentation.

- optimized for local Ollama inference
- includes UK demo tools, but corpus quality controls are still heuristic
- market filtering is metadata/text-inference based, not full geocoding
