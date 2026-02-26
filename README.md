# 🏥 Semantic Clinical Matching

A two-stage NLP pipeline that matches medical professionals to clinical job postings using **vector retrieval** and **LLM-powered recruiter analysis**.

> **Stage 1** — FAISS cosine-similarity search narrows 500+ resumes to the top-N candidates.
> **Stage 2** — An LLM (via Ollama) acts as a strict medical recruiter, evaluating licensing, certifications, experience, and role fit — then returns structured, explainable rankings.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI REST API                        │
│           /health    /resumes/ingest    /match              │
└────────────────────────┬────────────────────────────────────┘
                         │
                  ┌──────▼──────┐
                  │  Pipeline   │
                  │ Orchestrator│
                  └──┬──────┬───┘
                     │      │
          ┌──────────▼┐  ┌──▼──────────┐
          │  Stage 1  │  │   Stage 2   │
          │   FAISS   │  │ LLM Reranker│
          │ Retriever │  │  (Ollama)   │
          └─────┬─────┘  └──────┬──────┘
                │               │
       ┌────────▼────┐   ┌──────▼─────┐
       │  nomic-     │   │   llama3   │
       │  embed-text │   │  (strict   │
       │  embeddings │   │  recruiter │
       │  (Ollama)   │   │  prompt)   │
       └─────────────┘   └────────────┘
```

### How it works

1. **Ingest** — Resume `.txt` files are loaded, embedded with `nomic-embed-text` via Ollama, and stored in a FAISS inner-product index (L2-normalized = cosine similarity).
2. **Match** — A job posting is embedded and searched against the index to retrieve the top-N candidates.
3. **Rerank** — The top-N candidates are sent to an LLM with a strict medical-recruiter system prompt. The LLM evaluates each candidate on licensing, certifications, experience, and specialty alignment, then returns a structured JSON array of PASS/FAIL decisions with reasoning.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Frontend | Vanilla HTML / CSS / JS (dark glassmorphism theme) |
| API Framework | FastAPI + Uvicorn |
| Embeddings | LlamaIndex + Ollama (`nomic-embed-text`) |
| Vector Search | FAISS (CPU) |
| LLM Reranking | LlamaIndex + Ollama (`llama3`) |
| Configuration | pydantic-settings |
| Package Manager | uv |
| Testing | pytest |
| Linting | Ruff |

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **[Ollama](https://ollama.com/)** — local LLM inference server

### 1. Install dependencies

```bash
git clone <repo-url>
cd semantic-clinical-matching
uv sync --dev
```

### 2. Pull Ollama models

```bash
ollama pull nomic-embed-text   # Embedding model (Stage 1)
ollama pull llama3             # LLM for reranking (Stage 2)
```

### 3. Prepare the dataset

The project uses two Kaggle datasets that should be extracted into `data/raw/`:

| Dataset | Source | Contents |
|---------|--------|----------|
| **54k Resume Dataset** | [suriyaganesh/54k-resume-dataset-structured](https://www.kaggle.com/datasets/suriyaganesh/54k-resume-dataset-structured) | 6 normalized CSVs (people, abilities, education, experience, skills) |
| **eMedCareers** | [promptcloud/latest-job-postings-in-europe](https://www.kaggle.com/datasets/promptcloud/latest-job-postings-in-europe) | 40k+ healthcare job postings (XML) |

```
data/raw/
├── 54k Resume dataset/
│   ├── 01_people.csv
│   ├── 02_abilities.csv
│   ├── 03_education.csv
│   ├── 04_experience.csv
│   ├── 05_person_skills.csv
│   └── 06_skills.csv
└── eMedCareers/
    └── emedcareers_...xml
```

Then run the data pipeline:

```bash
python scripts/prepare_data.py --max-resumes 500 --max-jobs 200
```

For a UK+UK weekend demo subset:

```bash
python scripts/build_demo_uk_dataset.py --max-jobs 200 --max-resumes 500
```

This will:
- Join the 5 normalized resume CSVs by `person_id`
- Filter to healthcare/clinical roles using keyword matching
- Parse the eMedCareers XML for job postings
- Output cleaned `.txt` files to `data/processed/resumes/` and `data/processed/jobs/`

### 4. Start the server

```bash
uv run uvicorn app.main:app --reload
```

Open **http://localhost:8000** for the web UI. API docs at **http://localhost:8000/docs**.

---

## API Reference

### `GET /health`

Health check.

```json
{ "status": "ok", "version": "0.1.0" }
```

### `POST /resumes/ingest`

Embed and index resumes into the FAISS vector store.

**Request body** (optional):
```json
{ "resumes_dir": "/custom/path/to/resumes" }
```

**Response:**
```json
{ "message": "Successfully ingested 500 resumes", "count": 500 }
```

### `POST /match`

Match a job posting against indexed resumes.

**Request body:**
```json
{
  "job_text": "Title: ICU Registered Nurse\nLocation: Calgary, Alberta\nRequirements:\n- Active CARNA registration\n- Minimum 2 years ICU experience\n- BLS and ACLS certification required",
  "top_k": 10,
  "market": "UK"
}
```

**Response:**
```json
{
  "retrieval_results": [
    { "resume_id": "resume_00042", "score": 0.87, "text": "..." }
  ],
  "ranked_candidates": [
    {
      "resume_id": "resume_00042",
      "rank": 1,
      "status": "PASS",
      "skill_overlaps": ["ICU experience", "ACLS", "CARNA license"],
      "missing_criteria": [],
      "reasoning": "Candidate holds active CARNA registration, 6+ years ICU experience, and all required certifications."  // Provides audit-ready reporting for recruiter decisions
    },
    {
      "resume_id": "resume_00105",
      "rank": null,
      "status": "FAIL",
      "skill_overlaps": ["ACLS"],
      "missing_criteria": ["CARNA registration", "ICU experience"],
      "reasoning": "Candidate is a cardiologist in Ontario without CARNA registration or ICU nursing experience."
    }
  ]
}
```

---

## Project Structure

```
semantic-clinical-matching/
├── app/
│   ├── __init__.py
│   ├── config.py                  # Settings via pydantic-settings (SCM_ env prefix)
│   ├── main.py                    # FastAPI app, endpoints, static file serving
│   ├── models.py                  # Pydantic request/response schemas
│   ├── pipeline.py                # Two-stage orchestrator
│   ├── ingestion/
│   │   └── resume_loader.py       # Load .txt resumes → LlamaIndex Documents
│   ├── retriever/
│   │   └── faiss_retriever.py     # FAISS vector index (build, query, save, load)
│   └── reranker/
│       ├── llm_reranker.py        # LLM-based candidate evaluation
│       └── prompts.py             # Medical recruiter system prompt
├── static/
│   ├── index.html                 # Web UI — single-page app
│   ├── style.css                  # Dark healthcare theme (glassmorphism)
│   └── app.js                     # Frontend logic (API calls, rendering)
├── scripts/
│   └── prepare_data.py            # Dataset extraction and cleaning pipeline
│   └── build_demo_uk_dataset.py   # Build UK+UK weekend-demo subset + sample queries
├── tests/
│   ├── conftest.py                # Shared fixtures (sample resumes, jobs, LLM output)
│   ├── test_api.py                # API endpoint tests
│   ├── test_pipeline.py           # Integration tests
│   ├── test_reranker.py           # Reranker/prompt unit tests
│   └── test_retriever.py          # FAISS retriever unit tests
├── data/
│   ├── raw/                       # Extracted Kaggle datasets (gitignored)
│   ├── processed/                 # Cleaned .txt files (gitignored)
│   └── faiss_index/               # Persisted FAISS index (gitignored)
├── pyproject.toml
└── uv.lock
```

---

## Configuration

All settings are managed via environment variables with the `SCM_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCM_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `SCM_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `SCM_LLM_MODEL` | `llama3` | LLM model for reranking |
| `SCM_TOP_K` | `20` | Number of candidates for Stage 1 retrieval |
| `SCM_EMBEDDING_DIMENSION` | `768` | Embedding vector dimension |
| `SCM_RESUMES_DIR` | `data/processed/resumes` | Path to resume text files |
| `SCM_JOBS_DIR` | `data/processed/jobs` | Path to job posting text files |
| `SCM_FAISS_INDEX_PATH` | `data/faiss_index` | Path to persisted FAISS index |
| `SCM_LLM_REQUEST_TIMEOUT` | `120.0` | LLM request timeout (seconds) |

---

## Development

### Run tests

```bash
uv run pytest tests/ -v
```

All Ollama calls are mocked in tests — **no live Ollama server is required** for the test suite.

### Lint and format

```bash
uv run ruff check app/ tests/        # Check for issues
uv run ruff check app/ tests/ --fix  # Auto-fix
uv run ruff format app/ tests/       # Format code
```

### Test coverage

```bash
uv run pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## How the Reranker Prompt Works

The LLM receives a **strict medical recruiter** system prompt that enforces:

1. **Licensing** — Candidate must hold a valid license in the correct jurisdiction
2. **Required certifications** — All "required" certifications must be present
3. **Minimum experience** — Must meet the minimum years in the specified clinical area
4. **Role relevance** — Clinical specialty must align with the posted role

Each candidate gets a structured evaluation:
- `status`: **PASS** or **FAIL** on mandatory requirements
- `skill_overlaps`: matched qualifications
- `missing_criteria`: required items the candidate lacks
- `reasoning`: one-paragraph explanation
- `rank`: integer rank for PASS candidates (1 = best), `null` for FAIL

---

## Web UI

The built-in web interface is served at the root URL (`/`) and provides:

- **Dark-mode healthcare theme** — glassmorphism cards with teal/blue gradient accents
- **Split-panel layout** — job posting input (left) and results (right)
- **Stage tabs** — toggle between Stage 2 ranked candidates and Stage 1 retrieval scores
- **Candidate cards** — PASS/FAIL badges, skill overlap chips, missing criteria, expandable reasoning
- **Pipeline stats** — retrieved / passed / failed counts at a glance
- **Sample jobs** — pre-built ICU Nurse, Cardiologist, Pharmacist, and Physiotherapist postings
- **Keyboard shortcut** — `Ctrl+Enter` to submit

No build step required — plain HTML, CSS, and JS served directly by FastAPI.

---

## License

MIT
