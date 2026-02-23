"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import pytest

from app.config import Settings  # noqa: I001

SAMPLE_RESUME_TEXT = """\
Name: Sarah Chen, RN, BScN
Location: Calgary, Alberta, Canada
License: Active CARNA registration (Alberta)
Experience:
- ICU Nurse, Foothills Medical Centre, Calgary (2018–present)
  - Managed ventilator-dependent patients in a 30-bed medical-surgical ICU.
  - Performed hemodynamic monitoring, arterial line management, and titration of vasoactive drips.
Certifications: ACLS, BLS, CNCCP(C) — Canadian Nursing Critical Care Certification
Education: BScN, University of Calgary (2015)
Skills: Critical care, ventilator management, hemodynamic monitoring, patient assessment
"""

SAMPLE_RESUME_TEXT_2 = """\
Name: Dr. James Okafor, MD, FRCPC
Location: Toronto, Ontario, Canada
License: Active CPSO registration (Ontario)
Experience:
- Staff Cardiologist, Toronto General Hospital (2016–present)
  - Performed diagnostic and interventional cardiac catheterizations.
Certifications: FRCPC (Internal Medicine), FRCPC (Cardiology), ACLS Instructor
Education: MD, University of Lagos (2007)
Skills: Interventional cardiology, cardiac catheterization, echocardiography
"""

SAMPLE_JOB_TEXT = """\
Title: ICU Registered Nurse
Location: Calgary, Alberta, Canada
Employer: Alberta Health Services
Requirements:
- Active registration with CARNA
- Minimum 2 years ICU/critical care experience
- BLS and ACLS certification required
- CNCCP(C) certification preferred
- Experience with hemodynamic monitoring and ventilator management
"""

SAMPLE_LLM_RESPONSE = """[
  {
    "resume_id": "resume_001",
    "rank": 1,
    "status": "PASS",
    "skill_overlaps": [
      "ICU experience", "ACLS", "BLS",
      "hemodynamic monitoring",
      "ventilator management", "CARNA license"
    ],
    "missing_criteria": [],
    "reasoning": "Sarah Chen meets all requirements."
  },
  {
    "resume_id": "resume_002",
    "rank": null,
    "status": "FAIL",
    "skill_overlaps": ["ACLS"],
    "missing_criteria": [
      "CARNA registration",
      "ICU experience",
      "BLS certification"
    ],
    "reasoning": "Dr. Okafor is a cardiologist in Ontario."
  }
]"""


@pytest.fixture
def sample_resume_text() -> str:
    return SAMPLE_RESUME_TEXT


@pytest.fixture
def sample_resume_text_2() -> str:
    return SAMPLE_RESUME_TEXT_2


@pytest.fixture
def sample_job_text() -> str:
    return SAMPLE_JOB_TEXT


@pytest.fixture
def sample_llm_response() -> str:
    return SAMPLE_LLM_RESPONSE


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    """Settings configured for testing with temp directories."""
    return Settings(
        ollama_base_url="http://localhost:11434",
        embedding_model="nomic-embed-text",
        llm_model="llama3",
        top_k=5,
        resumes_dir=str(tmp_path / "resumes"),
        jobs_dir=str(tmp_path / "jobs"),
        faiss_index_path=str(tmp_path / "faiss_index"),
    )
