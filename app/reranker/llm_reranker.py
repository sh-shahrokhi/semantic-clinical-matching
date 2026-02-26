"""Stage 2 — LLM-based reranking and reasoning for clinical credential matching."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from llama_index.llms.ollama import Ollama

from app.reranker.prompts import (
    JSON_REPAIR_SYSTEM_PROMPT,
    RANKING_SYSTEM_PROMPT,
    RANKING_SYSTEM_PROMPT_CLINICAL,
    RANKING_SYSTEM_PROMPT_NONCLINICAL,
    SINGLE_CANDIDATE_SYSTEM_PROMPT_CLINICAL,
    SINGLE_CANDIDATE_SYSTEM_PROMPT_NONCLINICAL,
    build_ranking_prompt,
    build_single_candidate_prompt,
    build_single_json_repair_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    """A candidate after LLM evaluation."""

    resume_id: str
    rank: int | None
    status: str  # "PASS" or "FAIL"
    skill_overlaps: list[str]
    missing_criteria: list[str]
    reasoning: str


class LLMResponseError(ValueError):
    """Raised when the model output is empty or cannot be parsed as valid JSON."""


class LLMReranker:
    """Rerank and evaluate candidates using an LLM acting as a medical recruiter.

    Evaluates each candidate individually (with async concurrency), then ranks
    the PASSing candidates in a lightweight follow-up call.

    Args:
        llm_model: Name of the Ollama LLM model.
        ollama_base_url: Ollama server URL.
        request_timeout: Timeout for LLM requests in seconds.
        max_concurrency: Maximum number of concurrent LLM evaluation calls.
    """

    def __init__(
        self,
        llm_model: str = "llama3",
        ollama_base_url: str = "http://localhost:11434",
        request_timeout: float = 120.0,
        max_concurrency: int = 3,
    ) -> None:
        self.default_model = llm_model
        self.ollama_base_url = ollama_base_url
        self.request_timeout = request_timeout
        self.max_concurrency = max_concurrency
        self.llm = self._build_client(llm_model)

    def _build_client(self, model_name: str) -> Ollama:
        """Create an Ollama client for a specific model."""
        return Ollama(
            model=model_name,
            base_url=self.ollama_base_url,
            request_timeout=self.request_timeout,
            temperature=0.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _compose_prompt(system_prompt: str, user_prompt: str) -> str:
        """Inline system guidance into the prompt for completion-style calls."""
        return f"{system_prompt.strip()}\n\n{user_prompt.strip()}"

    @staticmethod
    def _is_nonclinical_role(job_text: str) -> bool:
        """Heuristic role detector for prompt routing."""
        text = job_text.lower()

        commercial_signals = (
            "medical sales",
            "sales representative",
            "territory manager",
            "account manager",
            "business development",
            "commercial role",
            "pharmaceutical sales",
        )
        nonclinical_signals = (
            "administrator",
            "administrative",
            "coordinator",
            "operations",
            "support assistant",
            "intake specialist",
            "customer service",
            "office manager",
            "program specialist",
        )
        clinical_care_signals = (
            "registered nurse",
            "rn",
            "licensed practical nurse",
            "physician",
            "surgeon",
            "midwife",
            "icu",
            "emergency room",
            "bedside",
            "patient care",
            "clinical specialty",
        )

        has_commercial = any(token in text for token in commercial_signals)
        has_nonclinical = any(token in text for token in nonclinical_signals)
        has_clinical_care = any(token in text for token in clinical_care_signals)

        if has_commercial:
            return True
        if has_nonclinical and not has_clinical_care:
            return True
        return False

    @classmethod
    def _select_single_candidate_system_prompt(cls, job_text: str) -> str:
        """Choose per-candidate evaluation policy by job family."""
        if cls._is_nonclinical_role(job_text):
            return SINGLE_CANDIDATE_SYSTEM_PROMPT_NONCLINICAL
        return SINGLE_CANDIDATE_SYSTEM_PROMPT_CLINICAL

    @classmethod
    def _select_ranking_system_prompt(cls, job_text: str) -> str:
        """Choose ranking policy by job family."""
        if cls._is_nonclinical_role(job_text):
            return RANKING_SYSTEM_PROMPT_NONCLINICAL
        return RANKING_SYSTEM_PROMPT_CLINICAL

    async def rerank(
        self,
        job_text: str,
        candidates: list[dict[str, str]],
        llm_model: str | None = None,
    ) -> list[RankedCandidate]:
        """Evaluate and rank candidates against a job posting.

        Each candidate is evaluated individually with capped concurrency,
        then PASSing candidates are ranked in a follow-up call.

        Args:
            job_text: Full job posting text.
            candidates: List of dicts with ``resume_id`` and ``text`` keys.
            llm_model: Optional model override for this request.

        Returns:
            List of RankedCandidate objects, PASS first (ranked), then FAIL.
        """
        model_name = llm_model or self.default_model
        llm_client = (
            self.llm
            if model_name == self.default_model
            else self._build_client(model_name)
        )
        candidate_system_prompt = self._select_single_candidate_system_prompt(job_text)
        ranking_system_prompt = self._select_ranking_system_prompt(job_text)

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _eval_with_limit(candidate: dict[str, str]) -> RankedCandidate:
            async with sem:
                return await asyncio.to_thread(
                    self._evaluate_one,
                    job_text,
                    candidate,
                    llm_client,
                    candidate_system_prompt,
                )

        raw_results = await asyncio.gather(
            *[_eval_with_limit(c) for c in candidates],
            return_exceptions=True,
        )

        passing: list[RankedCandidate] = []
        failing: list[RankedCandidate] = []

        for i, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                logger.error(
                    "Candidate %s evaluation failed: %s",
                    candidates[i].get("resume_id", f"index-{i}"),
                    result,
                )
                failing.append(
                    RankedCandidate(
                        resume_id=candidates[i].get("resume_id", f"unknown-{i}"),
                        rank=None,
                        status="FAIL",
                        skill_overlaps=[],
                        missing_criteria=["evaluation_error"],
                        reasoning=f"LLM evaluation failed: {result}",
                    )
                )
            elif result.status == "PASS":
                passing.append(result)
            else:
                failing.append(result)

        # Rank passing candidates
        if len(passing) > 1:
            passing = self._rank_passing(
                job_text, passing, llm_client, ranking_system_prompt,
            )
        elif len(passing) == 1:
            passing[0].rank = 1

        return passing + failing

    # ------------------------------------------------------------------
    # Per-candidate evaluation
    # ------------------------------------------------------------------

    def _evaluate_one(
        self,
        job_text: str,
        candidate: dict[str, str],
        llm_client: Ollama,
        candidate_system_prompt: str,
    ) -> RankedCandidate:
        """Evaluate a single candidate against the job posting.

        Args:
            job_text: Full job posting text.
            candidate: Dict with ``resume_id`` and ``text`` keys.
            llm_client: Ollama LLM client to use.

        Returns:
            A RankedCandidate with rank=None (ranking assigned later).
        """
        candidate_id = candidate["resume_id"]
        prompt = build_single_candidate_prompt(
            job_text, candidate_id, candidate["text"],
        )

        max_attempts = 2
        last_error = None

        for attempt in range(max_attempts):
            try:
                request_prompt = self._compose_prompt(
                    candidate_system_prompt, prompt,
                )
                response = llm_client.complete(prompt=request_prompt)

                raw_text = response.text or ""
                if not raw_text.strip():
                    raise ValueError("LLM returned an empty response.")

                logger.debug(
                    "Raw LLM response for %s (len=%d):\n%s",
                    candidate_id, len(raw_text), raw_text[:500],
                )

                try:
                    result = self._parse_single_response(raw_text, candidate_id)
                except ValueError as first_error:
                    logger.warning(
                        "JSON parsing failed for %s; retrying with repair: %s",
                        candidate_id, first_error,
                    )
                    repair_prompt = build_single_json_repair_prompt(raw_text)
                    repair_request_prompt = self._compose_prompt(
                        JSON_REPAIR_SYSTEM_PROMPT, repair_prompt,
                    )
                    repair_response = llm_client.complete(prompt=repair_request_prompt)
                    repaired_text = repair_response.text or ""
                    if not repaired_text.strip():
                        raise ValueError("LLM returned empty output after JSON repair.")

                    result = self._parse_single_response(repaired_text, candidate_id)

                # Validate for empty skeleton hallucination
                if (
                    result.status == "FAIL"
                    and not result.missing_criteria
                    and not result.reasoning
                ):
                    raise ValueError(
                        "LLM returned an empty skeleton with no reasoning "
                        "or missing criteria"
                    )

                return result

            except ValueError as e:
                last_error = e
                logger.warning("Attempt %d for %s failed: %s", attempt + 1, candidate_id, e)

        raise LLMResponseError(
            f"Failed to get valid evaluation for {candidate_id} "
            f"after {max_attempts} attempts. Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Ranking pass
    # ------------------------------------------------------------------

    def _rank_passing(
        self,
        job_text: str,
        passing: list[RankedCandidate],
        llm_client: Ollama,
        ranking_system_prompt: str = RANKING_SYSTEM_PROMPT,
    ) -> list[RankedCandidate]:
        """Assign ranks to PASSing candidates via a lightweight LLM call.

        Falls back to original order on failure.
        """
        summaries = [
            {
                "resume_id": c.resume_id,
                "skill_overlaps": c.skill_overlaps,
                "reasoning": c.reasoning,
            }
            for c in passing
        ]
        prompt = build_ranking_prompt(job_text, summaries)

        try:
            ranking_prompt = self._compose_prompt(
                ranking_system_prompt, prompt,
            )
            response = llm_client.complete(prompt=ranking_prompt)
            raw_text = response.text or ""
            ranking = self._parse_ranking_response(raw_text)

            rank_map = {item["resume_id"]: item["rank"] for item in ranking}
            for candidate in passing:
                candidate.rank = rank_map.get(candidate.resume_id)

            # Sort by rank (None last)
            passing.sort(key=lambda c: c.rank if c.rank is not None else 999)
        except Exception as e:
            logger.warning("Ranking call failed, using insertion order: %s", e)
            for i, candidate in enumerate(passing, start=1):
                candidate.rank = i

        return passing

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences and <think> tags if present."""
        import re

        # Remove <think>...</think> anywhere in the text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)

        return text.strip()

    @staticmethod
    def _parse_single_response(
        response_text: str, candidate_id: str,
    ) -> RankedCandidate:
        """Parse an LLM response containing a single JSON object.

        Args:
            response_text: Raw LLM output text.
            candidate_id: Expected candidate ID (fallback).

        Returns:
            A RankedCandidate with rank=None.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        text = LLMReranker._strip_fences(response_text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            logger.debug("Raw response: %s", response_text)
            raise ValueError(f"LLM response is not valid JSON: {e}") from e

        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError("LLM returned an empty JSON array")
            data = data[0]

        if not isinstance(data, dict):
            raise ValueError("LLM response must be a JSON object")

        # Check for schema hallucinations
        if "status" not in data and "reasoning" not in data:
            raise ValueError("LLM returned valid JSON but with hallucinated schema keys.")

        raw_status = data.get("status", "FAIL")
        status = str(raw_status).upper().strip() if raw_status is not None else "FAIL"

        def _normalize_str_list(value: object) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                text = value.strip()
                return [text] if text else []
            if isinstance(value, list | tuple | set):
                normalized = [str(v).strip() for v in value]
                return [v for v in normalized if v]
            text = str(value).strip()
            return [text] if text else []

        skill_overlaps = _normalize_str_list(data.get("skill_overlaps", []))
        missing_criteria = _normalize_str_list(data.get("missing_criteria", []))
        reasoning_value = data.get("reasoning", "")
        reasoning = (
            reasoning_value.strip()
            if isinstance(reasoning_value, str)
            else str(reasoning_value).strip()
        )

        # Strict PASS/FAIL enforcement
        if status not in ("PASS", "FAIL"):
            logger.warning(
                "Candidate %s: LLM returned invalid status '%s' — defaulting to FAIL",
                candidate_id, status,
            )
            status = "FAIL"

        # Consistency check: override contradictory status
        if status == "PASS" and missing_criteria:
            logger.warning(
                "Candidate %s: LLM said PASS but has missing_criteria — overriding to FAIL",
                candidate_id,
            )
            status = "FAIL"

        return RankedCandidate(
            resume_id=candidate_id,
            rank=None,
            status=status,
            skill_overlaps=skill_overlaps,
            missing_criteria=missing_criteria,
            reasoning=reasoning,
        )

    @staticmethod
    def _parse_ranking_response(response_text: str) -> list[dict]:
        """Parse the ranking LLM response into a list of {resume_id, rank} dicts."""
        text = LLMReranker._strip_fences(response_text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ranking response is not valid JSON: {e}") from e

        if not isinstance(data, list):
            raise ValueError("Ranking response must be a JSON array")

        return data

    @staticmethod
    def _parse_response(response_text: str) -> list[RankedCandidate]:
        """Parse the LLM JSON response into RankedCandidate objects.

        Kept for backward compatibility with tests.

        Args:
            response_text: Raw LLM output text.

        Returns:
            List of RankedCandidate objects.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        text = LLMReranker._strip_fences(response_text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            logger.debug("Raw response: %s", response_text)
            raise ValueError(f"LLM response is not valid JSON: {e}") from e

        if not isinstance(data, list):
            raise ValueError("LLM response must be a JSON array")

        results: list[RankedCandidate] = []
        for item in data:
            results.append(
                RankedCandidate(
                    resume_id=item.get("resume_id", "unknown"),
                    rank=item.get("rank"),
                    status=item.get("status", "FAIL"),
                    skill_overlaps=item.get("skill_overlaps", []),
                    missing_criteria=item.get("missing_criteria", []),
                    reasoning=item.get("reasoning", ""),
                )
            )

        return results
