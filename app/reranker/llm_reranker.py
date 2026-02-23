"""Stage 2 — LLM-based reranking and reasoning for clinical credential matching."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from llama_index.llms.ollama import Ollama

from app.reranker.prompts import RECRUITER_SYSTEM_PROMPT, build_reranker_prompt

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


class LLMReranker:
    """Rerank and evaluate candidates using an LLM acting as a medical recruiter.

    Args:
        llm_model: Name of the Ollama LLM model.
        ollama_base_url: Ollama server URL.
        request_timeout: Timeout for LLM requests in seconds.
    """

    def __init__(
        self,
        llm_model: str = "llama3",
        ollama_base_url: str = "http://localhost:11434",
        request_timeout: float = 120.0,
    ) -> None:
        self.llm = Ollama(
            model=llm_model,
            base_url=ollama_base_url,
            request_timeout=request_timeout,
        )

    def rerank(
        self,
        job_text: str,
        candidates: list[dict[str, str]],
    ) -> list[RankedCandidate]:
        """Evaluate and rank candidates against a job posting.

        Args:
            job_text: Full job posting text.
            candidates: List of dicts with ``resume_id`` and ``text`` keys.

        Returns:
            List of RankedCandidate objects, PASS first (ranked), then FAIL.
        """
        user_prompt = build_reranker_prompt(job_text, candidates)

        response = self.llm.complete(
            prompt=user_prompt,
            system_prompt=RECRUITER_SYSTEM_PROMPT,
        )

        return self._parse_response(response.text)

    @staticmethod
    def _parse_response(response_text: str) -> list[RankedCandidate]:
        """Parse the LLM JSON response into RankedCandidate objects.

        Handles common LLM quirks like markdown fences around JSON.

        Args:
            response_text: Raw LLM output text.

        Returns:
            List of RankedCandidate objects.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        text = response_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)

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
