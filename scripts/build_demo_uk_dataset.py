#!/usr/bin/env python3
"""Build a UK+UK demo subset from processed text files.

Selects UK-focused healthcare jobs and resumes from `data/processed/*` and
writes a weekend-demo corpus to `data/demo_uk`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

UK_TOKENS = (
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
)

HEALTH_TOKENS = (
    " healthcare ",
    " medical ",
    " clinical ",
    " hospital ",
    " patient ",
    " pharma ",
    " pharmaceutical ",
    " nurse ",
)

ROLE_TOKENS = (
    " sales ",
    " account manager ",
    " business development ",
    " territory ",
    " administrator ",
    " administrative ",
    " coordinator ",
    " support ",
    " operations ",
    " registered nurse ",
    " icu ",
)


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    low = f" {text.lower()} "
    return any(tok in low for tok in tokens)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _job_is_demo_uk(text: str) -> bool:
    return (
        _contains_any(text, UK_TOKENS)
        and _contains_any(text, HEALTH_TOKENS)
        and _contains_any(text, ROLE_TOKENS)
    )


def _resume_is_demo_uk(text: str) -> bool:
    return (
        _contains_any(text, UK_TOKENS)
        and (_contains_any(text, HEALTH_TOKENS) or _contains_any(text, ROLE_TOKENS))
    )


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("title:"):
            return stripped.split(":", 1)[1].strip()
    return "Untitled role"


def _extract_location(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("location:"):
            return stripped.split(":", 1)[1].strip()
    return "UK"


def _copy_subset(
    source_dir: Path,
    out_dir: Path,
    max_items: int,
    predicate,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []

    for path in sorted(source_dir.glob("*.txt")):
        text = _read_text(path)
        if not text.strip():
            continue
        if not predicate(text):
            continue
        shutil.copy2(path, out_dir / path.name)
        selected.append(path)
        if len(selected) >= max_items:
            break

    return selected


def _build_queries(job_files: list[Path], out_file: Path, max_queries: int) -> None:
    rows: list[dict[str, str]] = []
    for job_path in job_files[:max_queries]:
        text = _read_text(job_path)
        rows.append(
            {
                "job_file": job_path.name,
                "title": _extract_title(text),
                "location": _extract_location(text),
                "query": text[:1200].strip(),
                "expected_match_signals": (
                    "UK location alignment, healthcare role relevance, "
                    "and required-experience overlap."
                ),
            }
        )
    out_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", type=Path, default=Path("data/processed/jobs"))
    parser.add_argument(
        "--resumes-dir", type=Path, default=Path("data/processed/resumes"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/demo_uk"))
    parser.add_argument("--max-jobs", type=int, default=200)
    parser.add_argument("--max-resumes", type=int, default=500)
    parser.add_argument("--max-queries", type=int, default=10)
    args = parser.parse_args()

    if not args.jobs_dir.is_dir():
        raise FileNotFoundError(f"Jobs directory not found: {args.jobs_dir}")
    if not args.resumes_dir.is_dir():
        raise FileNotFoundError(f"Resumes directory not found: {args.resumes_dir}")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs_out = args.out_dir / "jobs"
    resumes_out = args.out_dir / "resumes"

    selected_jobs = _copy_subset(
        source_dir=args.jobs_dir,
        out_dir=jobs_out,
        max_items=args.max_jobs,
        predicate=_job_is_demo_uk,
    )
    selected_resumes = _copy_subset(
        source_dir=args.resumes_dir,
        out_dir=resumes_out,
        max_items=args.max_resumes,
        predicate=_resume_is_demo_uk,
    )
    _build_queries(
        selected_jobs, args.out_dir / "demo_queries_uk.json", args.max_queries,
    )

    print("UK demo dataset ready")
    print(f"- Jobs selected: {len(selected_jobs)} / {len(list(args.jobs_dir.glob('*.txt')))}")
    print(
        f"- Resumes selected: {len(selected_resumes)} / "
        f"{len(list(args.resumes_dir.glob('*.txt')))}"
    )
    print(f"- Output: {args.out_dir}")


if __name__ == "__main__":
    main()
