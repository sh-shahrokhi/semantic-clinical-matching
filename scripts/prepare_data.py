"""Extract and clean real clinical datasets from pre-extracted raw data.

Expected directory layout under data/raw/:
  1. data/raw/54k Resume dataset/  — 6 normalized CSVs (people, abilities, etc.)
  2. data/raw/eMedCareers/         — XML file with 40k+ healthcare job postings

Usage:
  python scripts/prepare_data.py [--max-resumes 500] [--max-jobs 200]
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESUMES_OUT = PROCESSED_DIR / "resumes"
JOBS_OUT = PROCESSED_DIR / "jobs"

RESUME_RAW_DIR = RAW_DIR / "54k Resume dataset"
JOBS_RAW_DIR = RAW_DIR / "eMedCareers"

# Healthcare-related keywords used to filter resumes to clinical roles
CLINICAL_KEYWORDS = re.compile(
    r"(?i)\b("
    r"nurse|nursing|rn\b|lpn\b|bscn|msn\b|np\b|"
    r"physician|doctor|md\b|mbbs|do\b|"
    r"surgeon|surgery|"
    r"pharmacist|pharmacy|pharmd|"
    r"therapist|therapy|physiotherapy|respiratory|"
    r"radiolog|radiology|"
    r"dentist|dental|"
    r"midwife|midwifery|"
    r"paramedic|emt\b|"
    r"psycholog|psychiatr|"
    r"hospital|clinical|icu\b|nicu\b|"
    r"medical|healthcare|health care|"
    r"patient care|bedside|"
    r"cardiol|oncol|pediatr|geriatr|"
    r"acls|bls\b|cpr\b|"
    r"licensed practical|registered nurse"
    r")"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Strip HTML tags, unescape entities, and normalise whitespace."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"\r\n|\r", "\n", text)          # normalise line endings
    text = re.sub(r"[ \t]+", " ", text)            # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)         # max two consecutive newlines
    return text.strip()





def _read_csv_safe(path: Path) -> list[dict[str, str]]:
    """Read a CSV file with error handling for encoding issues."""
    # Increase CSV field size limit for large fields
    csv.field_size_limit(10_000_000)
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Resume processing  (join 5 normalised CSVs by person_id)
# ---------------------------------------------------------------------------


def _process_resumes(extract_dir: Path, max_count: int) -> int:
    """Join the 6 normalised resume CSVs and output clinical resumes as text.

    Tables:
      01_people.csv       — person_id, name, email, phone, linkedin
      02_abilities.csv    — person_id, ability
      03_education.csv    — person_id, institution, program, start_date, location
      04_experience.csv   — person_id, title, firm, start_date, end_date, location
      05_person_skills.csv — person_id, skill
    """
    RESUMES_OUT.mkdir(parents=True, exist_ok=True)

    # ── Load people ──────────────────────────────────────────────────────
    people_path = extract_dir / "01_people.csv"
    print(f"  ↳ Reading people from {people_path.name} …")
    people = {}
    for row in _read_csv_safe(people_path):
        pid = row.get("person_id", "").strip()
        if pid:
            people[pid] = row

    # ── Load experience (multi-valued) ───────────────────────────────────
    exp_path = extract_dir / "04_experience.csv"
    print(f"  ↳ Reading experience from {exp_path.name} …")
    experience: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv_safe(exp_path):
        pid = row.get("person_id", "").strip()
        if pid:
            experience[pid].append(row)

    # ── Load education (multi-valued) ────────────────────────────────────
    edu_path = extract_dir / "03_education.csv"
    print(f"  ↳ Reading education from {edu_path.name} …")
    education: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv_safe(edu_path):
        pid = row.get("person_id", "").strip()
        if pid:
            education[pid].append(row)

    # ── Load skills (multi-valued) ───────────────────────────────────────
    skills_path = extract_dir / "05_person_skills.csv"
    print(f"  ↳ Reading skills from {skills_path.name} …")
    skills: dict[str, list[str]] = defaultdict(list)
    for row in _read_csv_safe(skills_path):
        pid = row.get("person_id", "").strip()
        skill = row.get("skill", "").strip()
        if pid and skill:
            # Deduplicate while preserving order
            if skill not in skills[pid]:
                skills[pid].append(skill)

    # ── Load abilities (multi-valued) ────────────────────────────────────
    abilities_path = extract_dir / "02_abilities.csv"
    print(f"  ↳ Reading abilities from {abilities_path.name} …")
    abilities: dict[str, list[str]] = defaultdict(list)
    for row in _read_csv_safe(abilities_path):
        pid = row.get("person_id", "").strip()
        ability = row.get("ability", "").strip()
        if pid and ability:
            if ability not in abilities[pid]:
                abilities[pid].append(ability)

    # ── Assemble and filter ──────────────────────────────────────────────
    print(f"  ↳ Assembling resumes and filtering for clinical roles …")
    written = 0
    for pid, person in people.items():
        if written >= max_count:
            break

        parts: list[str] = []

        # Name
        name = person.get("name", "").strip()
        if name:
            parts.append(f"Name: {name}")

        # Experience
        if pid in experience:
            exp_lines: list[str] = []
            for e in experience[pid]:
                title = e.get("title", "").strip()
                firm = e.get("firm", "").strip()
                start = e.get("start_date", "").strip()
                end = e.get("end_date", "").strip()
                loc = e.get("location", "").strip()
                line = f"  - {title}"
                if firm:
                    line += f" at {firm}"
                dates = f"{start}–{end}" if start or end else ""
                if dates:
                    line += f" ({dates})"
                if loc:
                    line += f", {loc}"
                exp_lines.append(line)
            if exp_lines:
                parts.append("Experience:\n" + "\n".join(exp_lines))

        # Education
        if pid in education:
            edu_lines: list[str] = []
            for e in education[pid]:
                inst = e.get("institution", "").strip()
                prog = e.get("program", "").strip()
                loc = e.get("location", "").strip()
                line = f"  - {prog}" if prog else f"  - {inst}"
                if inst and prog:
                    line += f" — {inst}"
                if loc:
                    line += f", {loc}"
                edu_lines.append(line)
            if edu_lines:
                parts.append("Education:\n" + "\n".join(edu_lines))

        # Skills
        if pid in skills:
            # Limit to 30 skills to keep resumes concise
            sk = skills[pid][:30]
            parts.append("Skills: " + ", ".join(sk))

        # Abilities
        if pid in abilities:
            ab = abilities[pid][:20]
            parts.append("Abilities: " + ", ".join(ab))

        text = "\n\n".join(parts)
        if not text or len(text) < 80:
            continue

        # Filter to clinical/healthcare resumes only
        if not CLINICAL_KEYWORDS.search(text):
            continue

        resume_id = f"resume_{written + 1:05d}"
        out_path = RESUMES_OUT / f"{resume_id}.txt"
        out_path.write_text(text, encoding="utf-8")
        written += 1

    return written


# ---------------------------------------------------------------------------
# Job processing  (parse XML)
# ---------------------------------------------------------------------------


def _process_jobs(extract_dir: Path, max_count: int) -> int:
    """Parse the eMedCareers XML dataset into individual text files.

    XML structure:
      <root>
        <page>
          <pageurl>...</pageurl>
          <record>
            <job_title>...</job_title>
            <category>...</category>
            <company_name>...</company_name>
            <job_description>...</job_description>
            <job_type>...</job_type>
            <salary_offered>...</salary_offered>
            <location>...</location>
            ...
          </record>
        </page>
        ...
      </root>
    """
    JOBS_OUT.mkdir(parents=True, exist_ok=True)

    # Find the XML file
    xml_files = list(extract_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {extract_dir}")
    xml_file = xml_files[0]
    print(f"  ↳ Parsing jobs from {xml_file.name} ({xml_file.stat().st_size // 1_000_000}MB) …")

    written = 0

    # Use iterparse for memory efficiency on this 145MB file
    context = ET.iterparse(str(xml_file), events=("end",))
    for event, elem in context:
        if written >= max_count:
            break

        if elem.tag != "record":
            continue

        parts: list[str] = []

        # Extract fields
        for tag, label in [
            ("job_title", "Title"),
            ("location", "Location"),
            ("company_name", "Employer"),
            ("category", "Category"),
            ("job_type", "Type"),
            ("salary_offered", "Salary"),
        ]:
            child = elem.find(tag)
            if child is not None and child.text and child.text.strip():
                parts.append(f"{label}: {child.text.strip()}")

        # Description (longer field)
        desc_el = elem.find("job_description")
        if desc_el is not None and desc_el.text and desc_el.text.strip():
            parts.append(f"Description:\n{_clean_text(desc_el.text)}")

        text = "\n".join(parts)
        if not text or len(text) < 50:
            # Clear element to free memory
            elem.clear()
            continue

        job_id = f"job_{written + 1:05d}"
        out_path = JOBS_OUT / f"{job_id}.txt"
        out_path.write_text(text, encoding="utf-8")
        written += 1

        # Clear element to free memory
        elem.clear()

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Process pre-extracted Kaggle datasets")
    parser.add_argument(
        "--max-resumes",
        type=int,
        default=500,
        help="Max number of clinical resumes to output (default: 500)",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=200,
        help="Max number of job postings to output (default: 200)",
    )
    args = parser.parse_args()

    # --- Resumes ---
    print("\n📄 Resumes — 54k Resume Dataset (Structured)")
    if not RESUME_RAW_DIR.exists():
        print(f"  ❌ Directory not found: {RESUME_RAW_DIR}")
        print(f"     Extract '54k Resume dataset' into {RAW_DIR}")
    else:
        resume_count = _process_resumes(RESUME_RAW_DIR, args.max_resumes)
        print(f"  ✅ Wrote {resume_count} clinical resumes to {RESUMES_OUT}")

    # --- Jobs ---
    print("\n💼 Jobs — 40k+ Healthcare Job Postings (eMedCareers)")
    if not JOBS_RAW_DIR.exists():
        print(f"  ❌ Directory not found: {JOBS_RAW_DIR}")
        print(f"     Extract 'eMedCareers' into {RAW_DIR}")
    else:
        job_count = _process_jobs(JOBS_RAW_DIR, args.max_jobs)
        print(f"  ✅ Wrote {job_count} job postings to {JOBS_OUT}")

    print(f"\n🎉 Done! Check {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
