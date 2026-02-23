"""Resume loading and chunking for the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import Document


def load_resumes(resumes_dir: str | Path) -> list[Document]:
    """Load resume text files from a directory and return LlamaIndex Documents.

    Each resume file becomes one Document. The filename (without extension)
    is stored as ``resume_id`` in the document metadata.

    Args:
        resumes_dir: Path to directory containing ``.txt`` resume files.

    Returns:
        List of LlamaIndex Document objects ready for embedding.
    """
    resumes_path = Path(resumes_dir)
    if not resumes_path.is_dir():
        raise FileNotFoundError(f"Resumes directory not found: {resumes_path}")

    documents: list[Document] = []
    for txt_file in sorted(resumes_path.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        doc = Document(
            text=text,
            metadata={
                "resume_id": txt_file.stem,
                "source_file": str(txt_file),
            },
        )
        documents.append(doc)

    return documents
