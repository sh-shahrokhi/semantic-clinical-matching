"""Application settings managed via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration for the semantic-clinical-matching pipeline."""

    # Ollama connection
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3"

    # Retriever settings
    top_k: int = 20
    embedding_dimension: int = 768

    # Paths
    faiss_index_path: str = "data/faiss_index"
    resumes_dir: str = "data/processed/resumes"
    jobs_dir: str = "data/processed/jobs"

    # LLM settings
    llm_request_timeout: float = 120.0
    llm_max_concurrency: int = 3

    model_config = {"env_prefix": "SCM_"}

    @property
    def resumes_path(self) -> Path:
        return Path(self.resumes_dir)

    @property
    def jobs_path(self) -> Path:
        return Path(self.jobs_dir)

    @property
    def faiss_path(self) -> Path:
        return Path(self.faiss_index_path)


def get_settings() -> Settings:
    """Return a Settings instance (can be overridden in tests via FastAPI dependency injection)."""
    return Settings()
