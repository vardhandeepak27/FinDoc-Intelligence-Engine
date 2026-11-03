"""Central configuration for FinDoc Intelligence Engine."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="FINDOC_",
        extra="ignore",
    )

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "indexes"

    # --- Chunking ---
    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, gt=0)

    # --- Retrieval ---
    top_k: int = Field(default=10, gt=0)
    candidate_k: int = Field(default=50, gt=0)
    rrf_k: int = Field(default=60, gt=0)

    # --- LLM ---
    llm_provider: Literal["groq", "gemini", "ollama"] = "groq"
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("data_dir", "index_dir")
    @classmethod
    def resolve_relative_to_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def check_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        return self

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.csv"

    @property
    def documents_path(self) -> Path:
        return self.processed_dir / "documents.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def diagnostics_path(self) -> Path:
        return self.processed_dir / "page_diagnostics.csv"

    @property
    def index_path(self) -> Path:
        return self.index_dir / "financial.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "chunk_metadata.jsonl"

    @property
    def index_info_path(self) -> Path:
        return self.index_dir / "index_info.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


if __name__ == "__main__":
    for key, value in settings.model_dump().items():
        print(f"{key:22} = {value}")
    print(f"{'chunks_path':22} = {settings.chunks_path}")
