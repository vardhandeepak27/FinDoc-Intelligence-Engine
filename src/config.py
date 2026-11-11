"""Central configuration for FinDoc Intelligence Engine."""
from functools import lru_cache
import re
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

    # --- Extraction ---
    use_ocr_fallback: bool = True          # OCR pages whose text layer is broken
    tessdata_dir: Path | None = None       # Tesseract language data folder
    broken_text_threshold: float = Field(default=0.08, gt=0, lt=1)

    # --- Chunking ---
    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    min_chunk_chars: int = Field(default=200, ge=0)
    boilerplate_header_ratio: float = Field(default=0.3, gt=0, le=1)

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, gt=0)
    # Asymmetric models want different prefixes for queries and documents.
    # Wrong or missing prefixes don't error - they silently lower quality.
    embedding_query_prefix: str = ""
    embedding_doc_prefix: str = ""
    # Cap on tokens per chunk (None = the model's own limit). Long-context
    # models get slow on CPU, so we cap them just above our longest chunks.
    embedding_max_tokens: int | None = Field(default=None, gt=0)

    # --- Retrieval ---
    top_k: int = Field(default=10, gt=0)
    candidate_k: int = Field(default=50, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    # Weights for Reciprocal Rank Fusion (1.0 / 1.0 = classic RRF)
    bm25_weight: float = Field(default=1.0, ge=0)
    dense_weight: float = Field(default=1.0, ge=0)

    # --- Reranking (cross-encoder second stage). None = no reranking.
    reranker_model: str | None = None
    rerank_candidates: int = Field(default=30, gt=0)   # how many to rerank
    rerank_max_tokens: int = Field(default=512, gt=0)

    # --- Query understanding
    # A company-level figure with no qualifier conventionally means the
    # consolidated (group) figure. Measured: it HURT (hit@10 0.87 -> 0.84),
    # because dividends and share capital are parent-company (standalone)
    # facts. Off until we have a smarter rule; "consolidated" turns it on.
    default_basis: Literal["consolidated", "standalone", "none"] = "none"

    # --- LLM ---
    llm_provider: Literal["groq", "gemini", "ollama"] = "groq"
    llm_model: str | None = None            # None = the provider's default
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_max_tokens: int = Field(default=2048, gt=0)   # reasoning + answer
    llm_reasoning_effort: Literal["low", "medium", "high"] = "low"
    llm_timeout_seconds: float = Field(default=60, gt=0)
    llm_max_retries: int = Field(default=5, ge=0)
    # Stay under the provider's tokens-per-minute limit (Groq free tier for
    # gpt-oss-120b: 8,000). None = no client-side limiting.
    llm_tokens_per_minute: int | None = Field(default=7000, gt=0)
    # Per-model overrides (Groq's limits differ by model). As JSON in .env:
    # FINDOC_MODEL_TOKENS_PER_MINUTE={"qwen/qwen3.8-27b": 5000}
    model_tokens_per_minute: dict[str, int] = {"qwen/qwen3.8-27b": 5000}
    # Judge model for answer evaluation: a DIFFERENT model family from the
    # answering model, to reduce self-preference bias.
    # (Preview model on Groq: fine for offline evaluation, not for serving.)
    judge_model: str = "qwen/qwen3.8-27b"

    # --- Answer generation
    # Prompt versions are kept side by side so experiments are comparable:
    # v1 = baseline, v2 = exact units + report all valid figures.
    prompt_version: Literal["v1", "v2"] = "v2"
    answer_context_chunks: int = Field(default=6, gt=0)   # chunks sent to the LLM
    answer_context_tokens: int = Field(default=4000, gt=0)  # budget for those chunks

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
    def model_index_dir(self) -> Path:
        """One index folder per embedding model, e.g. indexes/bge-small-en-v1.5.

        Vectors from different models can't be mixed, so each model gets
        its own index; switching models never overwrites another's index.
        """
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", self.embedding_model.split("/")[-1])
        return self.index_dir / slug

    @property
    def index_path(self) -> Path:
        return self.model_index_dir / "financial.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.model_index_dir / "chunk_metadata.jsonl"

    @property
    def index_info_path(self) -> Path:
        return self.model_index_dir / "index_info.json"

    @property
    def embedding_cache_path(self) -> Path:
        return self.model_index_dir / "embedding_cache.npz"

    @property
    def eval_path(self) -> Path:
        return self.data_dir / "eval" / "retrieval_eval.jsonl"

    @property
    def eval_results_dir(self) -> Path:
        return self.data_dir / "eval" / "results"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


if __name__ == "__main__":
    for key, value in settings.model_dump().items():
        print(f"{key:22} = {value}")
    print(f"{'chunks_path':22} = {settings.chunks_path}")
