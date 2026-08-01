"""Configuration models with safe, local-only defaults."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from local_semantic_engine.core.errors import ConfigurationError


class OllamaSettings(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    generation_model: str = ""
    embedding_model: str = "embeddinggemma"
    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    read_timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=3)
    keep_alive: str = "10m"
    context_tokens: int = Field(default=8192, ge=512)

    @field_validator("base_url")
    @classmethod
    def require_http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("Ollama base URL must use http:// or https://.")
        return value.rstrip("/")


class StorageSettings(BaseModel):
    database_path: Path = Path("data/local_semantic_engine.sqlite3")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    index_data_dir: Path = Path("data/indexes")


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    inference_concurrency: int = Field(default=1, ge=1, le=4)

    @field_validator("host")
    @classmethod
    def require_loopback_host(cls, value: str) -> str:
        allowed = {"127.0.0.1", "localhost", "::1"}
        if value not in allowed:
            raise ValueError("Version 1 only permits loopback API hosts.")
        return value


class RetrievalSettings(BaseModel):
    broad_candidate_count: int = Field(default=100, ge=10, le=1000)
    negative_weight: float = Field(default=0.35, ge=0.0, le=2.0)
    quality_weight: float = Field(default=0.05, ge=0.0, le=1.0)


class ValidationSettings(BaseModel):
    max_correction_attempts: int = Field(default=2, ge=0, le=5)
    strict_missing_data: bool = True


class ExecutionProfile(BaseModel):
    rerank_candidate_count: int = Field(ge=1, le=50)
    candidate_summary_characters: int = Field(ge=80, le=2000)
    max_output_tokens: int = Field(ge=64, le=4096)
    temperature: float = Field(ge=0.0, le=2.0)
    thinking: bool = False
    context_tokens: int = Field(ge=512, le=65536)
    max_correction_attempts: int = Field(ge=0, le=5)


def _default_profiles() -> dict[str, ExecutionProfile]:
    return {
        "fast": ExecutionProfile(
            rerank_candidate_count=12,
            candidate_summary_characters=220,
            max_output_tokens=400,
            temperature=0.0,
            thinking=False,
            context_tokens=8192,
            max_correction_attempts=1,
        ),
        "balanced": ExecutionProfile(
            rerank_candidate_count=20,
            candidate_summary_characters=400,
            max_output_tokens=700,
            temperature=0.0,
            thinking=False,
            context_tokens=8192,
            max_correction_attempts=2,
        ),
        "quality": ExecutionProfile(
            rerank_candidate_count=30,
            candidate_summary_characters=650,
            max_output_tokens=1000,
            temperature=0.1,
            thinking=False,
            context_tokens=16384,
            max_correction_attempts=2,
        ),
    }


class AppSettings(BaseModel):
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    profiles: dict[str, ExecutionProfile] = Field(default_factory=_default_profiles)

    @model_validator(mode="after")
    def require_standard_profiles(self) -> AppSettings:
        missing = {"fast", "balanced", "quality"}.difference(self.profiles)
        if missing:
            raise ValueError(f"Missing required execution profiles: {', '.join(sorted(missing))}.")
        return self

    def profile(self, name: str) -> ExecutionProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown execution profile: {name}.") from exc
