"""Application configuration — loaded once at startup from environment."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ─────────────────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "production"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    secret_key: str = Field(default="dev-secret-change-in-prod")

    # ── API ─────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:8501"]

    # ── Streamlit ────────────────────────────────────────────────────────────
    api_base_url: str = "http://localhost:8000"

    # ── LLM Provider Keys ───────────────────────────────────────────────────
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None
    together_api_key: str | None = None

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_max_rows: int = 500
    eval_sample_seed: int = 42
    eval_timeout_seconds: int = 60
    eval_concurrency: int = 3

    # ── Upload ──────────────────────────────────────────────────────────────
    max_upload_mb: int = 50
    upload_dir: str = "/tmp/costguard"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def available_providers(self) -> list[str]:
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.google_api_key:
            providers.append("google")
        if self.groq_api_key:
            providers.append("groq")
        if self.together_api_key:
            providers.append("together")
        return providers

    def ensure_upload_dir(self) -> None:
        os.makedirs(self.upload_dir, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    settings = Settings()
    settings.ensure_upload_dir()
    return settings
