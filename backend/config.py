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

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        if v == "dev-secret-change-in-prod":
            import os
            if os.getenv("APP_ENV", "production") == "production":
                import warnings
                warnings.warn(
                    "SECRET_KEY is set to the default development value. "
                    "Generate a real key: openssl rand -hex 32",
                    stacklevel=2,
                )
        return v

    # ── API ─────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:8501"]

    # ── Streamlit ────────────────────────────────────────────────────────────
    api_base_url: str = "http://localhost:8000"

    # ── LLM Provider Keys (mirrors RealDataAgentBench provider support) ─────
    # These are server-side keys (set via env / .env file).
    # Per-request session keys from the UI are handled separately in SessionKeys.
    anthropic_api_key: str | None = None   # Claude Sonnet 4.6, Opus 4.6, Haiku 4.5
    openai_api_key: str | None = None      # GPT-5, GPT-4.1, GPT-4o, GPT-4o-mini
    groq_api_key: str | None = None        # Llama 3.3-70B, Mixtral 8x7B
    xai_api_key: str | None = None         # Grok-3, Grok-3-mini, Grok-3-fast
    gemini_api_key: str | None = None      # Gemini 2.5 Pro/Flash

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
        """Return provider names that have server-side API keys configured."""
        providers = []
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.openai_api_key:
            providers.append("openai")
        if self.groq_api_key:
            providers.append("groq")
        if self.xai_api_key:
            providers.append("xai")
        if self.gemini_api_key:
            providers.append("google")
        return providers

    def rdab_env_dict(self) -> dict[str, str]:
        """
        Return a dict of env vars to inject when calling RealDataAgentBench.
        RDAB reads these directly from the environment.
        Only includes server-side keys — session keys are merged by the engine.
        """
        env: dict[str, str] = {}
        if self.anthropic_api_key:
            env["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        if self.openai_api_key:
            env["OPENAI_API_KEY"] = self.openai_api_key
        if self.groq_api_key:
            env["GROQ_API_KEY"] = self.groq_api_key
        if self.xai_api_key:
            env["XAI_API_KEY"] = self.xai_api_key
        if self.gemini_api_key:
            env["GEMINI_API_KEY"] = self.gemini_api_key
        return env

    def merged_live_providers(self, session_env: dict[str, str]) -> list[str]:
        """
        Combine server-side providers with any providers unlocked by session keys.

        Args:
            session_env: {ENV_VAR_NAME: key} dict from SessionKeys.to_env_dict()

        Returns:
            Deduplicated list of provider names that have a live API key.
        """
        from evaluation.pricing import PROVIDER_ENV_VARS  # avoid circular at module load

        providers = set(self.available_providers)
        for provider, env_var in PROVIDER_ENV_VARS.items():
            if env_var in session_env:
                providers.add(provider)
        return sorted(providers)

    def ensure_upload_dir(self) -> None:
        os.makedirs(self.upload_dir, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    settings = Settings()
    settings.ensure_upload_dir()
    return settings
