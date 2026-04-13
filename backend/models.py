"""Pydantic schemas for API request/response objects."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalMode(str, Enum):
    LIVE = "live"
    SIMULATION = "simulation"


class ModelTier(str, Enum):
    PREMIUM = "premium"
    BALANCED = "balanced"
    ECONOMY = "economy"


class DatasetStats(BaseModel):
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_pct: float = Field(ge=0, le=100)
    file_size_kb: float
    file_format: str  # CSV | PARQUET


class RDABScoreCard(BaseModel):
    """
    Four-dimensional score from RealDataAgentBench evaluation.
    Mirrors realdataagentbench.scoring.ScoreCard.
    """
    rdab_score: float = Field(ge=0, le=1, description="Composite RDAB score (weighted)")
    correctness: float = Field(ge=0, le=1, description="Ground-truth accuracy (50% weight)")
    code_quality: float = Field(ge=0, le=1, description="Code pattern quality (20% weight)")
    efficiency: float = Field(ge=0, le=1, description="Token + step budget efficiency (15% weight)")
    stat_validity: float = Field(ge=0, le=1, description="Statistical rigour (15% weight)")
    token_count: int = Field(default=0, description="Total tokens consumed")
    step_count: int = Field(default=0, description="Total agent steps taken")
    simulated: bool = Field(default=False, description="True if no live API key was available")


class ModelResult(BaseModel):
    model_id: str
    provider: str
    display_name: str
    tier: ModelTier
    # RDAB 4-dimensional scores
    rdab_scorecard: RDABScoreCard
    # Legacy scalar for backward compat (= rdab_scorecard.rdab_score)
    accuracy_score: float = Field(ge=0, le=1)
    latency_ms: float
    input_cost_per_1k: float
    output_cost_per_1k: float
    estimated_tokens_input: int
    estimated_tokens_output: int
    estimated_total_cost_usd: float
    strengths: list[str]
    limitations: list[str]
    config_snippet: dict[str, Any]


class EvalRequest(BaseModel):
    filename: str
    task_description: str = Field(
        default="Analyze this dataset and answer questions about it.",
        max_length=500,
    )
    num_questions: int = Field(default=5, ge=1, le=20)


class SessionKeys(BaseModel):
    """
    Per-request API keys supplied by the user via the UI.
    Never persisted — only used for the duration of a single evaluation call.
    """
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    xai_api_key: str | None = None
    gemini_api_key: str | None = None

    def to_env_dict(self) -> dict[str, str]:
        """Return a {ENV_VAR_NAME: key_value} dict for RDAB environment injection."""
        mapping = {
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "openai_api_key": "OPENAI_API_KEY",
            "groq_api_key": "GROQ_API_KEY",
            "xai_api_key": "XAI_API_KEY",
            "gemini_api_key": "GEMINI_API_KEY",
        }
        return {
            env_var: getattr(self, attr)
            for attr, env_var in mapping.items()
            if getattr(self, attr)
        }

    def live_providers(self) -> list[str]:
        """Return provider names for which a session key was supplied."""
        providers: list[str] = []
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

    def has_any_key(self) -> bool:
        return bool(self.to_env_dict())


class EvalResponse(BaseModel):
    eval_id: str
    status: EvalStatus
    eval_mode: EvalMode = Field(
        default=EvalMode.SIMULATION,
        description="live if any provider key was used, simulation otherwise",
    )
    live_providers: list[str] = Field(
        default_factory=list,
        description="Providers that ran with real API keys in this evaluation",
    )
    dataset_stats: DatasetStats
    results: list[ModelResult]
    recommended_model: ModelResult
    total_eval_duration_s: float
    recommendation_reason: str
    copyable_config: str  # JSON string of the recommended config


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    available_providers: list[str]
    environment: str
    rdab_available: bool = Field(default=False, description="Whether RDAB package is importable")


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
