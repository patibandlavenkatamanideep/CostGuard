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


class ModelTier(str, Enum):
    PREMIUM = "premium"       # GPT-4o, Claude 3.5 Sonnet
    BALANCED = "balanced"     # GPT-4o-mini, Claude 3 Haiku
    ECONOMY = "economy"       # Groq, Together AI


class DatasetStats(BaseModel):
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_pct: float = Field(ge=0, le=100)
    file_size_kb: float
    file_format: str  # csv | parquet


class ModelResult(BaseModel):
    model_id: str
    provider: str
    display_name: str
    tier: ModelTier
    accuracy_score: float = Field(ge=0, le=1)
    latency_ms: float
    input_cost_per_1k: float   # USD per 1K input tokens
    output_cost_per_1k: float  # USD per 1K output tokens
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


class EvalResponse(BaseModel):
    eval_id: str
    status: EvalStatus
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


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
