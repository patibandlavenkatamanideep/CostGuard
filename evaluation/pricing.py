"""
LLM pricing catalogue — aligned with RealDataAgentBench provider support.
Prices are in USD per 1,000 tokens (input / output).
Source: official provider pricing pages, April 2026.
Update this file as prices change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPricing:
    model_id: str           # Exact model ID used in API calls
    rdab_alias: str         # Alias recognised by RealDataAgentBench
    provider: str           # anthropic | openai | groq | xai | google
    display_name: str
    tier: str               # premium | balanced | economy
    input_per_1k: float     # USD per 1K input tokens
    output_per_1k: float    # USD per 1K output tokens
    context_window: int
    strengths: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000 * self.input_per_1k) + (
            output_tokens / 1000 * self.output_per_1k
        )


# ─── Model Catalogue (mirrors realdataagentbench/harness/pricing.py) ─────────
# Prices converted from per-1M to per-1K for compatibility.
MODELS: dict[str, ModelPricing] = {

    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-sonnet-4-6": ModelPricing(
        model_id="claude-sonnet-4-6",
        rdab_alias="sonnet",
        provider="anthropic",
        display_name="Claude Sonnet 4.6",
        tier="premium",
        input_per_1k=0.003,       # $3.00 / 1M
        output_per_1k=0.015,      # $15.00 / 1M
        context_window=200_000,
        strengths=["Best reasoning", "200K context", "Default RDAB model"],
        limitations=["Higher cost", "Slower than Haiku"],
    ),
    "claude-opus-4-6": ModelPricing(
        model_id="claude-opus-4-6",
        rdab_alias="opus",
        provider="anthropic",
        display_name="Claude Opus 4.6",
        tier="premium",
        input_per_1k=0.015,       # $15.00 / 1M
        output_per_1k=0.075,      # $75.00 / 1M
        context_window=200_000,
        strengths=["Highest Anthropic capability", "Complex reasoning"],
        limitations=["Most expensive model", "Slower"],
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        model_id="claude-haiku-4-5-20251001",
        rdab_alias="haiku",
        provider="anthropic",
        display_name="Claude Haiku 4.5",
        tier="economy",
        input_per_1k=0.00025,     # $0.25 / 1M
        output_per_1k=0.00125,    # $1.25 / 1M
        context_window=200_000,
        strengths=["Cheapest Anthropic", "Fast", "200K context"],
        limitations=["Less capable for complex reasoning"],
    ),

    # ── OpenAI ───────────────────────────────────────────────────────────────
    "gpt-4.1": ModelPricing(
        model_id="gpt-4.1",
        rdab_alias="gpt4.1",
        provider="openai",
        display_name="GPT-4.1",
        tier="premium",
        input_per_1k=0.002,       # $2.00 / 1M
        output_per_1k=0.008,      # $8.00 / 1M
        context_window=1_000_000,
        strengths=["Best cost-performance (RDAB benchmark)", "1M context"],
        limitations=["New model, less battle-tested"],
    ),
    "gpt-4.1-mini": ModelPricing(
        model_id="gpt-4.1-mini",
        rdab_alias="gpt4.1-mini",
        provider="openai",
        display_name="GPT-4.1 mini",
        tier="balanced",
        input_per_1k=0.0004,      # $0.40 / 1M
        output_per_1k=0.0016,     # $1.60 / 1M
        context_window=1_000_000,
        strengths=["Very fast", "1M context", "Low cost"],
        limitations=["Less reasoning depth vs GPT-4.1"],
    ),
    "gpt-4.1-nano": ModelPricing(
        model_id="gpt-4.1-nano",
        rdab_alias="gpt4.1-nano",
        provider="openai",
        display_name="GPT-4.1 nano",
        tier="economy",
        input_per_1k=0.0001,      # $0.10 / 1M
        output_per_1k=0.0004,     # $0.40 / 1M
        context_window=1_000_000,
        strengths=["Ultra-cheap", "Fast", "1M context"],
        limitations=["Weakest OpenAI option"],
    ),
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        rdab_alias="gpt4o",
        provider="openai",
        display_name="GPT-4o",
        tier="premium",
        input_per_1k=0.0025,      # $2.50 / 1M
        output_per_1k=0.010,      # $10.00 / 1M
        context_window=128_000,
        strengths=["Proven reliability", "Multi-modal"],
        limitations=["Smaller context than GPT-4.1"],
    ),
    "gpt-4o-mini": ModelPricing(
        model_id="gpt-4o-mini",
        rdab_alias="gpt4o-mini",
        provider="openai",
        display_name="GPT-4o mini",
        tier="balanced",
        input_per_1k=0.000150,    # $0.15 / 1M
        output_per_1k=0.000600,   # $0.60 / 1M
        context_window=128_000,
        strengths=["Very cheap", "Good structured output"],
        limitations=["Smaller context vs GPT-4.1 family"],
    ),
    "gpt-5": ModelPricing(
        model_id="gpt-5",
        rdab_alias="gpt5",
        provider="openai",
        display_name="GPT-5",
        tier="premium",
        input_per_1k=0.015,       # $15.00 / 1M
        output_per_1k=0.060,      # $60.00 / 1M
        context_window=128_000,
        strengths=["Highest OpenAI capability"],
        limitations=["Most expensive OpenAI", "~16× cost of GPT-4.1 for similar quality"],
    ),

    # ── Google Gemini ─────────────────────────────────────────────────────────
    "gemini-2.5-flash": ModelPricing(
        model_id="gemini-2.5-flash",
        rdab_alias="gemini-flash",
        provider="google",
        display_name="Gemini 2.5 Flash",
        tier="economy",
        input_per_1k=0.000075,    # $0.075 / 1M — cheapest overall
        output_per_1k=0.0003,     # $0.30 / 1M
        context_window=1_000_000,
        strengths=["Cheapest model overall", "Best cost-per-RDAB-score", "1M context"],
        limitations=["Less reasoning depth"],
    ),

    # ── Groq (ultra-fast inference) ───────────────────────────────────────────
    "llama-3.3-70b-versatile": ModelPricing(
        model_id="llama-3.3-70b-versatile",
        rdab_alias="llama",
        provider="groq",
        display_name="Llama 3.3 70B (Groq)",
        tier="balanced",
        input_per_1k=0.00059,     # $0.59 / 1M
        output_per_1k=0.00079,    # $0.79 / 1M
        context_window=128_000,
        strengths=["Ultra-fast (Groq)", "Outperforms on modeling tasks (RDAB)", "Open-source"],
        limitations=["Groq rate limits", "Variable on complex reasoning"],
    ),
    # ── xAI Grok ─────────────────────────────────────────────────────────────
    "grok-3-mini": ModelPricing(
        model_id="grok-3-mini",
        rdab_alias="grok-mini",
        provider="xai",
        display_name="Grok-3 mini",
        tier="balanced",
        input_per_1k=0.0003,      # $0.30 / 1M
        output_per_1k=0.0005,     # $0.50 / 1M
        context_window=131_072,
        strengths=["Fast", "Cheap", "Real-time knowledge"],
        limitations=["Less capable than Grok-3", "sklearn blind spot"],
    ),
}


def get_models_for_providers(providers: list[str]) -> list[ModelPricing]:
    """
    Return models available given a list of configured provider names.
    If providers is empty, returns all models (for demo/display).
    """
    if not providers:
        return list(MODELS.values())
    return [m for m in MODELS.values() if m.provider in providers]


def get_model(model_id: str) -> ModelPricing | None:
    return MODELS.get(model_id)


# Provider-to-canonical-env-var mapping (matches RDAB conventions)
PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "google": "GEMINI_API_KEY",
}
