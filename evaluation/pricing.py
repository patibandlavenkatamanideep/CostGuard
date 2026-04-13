"""
LLM pricing catalogue — sourced from official provider pricing pages.
Prices are in USD per 1,000 tokens (input / output).
Updated: 2025-Q1. Update this file as prices change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPricing:
    model_id: str
    provider: str
    display_name: str
    tier: str                        # premium | balanced | economy
    input_per_1k: float              # USD per 1K input tokens
    output_per_1k: float             # USD per 1K output tokens
    context_window: int              # max tokens
    strengths: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000 * self.input_per_1k) + (
            output_tokens / 1000 * self.output_per_1k
        )


# ─── Model Catalogue ─────────────────────────────────────────────────────────
MODELS: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        tier="premium",
        input_per_1k=0.0025,
        output_per_1k=0.010,
        context_window=128_000,
        strengths=["Best reasoning", "Code generation", "Multi-modal"],
        limitations=["Higher cost", "Slower than mini"],
    ),
    "gpt-4o-mini": ModelPricing(
        model_id="gpt-4o-mini",
        provider="openai",
        display_name="GPT-4o mini",
        tier="balanced",
        input_per_1k=0.000150,
        output_per_1k=0.000600,
        context_window=128_000,
        strengths=["Fast", "Very cheap", "Good for structured output"],
        limitations=["Less reasoning depth", "Weaker at complex tasks"],
    ),
    "gpt-3.5-turbo": ModelPricing(
        model_id="gpt-3.5-turbo",
        provider="openai",
        display_name="GPT-3.5 Turbo",
        tier="economy",
        input_per_1k=0.0005,
        output_per_1k=0.0015,
        context_window=16_385,
        strengths=["Very cheap", "Fast"],
        limitations=["Outdated", "Weaker instruction following", "Small context"],
    ),
    # Anthropic
    "claude-3-5-sonnet-20241022": ModelPricing(
        model_id="claude-3-5-sonnet-20241022",
        provider="anthropic",
        display_name="Claude 3.5 Sonnet",
        tier="premium",
        input_per_1k=0.003,
        output_per_1k=0.015,
        context_window=200_000,
        strengths=["Excellent reasoning", "200K context", "Data analysis"],
        limitations=["Higher cost", "Anthropic-only"],
    ),
    "claude-3-5-haiku-20241022": ModelPricing(
        model_id="claude-3-5-haiku-20241022",
        provider="anthropic",
        display_name="Claude 3.5 Haiku",
        tier="balanced",
        input_per_1k=0.0008,
        output_per_1k=0.004,
        context_window=200_000,
        strengths=["Fast", "Cheap", "Large context"],
        limitations=["Less powerful than Sonnet"],
    ),
    "claude-3-haiku-20240307": ModelPricing(
        model_id="claude-3-haiku-20240307",
        provider="anthropic",
        display_name="Claude 3 Haiku",
        tier="economy",
        input_per_1k=0.00025,
        output_per_1k=0.00125,
        context_window=200_000,
        strengths=["Cheapest Anthropic", "Fast", "Large context"],
        limitations=["Older model", "Less capable"],
    ),
    # Google
    "gemini-1.5-pro": ModelPricing(
        model_id="gemini-1.5-pro",
        provider="google",
        display_name="Gemini 1.5 Pro",
        tier="premium",
        input_per_1k=0.00125,
        output_per_1k=0.005,
        context_window=2_000_000,
        strengths=["2M context window", "Multi-modal", "Competitive pricing"],
        limitations=["Requires Google Cloud", "Variable quality"],
    ),
    "gemini-1.5-flash": ModelPricing(
        model_id="gemini-1.5-flash",
        provider="google",
        display_name="Gemini 1.5 Flash",
        tier="balanced",
        input_per_1k=0.000075,
        output_per_1k=0.0003,
        context_window=1_000_000,
        strengths=["Extremely cheap", "1M context", "Fast"],
        limitations=["Less reasoning depth"],
    ),
    # Groq (ultra-fast inference)
    "llama-3.1-70b-versatile": ModelPricing(
        model_id="llama-3.1-70b-versatile",
        provider="groq",
        display_name="Llama 3.1 70B (Groq)",
        tier="economy",
        input_per_1k=0.00059,
        output_per_1k=0.00079,
        context_window=128_000,
        strengths=["Ultra-fast (Groq)", "Open-source", "Good value"],
        limitations=["Variable quality", "Groq rate limits"],
    ),
    "mixtral-8x7b-32768": ModelPricing(
        model_id="mixtral-8x7b-32768",
        provider="groq",
        display_name="Mixtral 8x7B (Groq)",
        tier="economy",
        input_per_1k=0.00024,
        output_per_1k=0.00024,
        context_window=32_768,
        strengths=["Cheapest option", "Fast MoE architecture"],
        limitations=["Older model", "Smaller context"],
    ),
}


def get_models_for_providers(providers: list[str]) -> list[ModelPricing]:
    """Return models available given a list of configured provider names."""
    if not providers:
        # Return all models for demo/display purposes
        return list(MODELS.values())
    return [m for m in MODELS.values() if m.provider in providers]


def get_model(model_id: str) -> ModelPricing | None:
    return MODELS.get(model_id)
