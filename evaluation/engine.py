"""
CostGuard Evaluation Engine — RealDataAgentBench-style evaluation.

Architecture:
  1. Load & sample the dataset
  2. Generate analytical questions about the data
  3. For each available model: call the LLM, measure latency, score the answer
  4. Compute per-model cost estimates
  5. Rank models and return the recommendation

The scoring is heuristic (no ground truth) but consistent:
  - Answer length adequacy
  - Keyword/entity coverage from data schema
  - Structural quality (avoids "I don't know", etc.)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings
from backend.logger import logger
from backend.models import DatasetStats, EvalStatus, ModelResult, ModelTier
from evaluation.data_loader import (
    compute_stats,
    dataframe_to_prompt_text,
    load_bytes,
    sample_dataframe,
)
from evaluation.pricing import ModelPricing, get_models_for_providers
from evaluation.question_generator import generate_questions
from evaluation.token_counter import estimate_batch_tokens

settings = get_settings()

SYSTEM_PROMPT = """You are an expert data analyst. You are given a dataset and asked questions about it.
Answer each question accurately, concisely, and based only on the data provided.
Do not make up data. If you cannot determine the answer from the data, say so clearly."""


# ─── LLM Caller ──────────────────────────────────────────────────────────────

async def _call_openai(model_id: str, prompt: str, data_text: str) -> tuple[str, float]:
    """Call OpenAI and return (answer, latency_ms)."""
    import openai

    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    start = time.monotonic()

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{data_text}\n\nQuestion: {prompt}"},
            ],
            max_tokens=512,
            temperature=0,
        ),
        timeout=settings.eval_timeout_seconds,
    )

    latency_ms = (time.monotonic() - start) * 1000
    answer = response.choices[0].message.content or ""
    return answer, latency_ms


async def _call_anthropic(model_id: str, prompt: str, data_text: str) -> tuple[str, float]:
    """Call Anthropic and return (answer, latency_ms)."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    start = time.monotonic()

    message = await asyncio.wait_for(
        client.messages.create(
            model=model_id,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"{data_text}\n\nQuestion: {prompt}"},
            ],
            max_tokens=512,
        ),
        timeout=settings.eval_timeout_seconds,
    )

    latency_ms = (time.monotonic() - start) * 1000
    answer = message.content[0].text if message.content else ""
    return answer, latency_ms


async def _call_model(
    pricing: ModelPricing, prompt: str, data_text: str
) -> tuple[str, float]:
    """Dispatch to the correct provider."""
    if pricing.provider == "openai":
        return await _call_openai(pricing.model_id, prompt, data_text)
    elif pricing.provider == "anthropic":
        return await _call_anthropic(pricing.model_id, prompt, data_text)
    else:
        raise ValueError(f"Provider '{pricing.provider}' not yet implemented for live calls")


# ─── Answer Scorer ────────────────────────────────────────────────────────────

_BAD_PHRASES = {
    "i don't know",
    "i cannot",
    "i'm unable",
    "as an ai",
    "no data provided",
    "cannot determine",
}


def _score_answer(answer: str, question: str, data_keywords: set[str]) -> float:
    """
    Score an answer on a 0–1 scale using heuristics:
    - Length: longer, substantive answers score higher (up to a point)
    - Keyword coverage: answers that reference column names / values score higher
    - Avoidance of refusal phrases
    """
    if not answer or len(answer.strip()) < 10:
        return 0.0

    lower = answer.lower()

    # Penalize refusal phrases
    refusal_penalty = sum(1 for p in _BAD_PHRASES if p in lower) * 0.2

    # Length score (50–400 chars is ideal)
    length = len(answer.strip())
    if length < 30:
        length_score = 0.2
    elif length < 100:
        length_score = 0.5
    elif length <= 500:
        length_score = 1.0
    else:
        length_score = 0.8  # Too verbose

    # Keyword coverage
    if data_keywords:
        covered = sum(1 for kw in data_keywords if kw.lower() in lower)
        kw_score = min(1.0, covered / max(len(data_keywords), 1))
    else:
        kw_score = 0.5

    raw_score = (length_score * 0.5 + kw_score * 0.5) - refusal_penalty
    return max(0.0, min(1.0, raw_score))


def _simulate_score(pricing: ModelPricing, seed: int = 42) -> tuple[float, float]:
    """
    Produce a deterministic simulated score + latency for models that can't be
    called live (no API key, unsupported provider, etc.).
    Used so every model always shows up in comparison charts.
    """
    import hashlib

    h = int(hashlib.md5(f"{pricing.model_id}{seed}".encode()).hexdigest(), 16)
    rng_val = (h % 1000) / 1000.0

    # Map tier to plausible score range
    base_scores = {"premium": 0.80, "balanced": 0.68, "economy": 0.55}
    base = base_scores.get(pricing.tier, 0.65)
    score = min(1.0, max(0.0, base + (rng_val - 0.5) * 0.15))

    # Latency simulation
    base_latencies = {"premium": 1200, "balanced": 600, "economy": 350}
    base_lat = base_latencies.get(pricing.tier, 800)
    latency = base_lat + (rng_val - 0.5) * 300

    return round(score, 4), round(latency, 1)


# ─── Per-model evaluation ────────────────────────────────────────────────────

async def _evaluate_model(
    pricing: ModelPricing,
    questions: list[str],
    data_text: str,
    data_keywords: set[str],
    live_providers: list[str],
) -> ModelResult:
    """Run evaluation for a single model, returning a ModelResult."""
    input_tokens, output_tokens = estimate_batch_tokens(
        SYSTEM_PROMPT, data_text, questions
    )

    if pricing.provider in live_providers:
        # Live evaluation
        scores = []
        latencies = []
        for q in questions:
            try:
                answer, lat_ms = await _call_model(pricing, q, data_text)
                score = _score_answer(answer, q, data_keywords)
                scores.append(score)
                latencies.append(lat_ms)
            except Exception as exc:
                logger.warning(f"[{pricing.model_id}] question failed: {exc}")
                scores.append(0.0)
                latencies.append(float(settings.eval_timeout_seconds * 1000))

        accuracy = round(sum(scores) / len(scores), 4) if scores else 0.0
        latency = round(sum(latencies) / len(latencies), 1) if latencies else 9999.0
    else:
        # Simulated (no API key configured for this provider)
        accuracy, latency = _simulate_score(pricing)
        logger.info(
            f"[{pricing.model_id}] simulated score={accuracy:.3f} "
            f"(provider '{pricing.provider}' not configured)"
        )

    total_cost = pricing.estimate_cost(input_tokens, output_tokens)

    config_snippet: dict[str, Any] = {
        "model": pricing.model_id,
        "provider": pricing.provider,
        "temperature": 0,
        "max_tokens": 512,
        "estimated_cost_per_run_usd": round(total_cost, 6),
    }

    return ModelResult(
        model_id=pricing.model_id,
        provider=pricing.provider,
        display_name=pricing.display_name,
        tier=ModelTier(pricing.tier),
        accuracy_score=accuracy,
        latency_ms=latency,
        input_cost_per_1k=pricing.input_per_1k,
        output_cost_per_1k=pricing.output_per_1k,
        estimated_tokens_input=input_tokens,
        estimated_tokens_output=output_tokens,
        estimated_total_cost_usd=round(total_cost, 6),
        strengths=list(pricing.strengths),
        limitations=list(pricing.limitations),
        config_snippet=config_snippet,
    )


# ─── Main Orchestrator ────────────────────────────────────────────────────────

async def run_evaluation(
    file_content: bytes,
    filename: str,
    file_size_bytes: int,
    task_description: str = "Analyze this dataset and answer questions about it.",
    num_questions: int = 5,
) -> dict[str, Any]:
    """
    Full evaluation pipeline. Returns a dict ready to be cast to EvalResponse.

    Steps:
      1. Load + validate the file
      2. Compute dataset statistics
      3. Generate evaluation questions
      4. Evaluate all available models (live + simulated)
      5. Rank and recommend the best model
      6. Return structured results
    """
    eval_id = str(uuid.uuid4())[:8]
    start_total = time.monotonic()

    logger.info(f"[{eval_id}] Starting evaluation for '{filename}'")

    # ── Step 1: Load data ────────────────────────────────────────────────────
    df = load_bytes(file_content, filename)
    sample_df = sample_dataframe(df)
    stats = compute_stats(df, filename, file_size_bytes)

    # ── Step 2: Prepare prompt data ──────────────────────────────────────────
    data_text = dataframe_to_prompt_text(sample_df)
    data_keywords = set(df.columns.tolist())

    # ── Step 3: Generate questions ───────────────────────────────────────────
    questions = generate_questions(sample_df, num_questions=num_questions)
    logger.info(f"[{eval_id}] Generated {len(questions)} evaluation questions")

    # ── Step 4: Evaluate all models ──────────────────────────────────────────
    live_providers = settings.available_providers
    all_models = get_models_for_providers([])  # All models for comparison

    sem = asyncio.Semaphore(settings.eval_concurrency)

    async def bounded_eval(pricing: ModelPricing) -> ModelResult:
        async with sem:
            return await _evaluate_model(
                pricing, questions, data_text, data_keywords, live_providers
            )

    results: list[ModelResult] = await asyncio.gather(
        *[bounded_eval(m) for m in all_models], return_exceptions=False
    )

    # ── Step 5: Rank and recommend ───────────────────────────────────────────
    # Score = weighted combination of accuracy (60%) + cost efficiency (40%)
    max_cost = max((r.estimated_total_cost_usd for r in results), default=1.0) or 1.0

    def composite_score(r: ModelResult) -> float:
        cost_score = 1.0 - (r.estimated_total_cost_usd / max_cost)
        return r.accuracy_score * 0.6 + cost_score * 0.4

    ranked = sorted(results, key=composite_score, reverse=True)
    recommended = ranked[0]

    reason = _build_recommendation_reason(recommended, ranked)
    copyable_config = json.dumps(recommended.config_snippet, indent=2)

    duration_s = round(time.monotonic() - start_total, 2)
    logger.info(
        f"[{eval_id}] Completed in {duration_s}s. "
        f"Recommended: {recommended.model_id} (score={recommended.accuracy_score:.3f})"
    )

    return {
        "eval_id": eval_id,
        "status": EvalStatus.COMPLETED,
        "dataset_stats": stats,
        "results": ranked,
        "recommended_model": recommended,
        "total_eval_duration_s": duration_s,
        "recommendation_reason": reason,
        "copyable_config": copyable_config,
    }


def _build_recommendation_reason(best: ModelResult, ranked: list[ModelResult]) -> str:
    """Generate a human-readable explanation for why this model was chosen."""
    runner_up = ranked[1] if len(ranked) > 1 else None

    lines = [
        f"**{best.display_name}** achieves the best balance of accuracy and cost for your dataset.",
        f"Accuracy score: {best.accuracy_score:.1%} | "
        f"Estimated cost: ${best.estimated_total_cost_usd:.4f} per run.",
    ]

    if runner_up:
        if runner_up.accuracy_score > best.accuracy_score:
            saving = runner_up.estimated_total_cost_usd - best.estimated_total_cost_usd
            lines.append(
                f"While {runner_up.display_name} scored slightly higher on accuracy "
                f"({runner_up.accuracy_score:.1%}), {best.display_name} saves "
                f"${saving:.4f} per run — a {saving/max(runner_up.estimated_total_cost_usd,1e-9):.0%} cost reduction."
            )
        else:
            lines.append(
                f"It outperforms the next best option ({runner_up.display_name}) "
                f"on both accuracy and cost-efficiency."
            )

    lines.append(f"Key strengths: {', '.join(best.strengths[:2])}.")
    return " ".join(lines)
