"""
CostGuard Evaluation Engine — powered by RealDataAgentBench.

This module wraps realdataagentbench's harness and scoring pipeline to evaluate
any user-uploaded dataset across all supported LLM providers.

Architecture:
  1. Load & sample the uploaded CSV/Parquet
  2. Dynamically build a RDAB-compatible TaskSchema from the user's data
  3. For each available model:
     a. Inject API keys into the environment (server-side + session keys merged)
     b. Run the RDAB Agent (via harness.Agent + harness.Runner) on the task
     c. Score with RDAB's CompositeScorer (correctness, code_quality,
        efficiency, stat_validity)
  4. Models without a live API key receive a deterministic simulation score
  5. Rank all models by composite score and return the recommendation

Mode determination:
  - LIVE: at least one provider has a real API key (server env OR session key)
  - SIMULATION: no keys available at all — uses calibrated historical scores

RDAB scoring dimensions (ref: realdataagentbench/scoring/):
  - Correctness (50%): answer accuracy vs ground truth
  - Code Quality (20%): vectorisation, naming, magic numbers
  - Efficiency (15%): token + step budget adherence
  - Stat Validity (15%): statistical rigour in answers
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import get_settings
from backend.logger import logger
from backend.models import (
    DatasetStats,
    EvalMode,
    EvalStatus,
    ModelResult,
    ModelTier,
    RDABScoreCard,
    SessionKeys,
)
from evaluation.data_loader import (
    compute_stats,
    dataframe_to_prompt_text,
    load_bytes,
    sample_dataframe,
)
from evaluation.pricing import ModelPricing, get_models_for_providers
from evaluation.token_counter import estimate_batch_tokens

settings = get_settings()

# ─── RDAB availability check ─────────────────────────────────────────────────

try:
    from realdataagentbench.harness import Agent, Runner
    from realdataagentbench.harness.tools import get_dataframe_info, run_code
    from realdataagentbench.scoring import CompositeScorer
    from realdataagentbench.core import TaskRegistry

    RDAB_AVAILABLE = True
    logger.info("RealDataAgentBench package loaded successfully")
except ImportError as _rdab_err:
    RDAB_AVAILABLE = False
    logger.warning(
        f"RealDataAgentBench not installed ({_rdab_err}). "
        "All models will use simulation mode. "
        "Install with: pip install git+https://github.com/patibandlavenkatamanideep/RealDataAgentBench.git"
    )


# ─── Dynamic TaskSchema builder ──────────────────────────────────────────────

def _build_task_dict(
    df: pd.DataFrame,
    data_text: str,
    task_description: str,
    questions: list[str],
) -> dict[str, Any]:
    """
    Build a RDAB-compatible task dictionary from user-uploaded data.
    This mimics the structure of tasks/*.yaml but is generated on-the-fly.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Derive ground-truth hints from actual data (used by CorrectnessScorer)
    ground_truth: dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
    }
    if num_cols:
        col = num_cols[0]
        ground_truth["first_numeric_mean"] = round(float(df[col].mean()), 4)
        ground_truth["first_numeric_col"] = col
    if cat_cols:
        col = cat_cols[0]
        top_val = df[col].mode().iloc[0] if not df[col].mode().empty else ""
        ground_truth["first_categorical_top"] = str(top_val)
        ground_truth["first_categorical_col"] = col

    return {
        "id": f"user_upload_{uuid.uuid4().hex[:6]}",
        "category": "eda",
        "difficulty": "medium",
        "description": task_description,
        "prompt": (
            f"You are a data analyst. Analyze the following dataset.\n\n"
            f"{data_text}\n\n"
            f"Task: {task_description}\n\n"
            "Answer these questions:\n"
            + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        ),
        "ground_truth": ground_truth,
        "dataset": {"type": "inline"},
        "evaluation": {
            "max_steps": 10,
            "timeout_seconds": settings.eval_timeout_seconds,
            "allowed_tools": ["run_code", "get_dataframe_info"],
        },
        "scoring": {
            "correctness": 0.50,
            "code_quality": 0.20,
            "efficiency": 0.15,
            "stat_validity": 0.15,
        },
    }


# ─── RDAB live evaluation ─────────────────────────────────────────────────────

async def _run_rdab_agent(
    pricing: ModelPricing,
    task_dict: dict[str, Any],
    df: pd.DataFrame,
    merged_env: dict[str, str],
) -> tuple[dict[str, Any], float]:
    """
    Run a single RDAB agent evaluation for one model.

    Args:
        pricing: Model pricing / metadata object.
        task_dict: RDAB task specification.
        df: Sampled dataframe.
        merged_env: Combined server + session API key env vars.

    Returns:
        (result_dict, latency_ms).
    Raises on timeout or API error — caller catches and falls back to simulation.
    """
    old_env: dict[str, str | None] = {}
    for k, v in merged_env.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    tmp_path: str | None = None
    try:
        start = time.monotonic()

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        df.to_parquet(tmp_path, index=False)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _rdab_sync_run,
            pricing.rdab_alias,
            task_dict,
            tmp_path,
        )

        latency_ms = (time.monotonic() - start) * 1000
        return result, latency_ms

    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _rdab_sync_run(
    model_alias: str,
    task_dict: dict[str, Any],
    data_path: str,
) -> dict[str, Any]:
    """
    Synchronous RDAB agent run — called from a thread pool executor.
    Uses RDAB's Agent directly since we have a dynamic task (not a YAML file).
    """
    from realdataagentbench.harness import Agent
    from realdataagentbench.harness.tracer import Tracer

    df = pd.read_parquet(data_path)
    tracer = Tracer()

    agent = Agent(model=model_alias)
    result = agent.run(
        prompt=task_dict["prompt"],
        dataframe=df,
        tracer=tracer,
        max_steps=task_dict["evaluation"]["max_steps"],
        timeout=task_dict["evaluation"]["timeout_seconds"],
    )

    return {
        "task_id": task_dict["id"],
        "model": model_alias,
        "status": result.get("status", "success"),
        "final_answer": result.get("final_answer", ""),
        "trace": tracer.to_dict(),
        "error": result.get("error"),
    }


def _rdab_score(
    task_dict: dict[str, Any],
    result: dict[str, Any],
) -> RDABScoreCard:
    """Run RDAB's CompositeScorer on a result dict."""
    from realdataagentbench.scoring import CompositeScorer

    class _MinimalTask:
        id = task_dict["id"]
        difficulty = task_dict["difficulty"]
        ground_truth = task_dict["ground_truth"]
        evaluation = task_dict["evaluation"]
        scoring = task_dict["scoring"]

    scorer = CompositeScorer()
    scorecard = scorer.score(task=_MinimalTask(), result=result)

    trace = result.get("trace", {})
    token_count = trace.get("total_input_tokens", 0) + trace.get("total_output_tokens", 0)
    step_count = len(trace.get("steps", []))

    return RDABScoreCard(
        rdab_score=round(float(scorecard.dab_score), 4),
        correctness=round(float(scorecard.correctness), 4),
        code_quality=round(float(scorecard.code_quality), 4),
        efficiency=round(float(scorecard.efficiency), 4),
        stat_validity=round(float(scorecard.stat_validity), 4),
        token_count=token_count,
        step_count=step_count,
        simulated=False,
    )


# ─── Simulation fallback ──────────────────────────────────────────────────────

def _simulate_scorecard(pricing: ModelPricing, seed: int = 42) -> tuple[RDABScoreCard, float]:
    """
    Produce a deterministic simulated RDAB scorecard for models without a live key.
    Based on RDAB benchmark findings (April 2026 leaderboard data).

    Key empirical findings from RDAB's 163 benchmark runs:
    - GPT-4.1 leads EDA/inference tasks
    - Llama 3.3-70B outperforms on modeling tasks
    - Gemini 2.5 Flash = best cost-per-score
    - All models score ~0.25 on stat_validity
    - Claude Haiku consumed 608K tokens on tasks where GPT-4.1 used 30K
    """
    import hashlib

    h = int(hashlib.md5(f"{pricing.model_id}{seed}".encode()).hexdigest(), 16)
    rng = (h % 10000) / 10000.0

    base = {
        "premium": {"correctness": 0.88, "code_quality": 0.79, "efficiency": 0.78, "stat_validity": 0.26},
        "balanced": {"correctness": 0.78, "code_quality": 0.72, "efficiency": 0.84, "stat_validity": 0.24},
        "economy":  {"correctness": 0.68, "code_quality": 0.66, "efficiency": 0.88, "stat_validity": 0.22},
    }[pricing.tier]

    overrides: dict[str, dict[str, float]] = {
        "gpt-4.1":                    {"correctness": 0.93, "efficiency": 0.97},
        "gemini-2.5-flash":           {"efficiency": 0.95},
        "llama-3.3-70b-versatile":    {"correctness": 0.82, "code_quality": 0.78},
        "claude-sonnet-4-6":          {"correctness": 0.90, "efficiency": 0.65},
        "claude-haiku-4-5-20251001":  {"efficiency": 0.40},
        "grok-3":                     {"code_quality": 0.55},
        "grok-3-mini":                {"code_quality": 0.50},
    }
    for key, vals in overrides.get(pricing.model_id, {}).items():
        base[key] = vals

    def _jitter(v: float, scale: float = 0.04) -> float:
        return max(0.0, min(1.0, v + (rng - 0.5) * scale))

    correctness   = _jitter(base["correctness"])
    code_quality  = _jitter(base["code_quality"])
    efficiency    = _jitter(base["efficiency"])
    stat_validity = _jitter(base["stat_validity"], scale=0.06)

    weights = {"correctness": 0.50, "code_quality": 0.20, "efficiency": 0.15, "stat_validity": 0.15}
    rdab_score = (
        correctness   * weights["correctness"] +
        code_quality  * weights["code_quality"] +
        efficiency    * weights["efficiency"] +
        stat_validity * weights["stat_validity"]
    )

    base_lat = {"premium": 1400, "balanced": 700, "economy": 380}[pricing.tier]
    latency = base_lat + (rng - 0.5) * 400

    sc = RDABScoreCard(
        rdab_score=round(rdab_score, 4),
        correctness=round(correctness, 4),
        code_quality=round(code_quality, 4),
        efficiency=round(efficiency, 4),
        stat_validity=round(stat_validity, 4),
        token_count=0,
        step_count=0,
        simulated=True,
    )
    return sc, round(latency, 1)


# ─── Per-model evaluation ─────────────────────────────────────────────────────

async def _evaluate_model(
    pricing: ModelPricing,
    task_dict: dict[str, Any],
    df: pd.DataFrame,
    questions: list[str],
    live_providers: list[str],
    merged_env: dict[str, str],
) -> ModelResult:
    """
    Evaluate one model: live RDAB agent if a key is available for its provider,
    else deterministic simulation.

    Args:
        pricing: Model spec.
        task_dict: RDAB task.
        df: Sampled dataset.
        questions: Generated evaluation questions.
        live_providers: Providers with a real API key (server + session merged).
        merged_env: Full env var dict for RDAB injection (server + session merged).
    """
    data_text = dataframe_to_prompt_text(df)
    input_tokens, output_tokens = estimate_batch_tokens(
        system_prompt="You are an expert data analyst.",
        data_text=data_text,
        questions=questions,
    )

    latency_ms: float
    rdab_scorecard: RDABScoreCard

    if RDAB_AVAILABLE and pricing.provider in live_providers:
        try:
            result, latency_ms = await _run_rdab_agent(pricing, task_dict, df, merged_env)
            rdab_scorecard = _rdab_score(task_dict, result)
            logger.info(
                f"[RDAB live] {pricing.model_id}: "
                f"score={rdab_scorecard.rdab_score:.3f} lat={latency_ms:.0f}ms"
            )
        except Exception as exc:
            logger.warning(
                f"[RDAB live] {pricing.model_id} failed ({exc}), falling back to simulation"
            )
            rdab_scorecard, latency_ms = _simulate_scorecard(pricing)
    else:
        reason = "RDAB not installed" if not RDAB_AVAILABLE else f"no API key for '{pricing.provider}'"
        logger.info(f"[RDAB sim] {pricing.model_id} ({reason})")
        rdab_scorecard, latency_ms = _simulate_scorecard(pricing)

    total_cost = pricing.estimate_cost(input_tokens, output_tokens)

    config_snippet: dict[str, Any] = {
        "model": pricing.model_id,
        "provider": pricing.provider,
        "temperature": 0,
        "max_tokens": 512,
        "estimated_cost_per_run_usd": round(total_cost, 6),
        "rdab_score": rdab_scorecard.rdab_score,
        "scores": {
            "correctness":  rdab_scorecard.correctness,
            "code_quality": rdab_scorecard.code_quality,
            "efficiency":   rdab_scorecard.efficiency,
            "stat_validity": rdab_scorecard.stat_validity,
        },
    }

    return ModelResult(
        model_id=pricing.model_id,
        provider=pricing.provider,
        display_name=pricing.display_name,
        tier=ModelTier(pricing.tier),
        rdab_scorecard=rdab_scorecard,
        accuracy_score=rdab_scorecard.rdab_score,
        latency_ms=latency_ms,
        input_cost_per_1k=pricing.input_per_1k,
        output_cost_per_1k=pricing.output_per_1k,
        estimated_tokens_input=input_tokens,
        estimated_tokens_output=output_tokens,
        estimated_total_cost_usd=round(total_cost, 6),
        strengths=list(pricing.strengths),
        limitations=list(pricing.limitations),
        config_snippet=config_snippet,
    )


# ─── Main orchestrator ────────────────────────────────────────────────────────

async def run_evaluation(
    file_content: bytes,
    filename: str,
    file_size_bytes: int,
    task_description: str = "Analyze this dataset and answer questions about it.",
    num_questions: int = 5,
    session_keys: SessionKeys | None = None,
) -> dict[str, Any]:
    """
    Full CostGuard evaluation pipeline powered by RealDataAgentBench.

    Supports two modes:
      - LIVE: at least one provider has an API key (server env or session key)
      - SIMULATION: no keys → deterministic scores from RDAB benchmark history

    Steps:
      1. Load + validate uploaded file
      2. Merge server-side and session API keys; determine mode
      3. Compute dataset statistics
      4. Generate analytical questions
      5. Build a dynamic RDAB TaskSchema
      6. Evaluate all models (live RDAB agent or deterministic simulation)
      7. Rank by composite score and produce recommendation
      8. Return structured EvalResponse dict

    Args:
        file_content: Raw bytes of the uploaded file.
        filename: Original filename (used to detect CSV vs Parquet).
        file_size_bytes: Size in bytes (for stats display).
        task_description: User-provided intent for the data.
        num_questions: Number of RDAB evaluation questions to generate.
        session_keys: Optional per-request API keys from the UI (never persisted).
    """
    from evaluation.question_generator import generate_questions

    eval_id = str(uuid.uuid4())[:8]
    start_total = time.monotonic()

    # ── Step 1: Resolve keys and mode ────────────────────────────────────────
    sk = session_keys or SessionKeys()
    session_env = sk.to_env_dict()

    # Merge: session keys take precedence over server-side keys for the same provider
    merged_env = {**settings.rdab_env_dict(), **session_env}
    live_providers = settings.merged_live_providers(session_env)
    eval_mode = EvalMode.LIVE if live_providers else EvalMode.SIMULATION

    logger.info(
        f"[{eval_id}] Starting evaluation for '{filename}' | "
        f"mode={eval_mode.value} | live_providers={live_providers} | "
        f"RDAB available={RDAB_AVAILABLE}"
    )

    # ── Step 2: Load data ─────────────────────────────────────────────────────
    df = load_bytes(file_content, filename)
    sample_df = sample_dataframe(df)
    stats = compute_stats(df, filename, file_size_bytes)

    # ── Step 3: Generate questions ────────────────────────────────────────────
    questions = generate_questions(sample_df, num_questions=num_questions)
    logger.info(f"[{eval_id}] Generated {len(questions)} evaluation questions")

    # ── Step 4: Build dynamic task ────────────────────────────────────────────
    data_text = dataframe_to_prompt_text(sample_df)
    task_dict = _build_task_dict(sample_df, data_text, task_description, questions)

    # ── Step 5: Evaluate all models ───────────────────────────────────────────
    all_models = get_models_for_providers([])  # Always show all for comparison

    sem = asyncio.Semaphore(settings.eval_concurrency)

    async def bounded_eval(pricing: ModelPricing) -> ModelResult:
        async with sem:
            return await _evaluate_model(
                pricing, task_dict, sample_df, questions, live_providers, merged_env
            )

    results: list[ModelResult] = await asyncio.gather(
        *[bounded_eval(m) for m in all_models]
    )

    # ── Step 6: Rank and recommend ────────────────────────────────────────────
    max_cost = max((r.estimated_total_cost_usd for r in results), default=1.0) or 1.0

    def composite_score(r: ModelResult) -> float:
        cost_score = 1.0 - (r.estimated_total_cost_usd / max_cost)
        return r.rdab_scorecard.rdab_score * 0.6 + cost_score * 0.4

    ranked = sorted(results, key=composite_score, reverse=True)
    recommended = ranked[0]
    reason = _build_recommendation_reason(recommended, ranked, eval_mode)
    copyable_config = json.dumps(recommended.config_snippet, indent=2)

    duration_s = round(time.monotonic() - start_total, 2)
    logger.info(
        f"[{eval_id}] Completed in {duration_s}s | mode={eval_mode.value} | "
        f"Recommended: {recommended.model_id} "
        f"(RDAB={recommended.rdab_scorecard.rdab_score:.3f}, "
        f"cost=${recommended.estimated_total_cost_usd:.5f})"
    )

    return {
        "eval_id": eval_id,
        "status": EvalStatus.COMPLETED,
        "eval_mode": eval_mode,
        "live_providers": live_providers,
        "dataset_stats": stats,
        "results": ranked,
        "recommended_model": recommended,
        "total_eval_duration_s": duration_s,
        "recommendation_reason": reason,
        "copyable_config": copyable_config,
    }


def _build_recommendation_reason(
    best: ModelResult,
    ranked: list[ModelResult],
    eval_mode: EvalMode,
) -> str:
    """Generate a human-readable RDAB-informed recommendation."""
    sc = best.rdab_scorecard
    mode_note = " *(simulated scores — add API keys for live benchmarking)*" if eval_mode == EvalMode.SIMULATION else " *(live RDAB benchmark)*"

    lines = [
        f"**{best.display_name}** achieves the best balance of RDAB score and cost for your data{mode_note}.",
        f"RDAB composite score: **{sc.rdab_score:.1%}** "
        f"(correctness {sc.correctness:.0%} · code quality {sc.code_quality:.0%} · "
        f"efficiency {sc.efficiency:.0%} · stat validity {sc.stat_validity:.0%}).",
        f"Estimated cost: **${best.estimated_total_cost_usd:.5f} per run**.",
    ]

    runner_up = ranked[1] if len(ranked) > 1 else None
    if runner_up:
        saving = runner_up.estimated_total_cost_usd - best.estimated_total_cost_usd
        if saving > 0:
            lines.append(
                f"Saves **${saving:.5f}** per run vs "
                f"{runner_up.display_name} ({runner_up.rdab_scorecard.rdab_score:.1%} RDAB)."
            )

    lines.append(f"Key strengths: {', '.join(best.strengths[:2])}.")
    return " ".join(lines)
