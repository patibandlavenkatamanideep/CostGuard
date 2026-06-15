"""
CostGuard Tether Replay — POST /replay

Reads a captured Tether run from a local SQLite database, replays every
prompt against an alternate model, scores both responses with the existing
heuristic scorer, and returns a quality delta with a 95% bootstrap CI.

Scope: text completions only. No streaming, tool calls, or multi-run replay.
See docs/INTEGRATION_MVP.md for the full spec.
"""

from __future__ import annotations

import contextlib
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from scipy import stats

from backend.logger import logger
from backend.proxy import _call_llm, _score_response_fast
from evaluation.tether_reader import TetherStep, iter_steps_for_run

router = APIRouter(prefix="/replay", tags=["Replay"])


# ─── Schemas ──────────────────────────────────────────────────────────────────


class ReplayRequest(BaseModel):
    tether_db_path: str = Field(description="Absolute path to Tether SQLite file.")
    run_id: str = Field(description="run_id from Tether's steps table.")
    alternate_model: str = Field(description="Model ID from CostGuard's pricing catalogue.")
    n_bootstrap_samples: int = Field(default=1000, ge=100, le=10_000)


class ReplayResponse(BaseModel):
    primary_model: str
    alternate_model: str
    n_calls: int
    primary_mean_score: float
    alternate_mean_score: float
    delta: float
    ci_low: float
    ci_high: float
    primary_cost_usd: float
    alternate_cost_usd: float
    savings_per_call_usd: float


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _parse_step(ts: TetherStep) -> dict | None:
    """Extract prompt, system_prompt, and response text from a TetherStep.

    Returns None if the step is unusable (empty prompt or malformed outputs).
    """
    messages: list[dict] = ts.inputs.get("messages", [])

    system_prompt = next(
        (m["content"] for m in messages if m.get("role") == "system" and m.get("content")),
        None,
    )
    user_prompt = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user" and m.get("content")),
        None,
    )
    if not user_prompt or not user_prompt.strip():
        return None

    if ts.outputs is None:
        return None
    try:
        response_text = ts.outputs["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return None

    return {
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "response": response_text,
        "input_tokens": ts.input_tokens,
        "output_tokens": ts.output_tokens,
        "cost_usd": ts.cost_usd,
        "model": ts.model,
    }


def _bootstrap_ci(
    deltas: list[float],
    n_samples: int,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrap CI on the mean delta using scipy.stats.bootstrap.

    Uses the percentile method with a fixed seed for reproducibility.

    Args:
        deltas: Per-call quality deltas (alternate_score - primary_score).
        n_samples: Number of bootstrap resamples (100–10,000).
        seed: Random seed for reproducibility.

    Returns:
        (ci_low, ci_high) — 2.5th and 97.5th percentiles of the bootstrap
        distribution of the mean.
    """
    arr = np.array(deltas, dtype=float)
    # scipy requires ≥ 2 samples for a meaningful CI; handle the edge case.
    if len(arr) < 2:
        v = float(arr[0]) if len(arr) == 1 else 0.0
        return v, v

    result = stats.bootstrap(
        (arr,),
        statistic=np.mean,
        n_resamples=n_samples,
        confidence_level=0.95,
        random_state=seed,
        method="percentile",
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


# ─── Endpoint ─────────────────────────────────────────────────────────────────


@router.post(
    "", response_model=ReplayResponse, summary="Replay a Tether run against an alternate model"
)
async def replay(req: ReplayRequest) -> ReplayResponse:
    """
    Read a captured Tether run, replay every prompt against `alternate_model`,
    and return a quality delta with a 95% bootstrap CI.

    Requires the alternate model's provider API key to be set in the environment
    (same keys used by POST /proxy).
    """
    # ── 1. Validate inputs ────────────────────────────────────────────────────
    if not Path(req.tether_db_path).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"tether_db_path does not exist: {req.tether_db_path}",
        )

    from evaluation.pricing import MODELS

    if req.alternate_model not in MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown alternate_model '{req.alternate_model}'. Check GET /models.",
        )

    # ── 2. Read steps from Tether via tether_reader ───────────────────────────
    try:
        raw_steps = list(iter_steps_for_run(req.tether_db_path, req.run_id))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot read Tether database: {exc}",
        ) from exc
    except sqlite3.OperationalError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot read Tether database: {exc}",
        ) from exc

    if not raw_steps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"run_id '{req.run_id}' not found or has no completed llm_call steps.",
        )

    # ── 3. Parse and filter steps ─────────────────────────────────────────────
    steps = [s for ts in raw_steps if (s := _parse_step(ts)) is not None]
    if not steps:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Run has no usable steps (all prompts are empty or outputs are malformed).",
        )

    primary_model = next(
        (s["model"] for s in steps if s["model"]),
        "unknown",
    )

    # ── 4. Score originals + replay ───────────────────────────────────────────
    alt_pricing = MODELS[req.alternate_model]
    primary_scores: list[float] = []
    alternate_scores: list[float] = []
    primary_cost_total = Decimal("0")
    alternate_cost_total = Decimal("0")

    for i, step in enumerate(steps):
        primary_score = _score_response_fast(step["prompt"], step["response"])
        primary_scores.append(primary_score.rdab_score)

        if step["cost_usd"] is not None:
            with contextlib.suppress(InvalidOperation):
                primary_cost_total += Decimal(str(step["cost_usd"]))

        try:
            import asyncio

            async with asyncio.timeout(30.0):
                alt_text, alt_in_tok, alt_out_tok = await _call_llm(
                    model_id=req.alternate_model,
                    prompt=step["prompt"],
                    system_prompt=step["system_prompt"],
                    max_tokens=512,
                    temperature=0.0,
                    api_key=None,
                )
        except TimeoutError:
            logger.warning(f"[replay] step {i} timed out — scoring as empty response")
            alt_text, alt_in_tok, alt_out_tok = "", 0, 0

        alt_score = _score_response_fast(step["prompt"], alt_text)
        alternate_scores.append(alt_score.rdab_score)
        alternate_cost_total += Decimal(str(alt_pricing.estimate_cost(alt_in_tok, alt_out_tok)))

    # ── 5. Bootstrap CI (scipy percentile, seed=42) ───────────────────────────
    n_calls = len(steps)
    deltas = [a - p for a, p in zip(alternate_scores, primary_scores, strict=True)]
    ci_low, ci_high = _bootstrap_ci(deltas, req.n_bootstrap_samples)

    primary_mean = mean(primary_scores)
    alternate_mean = mean(alternate_scores)
    delta = alternate_mean - primary_mean

    primary_cost_f = float(primary_cost_total)
    alternate_cost_f = float(alternate_cost_total)

    logger.info(
        f"[replay] run={req.run_id} n={n_calls} "
        f"primary={primary_model} alt={req.alternate_model} "
        f"delta={delta:+.4f} ci=[{ci_low:.4f}, {ci_high:.4f}]"
    )

    return ReplayResponse(
        primary_model=primary_model,
        alternate_model=req.alternate_model,
        n_calls=n_calls,
        primary_mean_score=round(primary_mean, 4),
        alternate_mean_score=round(alternate_mean, 4),
        delta=round(delta, 4),
        ci_low=round(ci_low, 4),
        ci_high=round(ci_high, 4),
        primary_cost_usd=round(primary_cost_f, 8),
        alternate_cost_usd=round(alternate_cost_f, 8),
        savings_per_call_usd=round((primary_cost_f - alternate_cost_f) / n_calls, 8),
    )
