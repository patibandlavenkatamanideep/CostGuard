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
import json
import random
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.logger import logger
from backend.proxy import _call_llm, _score_response_fast

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

def _read_steps(db_path: str, run_id: str) -> list[sqlite3.Row]:
    """Open Tether DB read-only and return llm_call steps for the run."""
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT inputs, outputs, model, input_tokens, output_tokens, cost_usd
            FROM   steps
            WHERE  run_id  = ?
              AND  kind    = 'llm_call'
              AND  outputs IS NOT NULL
            ORDER  BY sequence_number ASC
            """,
            (run_id,),
        ).fetchall()
        conn.close()
        return rows
    except sqlite3.OperationalError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot read Tether database: {exc}",
        ) from exc


def _parse_step(row: sqlite3.Row) -> dict | None:
    """
    Extract prompt, system_prompt, and response text from a Tether step row.
    Returns None if the row is unusable (malformed JSON, empty prompt).
    """
    try:
        inputs_dict = json.loads(row["inputs"])
        outputs_dict = json.loads(row["outputs"])
    except (json.JSONDecodeError, TypeError):
        return None

    messages: list[dict] = inputs_dict.get("messages", [])

    # Extract system prompt (first system message, if any)
    system_prompt = next(
        (m["content"] for m in messages if m.get("role") == "system" and m.get("content")),
        None,
    )

    # Build user prompt from the last user message (what the model was actually answering)
    user_prompt = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user" and m.get("content")),
        None,
    )
    if not user_prompt or not user_prompt.strip():
        return None

    # Extract response text from OpenAI-format outputs
    try:
        response_text = outputs_dict["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return None

    return {
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "response": response_text,
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "cost_usd": row["cost_usd"],
        "model": row["model"],
    }


def _bootstrap_ci(
    deltas: list[float],
    n_samples: int,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrap CI on the mean of deltas. Fixed seed for reproducibility."""
    rng = random.Random(seed)
    boot_means = sorted(
        mean(rng.choices(deltas, k=len(deltas)))
        for _ in range(n_samples)
    )
    lo = boot_means[int(0.025 * n_samples)]
    hi = boot_means[min(int(0.975 * n_samples), n_samples - 1)]
    return lo, hi


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("", response_model=ReplayResponse, summary="Replay a Tether run against an alternate model")
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

    # ── 2. Read steps from Tether ─────────────────────────────────────────────
    raw_rows = _read_steps(req.tether_db_path, req.run_id)
    if not raw_rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"run_id '{req.run_id}' not found or has no completed llm_call steps.",
        )

    # ── 3. Parse and filter steps ─────────────────────────────────────────────
    steps = [s for row in raw_rows if (s := _parse_step(row)) is not None]
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
        # Score the original captured response
        primary_score = _score_response_fast(step["prompt"], step["response"])
        primary_scores.append(primary_score.rdab_score)

        if step["cost_usd"] is not None:
            with contextlib.suppress(InvalidOperation):
                primary_cost_total += Decimal(str(step["cost_usd"]))

        # Replay against alternate model — 30 s timeout, no retry (spec: MVP scope)
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
        alternate_cost_total += Decimal(
            str(alt_pricing.estimate_cost(alt_in_tok, alt_out_tok))
        )

    # ── 5. Bootstrap CI ───────────────────────────────────────────────────────
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
