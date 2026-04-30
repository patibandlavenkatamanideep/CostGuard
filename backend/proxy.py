"""
CostGuard Real-Time Proxy — the production interception layer.

Any LangGraph, CrewAI, or custom agent calls POST /proxy instead of calling
the LLM provider directly. CostGuard:
  1. Routes to the chosen model (or picks the best one)
  2. Calls the LLM (with per-call timeout + circuit breaker check)
  3. Scores the response with a RDAB-calibrated heuristic scorer
  4. Rejects + retries with a fallback model if validity < threshold
  5. Logs cost, latency, validity for every call
  6. Returns the response + full metadata

Note on validity scoring: the /proxy endpoint uses a fast heuristic scorer
(~1ms overhead). Full RDAB agent evaluation is available via POST /evaluate.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from backend.alerting import AlertEngine
from backend.circuit_breaker import CircuitBreakerRegistry
from backend.config import get_settings
from backend.logger import logger
from backend.metrics import (
    proxy_fallbacks_total,
    proxy_latency_seconds,
    proxy_rejections_total,
    proxy_requests_total,
)
from backend.models import RDABScoreCard
from evaluation.observability import log_proxy_call

settings = get_settings()
router = APIRouter(prefix="/proxy", tags=["Proxy"])
_alert_engine = AlertEngine()
_circuit_registry = CircuitBreakerRegistry()

# LLM call timeout — prevents hung providers from holding the event loop
_LLM_TIMEOUT_SECONDS = float(settings.model_config.get("llm_timeout_seconds", 30.0))

# ─── Module-level client singletons (server-key only; per-request keys get fresh clients) ──

_anthropic_default: Any = None
_openai_default: Any = None
_groq_default: Any = None
_xai_default: Any = None

# ─── Retry configuration ──────────────────────────────────────────────────────

_RETRY_ATTEMPTS = 3      # total attempts (1 original + 2 retries)
_RETRY_WAIT_MIN = 1.0    # seconds before first retry
_RETRY_WAIT_MAX = 8.0    # cap on exponential backoff


def _is_retryable(exc: BaseException) -> bool:
    """
    Return True for transient provider errors worth retrying.
    Excludes configuration errors (HTTPException) and per-attempt timeouts.
    """
    if isinstance(exc, (asyncio.TimeoutError, HTTPException)):
        return False
    s = str(exc).lower()
    return (
        "429" in str(exc)
        or "rate_limit" in s
        or "rate limit" in s
        or "too many requests" in s
        or "503" in str(exc)
        or "service unavailable" in s
        or "overloaded" in s
        or ("connection" in s and "refused" in s)
        or isinstance(exc, (ConnectionError, OSError))
    )


def _get_anthropic(key: str):
    global _anthropic_default
    import anthropic
    if key == settings.anthropic_api_key:
        if _anthropic_default is None:
            _anthropic_default = anthropic.AsyncAnthropic(api_key=key)
        return _anthropic_default
    return anthropic.AsyncAnthropic(api_key=key)


def _get_openai(key: str, base_url: str | None = None):
    global _openai_default
    import openai
    if base_url is None and key == settings.openai_api_key:
        if _openai_default is None:
            _openai_default = openai.AsyncOpenAI(api_key=key)
        return _openai_default
    kwargs: dict[str, Any] = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.AsyncOpenAI(**kwargs)


def _get_groq(key: str):
    global _groq_default
    import groq
    if key == settings.groq_api_key:
        if _groq_default is None:
            _groq_default = groq.AsyncGroq(api_key=key)
        return _groq_default
    return groq.AsyncGroq(api_key=key)


# ─── Request / Response schemas ───────────────────────────────────────────────

class ProxyRequest(BaseModel):
    model_id: str = Field(
        default="gpt-4.1",
        description="Model to call. Omit to let CostGuard pick the best available.",
    )
    prompt: str = Field(description="The user prompt to send to the model")
    system_prompt: str | None = Field(default=None)
    context: str | None = Field(
        default=None,
        description="Additional context (dataset description, task, etc.)",
    )
    max_tokens: int = Field(default=512, ge=1, le=16384)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    reject_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Minimum validity score to accept a response (0–1). Default 0.30.",
    )
    fallback_models: list[str] = Field(
        default_factory=list,
        description="Ordered list of fallback model IDs if primary is rejected. Empty = no fallback.",
    )
    api_key: str | None = Field(
        default=None,
        description="Provider API key override for this request (session-only, never stored)",
    )
    auto_select: bool = Field(
        default=False,
        description="If true, ignore model_id and let CostGuard pick the best available model",
    )


class ProxyResponse(BaseModel):
    call_id: str
    model_id: str
    provider: str
    content: str
    accepted: bool
    rejection_reason: str | None = None
    fallback_used: bool = False
    fallback_model_id: str | None = None
    validity_score: RDABScoreCard
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    attempts: int = 1
    circuit_breaker_state: str = "closed"


# ─── LLM callers ─────────────────────────────────────────────────────────────

async def _call_llm(
    model_id: str,
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    api_key: str | None,
) -> tuple[str, int, int]:
    """
    Call the LLM provider directly. Returns (response_text, input_tokens, output_tokens).
    Raises HTTPException on configuration errors; raises raw exceptions on provider errors
    (caller handles these for circuit breaker and fallback logic).
    """
    from evaluation.pricing import MODELS

    pricing = MODELS.get(model_id)
    if not pricing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model_id '{model_id}'. Check GET /models for valid IDs.",
        )

    provider = pricing.provider
    effective_key = api_key or _get_server_key(provider)
    if not effective_key:
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail=f"No API key for provider '{provider}'. Set {provider.upper()}_API_KEY or pass api_key.",
        )

    if provider == "anthropic":
        return await _call_anthropic(pricing.model_id, prompt, system_prompt, max_tokens, temperature, effective_key)
    elif provider == "openai":
        return await _call_openai(pricing.model_id, prompt, system_prompt, max_tokens, temperature, effective_key)
    elif provider == "groq":
        return await _call_groq(pricing.model_id, prompt, system_prompt, max_tokens, temperature, effective_key)
    elif provider == "google":
        return await _call_gemini(pricing.model_id, prompt, system_prompt, max_tokens, temperature, effective_key)
    elif provider == "xai":
        return await _call_xai(pricing.model_id, prompt, system_prompt, max_tokens, temperature, effective_key)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider '{provider}'",
        )


def _get_server_key(provider: str) -> str | None:
    key_map = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "groq": settings.groq_api_key,
        "xai": settings.xai_api_key,
        "google": settings.gemini_api_key,
    }
    return key_map.get(provider)


async def _call_anthropic(
    model_id: str, prompt: str, system: str | None,
    max_tokens: int, temp: float, key: str,
) -> tuple[str, int, int]:
    client = _get_anthropic(key)
    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temp,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = await client.messages.create(**kwargs)
    text = resp.content[0].text if resp.content else ""
    return text, resp.usage.input_tokens, resp.usage.output_tokens


async def _call_openai(
    model_id: str, prompt: str, system: str | None,
    max_tokens: int, temp: float, key: str,
) -> tuple[str, int, int]:
    client = _get_openai(key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temp,
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


async def _call_groq(
    model_id: str, prompt: str, system: str | None,
    max_tokens: int, temp: float, key: str,
) -> tuple[str, int, int]:
    client = _get_groq(key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temp,
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


async def _call_gemini(
    model_id: str, prompt: str, system: str | None,
    max_tokens: int, temp: float, key: str,
) -> tuple[str, int, int]:
    # google-genai (>= 1.0) provides a proper client object — no global state,
    # no lock needed. Concurrent calls with different keys are fully safe.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        system_instruction=system or "",
        max_output_tokens=max_tokens,
        temperature=temp,
    )
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=prompt,
        config=config,
    )
    text = response.text or ""
    usage = response.usage_metadata
    in_tok = (usage.prompt_token_count or 0) if usage else 0
    out_tok = (usage.candidates_token_count or 0) if usage else 0
    return text, in_tok, out_tok


async def _call_xai(
    model_id: str, prompt: str, system: str | None,
    max_tokens: int, temp: float, key: str,
) -> tuple[str, int, int]:
    client = _get_openai(key, base_url="https://api.x.ai/v1")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temp,
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


# ─── Retry wrapper ────────────────────────────────────────────────────────────

async def _call_llm_with_retry(
    model_id: str,
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    api_key: str | None,
) -> tuple[str, int, int]:
    """
    Call _call_llm with per-attempt timeout and exponential backoff for transient
    errors (429 rate limits, 503, connection refused).

    Non-retryable errors (HTTPException for bad config/auth, asyncio.TimeoutError
    for hung calls) are re-raised immediately.
    """

    @retry(
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=_RETRY_WAIT_MIN, max=_RETRY_WAIT_MAX),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    async def _attempt() -> tuple[str, int, int]:
        async with asyncio.timeout(_LLM_TIMEOUT_SECONDS):
            return await _call_llm(
                model_id, prompt, system_prompt, max_tokens, temperature, api_key
            )

    return await _attempt()


# ─── Heuristic validity scorer (RDAB-calibrated) ─────────────────────────────

def _score_response_fast(
    prompt: str,
    response_text: str,
    context: str | None = None,
) -> RDABScoreCard:
    """
    Fast heuristic validity scorer for proxy mode (~1ms overhead).

    This is NOT a full RDAB evaluation — it's a lightweight pre-filter calibrated
    against RDAB benchmark findings. Use POST /evaluate for full RDAB benchmarking.

    Scoring rationale (derived from RDAB benchmark patterns):
    - stat_validity: presence of uncertainty quantification markers
    - correctness: penalizes known failure-mode phrases
    - code_quality: neutral default (cannot determine from text alone)
    - efficiency: penalizes verbose responses
    """
    text = response_text.strip()

    if not text or len(text) < 10:
        return RDABScoreCard(
            rdab_score=0.0, correctness=0.0, code_quality=0.0,
            efficiency=0.0, stat_validity=0.0, token_count=0, step_count=1, simulated=True,
        )

    text_lower = text.lower()

    stat_markers = [
        "p-value", "p <", "p=", "confidence interval", "95% ci", "standard deviation",
        "std dev", "margin of error", "statistically significant", "not significant",
        "±", "uncertainty", "likely", "approximately", "roughly",
    ]
    stat_score = min(1.0, sum(0.15 for m in stat_markers if m in text_lower))

    error_patterns = [
        "i cannot", "i'm unable", "i don't know", "i am not able",
        "as an ai", "i don't have access", "error:", "exception:",
        "traceback", "syntaxerror", "typeerror",
    ]
    correctness_penalty = sum(0.2 for p in error_patterns if p in text_lower)
    correctness = max(0.0, 0.75 - correctness_penalty)

    code_quality = 0.70

    word_count = len(text.split())
    efficiency = 0.85 if word_count < 500 else (0.65 if word_count < 1000 else 0.45)

    rdab_score = (
        correctness * 0.50
        + code_quality * 0.20
        + efficiency * 0.15
        + stat_score * 0.15
    )

    return RDABScoreCard(
        rdab_score=round(rdab_score, 4),
        correctness=round(correctness, 4),
        code_quality=round(code_quality, 4),
        efficiency=round(efficiency, 4),
        stat_validity=round(stat_score, 4),
        token_count=word_count,
        step_count=1,
        simulated=True,
    )


# ─── Auto-select best available model ────────────────────────────────────────

def _pick_best_available_model() -> str:
    """Pick the best model we have an API key for, based on RDAB benchmark scores."""
    preference_order = [
        ("openai", "gpt-4.1"),
        ("anthropic", "claude-sonnet-4-6"),
        ("google", "gemini-2.5-flash"),
        ("groq", "llama-3.3-70b-versatile"),
        ("xai", "grok-3-mini"),
    ]
    for provider, model_id in preference_order:
        if _get_server_key(provider):
            return model_id
    raise HTTPException(
        status_code=status.HTTP_424_FAILED_DEPENDENCY,
        detail="No provider API keys configured. Set at least one in environment.",
    )


# ─── Core proxy handler ───────────────────────────────────────────────────────

@router.post(
    "",
    response_model=ProxyResponse,
    summary="Real-time LLM proxy with validity evaluation",
    response_description="LLM response + heuristic validity scorecard + cost metrics",
)
async def proxy_call(req: ProxyRequest, request: Request) -> ProxyResponse:
    """
    Drop-in LLM proxy for any agent or application.

    Every response is scored with a RDAB-calibrated heuristic validator before
    being returned. If validity falls below `reject_threshold`, CostGuard rejects
    the response and retries with the next model in `fallback_models`.

    For full RDAB statistical evaluation, use POST /evaluate.

    **Integration example (Python):**
    ```python
    import httpx
    resp = httpx.post("http://costguard:8000/proxy", json={
        "model_id": "gpt-4.1",
        "prompt": "Analyze this dataset...",
        "reject_threshold": 0.30,
        "fallback_models": ["claude-sonnet-4-6", "gemini-2.5-flash"],
    })
    result = resp.json()
    print(result["content"])        # the LLM response
    print(result["validity_score"]) # heuristic scorecard
    print(result["cost_usd"])       # exact cost for this call
    ```
    """
    call_id = getattr(request.state, "request_id", str(uuid.uuid4())[:12])
    start_total = time.monotonic()

    model_id = req.model_id if not req.auto_select else _pick_best_available_model()
    models_to_try = [model_id] + [m for m in req.fallback_models if m != model_id]

    last_score: RDABScoreCard | None = None
    last_content = ""
    last_in_tokens = 0
    last_out_tokens = 0
    last_latency = 0.0
    accepted = False
    fallback_used = False
    fallback_model_id: str | None = None
    rejection_reason: str | None = None
    attempts = 0
    cb_state = "unknown"  # initialized safely; updated on first provider contact

    for attempt_idx, current_model in enumerate(models_to_try):
        attempts = attempt_idx + 1

        from evaluation.pricing import MODELS
        pricing = MODELS.get(current_model)
        if not pricing:
            logger.warning(f"[{call_id}] Unknown model '{current_model}', skipping")
            continue

        cb = _circuit_registry.get(pricing.provider)
        cb_state = cb.state
        if not cb.allow_request():
            logger.warning(f"[{call_id}] Circuit OPEN for '{pricing.provider}', skipping {current_model}")
            proxy_requests_total.labels(model=current_model, provider=pricing.provider, status="circuit_open").inc()
            continue

        call_start = time.monotonic()
        try:
            content, in_tokens, out_tokens = await _call_llm_with_retry(
                model_id=current_model,
                prompt=req.prompt,
                system_prompt=req.system_prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                api_key=req.api_key,
            )
            latency_ms = (time.monotonic() - call_start) * 1000

            cb.record_success()
            _alert_engine.record_success(current_model)

            score = _score_response_fast(req.prompt, content, req.context)

            last_content = content
            last_score = score
            last_in_tokens = in_tokens
            last_out_tokens = out_tokens
            last_latency = latency_ms
            cb_state = cb.state

            proxy_requests_total.labels(model=current_model, provider=pricing.provider, status="success").inc()
            proxy_latency_seconds.labels(model=current_model, provider=pricing.provider).observe(latency_ms / 1000)

            if score.rdab_score >= req.reject_threshold:
                accepted = True
                if attempt_idx > 0:
                    fallback_used = True
                    fallback_model_id = current_model
                    proxy_fallbacks_total.labels(from_model=model_id, to_model=current_model).inc()
                break
            else:
                rejection_reason = (
                    f"Validity score {score.rdab_score:.3f} below threshold {req.reject_threshold:.3f} "
                    f"(correctness={score.correctness:.2f}, stat_validity={score.stat_validity:.2f})"
                )
                proxy_rejections_total.labels(model=current_model, reason="low_validity").inc()
                logger.warning(f"[{call_id}] Rejected {current_model}: {rejection_reason}")

                await _alert_engine.check_validity(
                    call_id=call_id,
                    model_id=current_model,
                    validity_score=score.rdab_score,
                    threshold=req.reject_threshold,
                )
                await _alert_engine.check_consecutive_low_validity(
                    call_id=call_id,
                    model_id=current_model,
                    validity_score=score.rdab_score,
                    threshold=req.reject_threshold,
                )

        except asyncio.TimeoutError:
            cb.record_failure()
            latency_ms = (time.monotonic() - call_start) * 1000
            proxy_requests_total.labels(model=current_model, provider=pricing.provider, status="timeout").inc()
            logger.error(
                f"[{call_id}] Timeout after {_LLM_TIMEOUT_SECONDS}s for {current_model} "
                f"({_RETRY_ATTEMPTS} attempts exhausted)"
            )

            if cb.state == "open":
                await _alert_engine.check_circuit_breaker_opened(
                    provider=pricing.provider,
                    failure_count=cb._failure_count,
                )

            if attempt_idx == len(models_to_try) - 1:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=(
                        f"All models timed out. Last timeout on {current_model} "
                        f"after {_LLM_TIMEOUT_SECONDS}s × {_RETRY_ATTEMPTS} attempts."
                    ),
                )
            continue

        except HTTPException:
            raise

        except Exception as exc:
            cb.record_failure()
            latency_ms = (time.monotonic() - call_start) * 1000
            proxy_requests_total.labels(model=current_model, provider=pricing.provider, status="error").inc()
            logger.error(f"[{call_id}] LLM call failed for {current_model}: {exc}")

            if cb.state == "open":
                await _alert_engine.check_circuit_breaker_opened(
                    provider=pricing.provider,
                    failure_count=cb._failure_count,
                )

            await _alert_engine.check_failure_rate(
                call_id=call_id,
                model_id=current_model,
                provider=pricing.provider,
                error=str(exc),
            )

            # Check for rate limit response and alert
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                await _alert_engine.check_rate_limit(
                    call_id=call_id,
                    model_id=current_model,
                    provider=pricing.provider,
                )

            if attempt_idx == len(models_to_try) - 1:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"All models failed. Last error on {current_model}: {exc}",
                )
            continue

    if last_score is None:
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail="No model produced a response. Check API keys and circuit breaker status.",
        )

    from evaluation.pricing import MODELS as _MODELS
    final_model = fallback_model_id or model_id
    final_pricing = _MODELS.get(final_model)
    cost_usd = final_pricing.estimate_cost(last_in_tokens, last_out_tokens) if final_pricing else 0.0

    await _alert_engine.check_cost_spike(
        call_id=call_id,
        model_id=final_model,
        cost_usd=cost_usd,
    )

    # Offload synchronous SQLite write to thread pool — don't block event loop
    try:
        await asyncio.to_thread(log_proxy_call, {
            "call_id": call_id,
            "model_id": final_model,
            "accepted": accepted,
            "validity_score": last_score.rdab_score,
            "cost_usd": cost_usd,
            "latency_ms": last_latency,
            "input_tokens": last_in_tokens,
            "output_tokens": last_out_tokens,
            "fallback_used": fallback_used,
            "attempts": attempts,
        })
    except Exception as _obs_err:
        logger.warning(f"[{call_id}] Proxy observability logging failed: {_obs_err}")

    return ProxyResponse(
        call_id=call_id,
        model_id=final_model,
        provider=final_pricing.provider if final_pricing else "unknown",
        content=last_content,
        accepted=accepted,
        rejection_reason=rejection_reason if not accepted else None,
        fallback_used=fallback_used,
        fallback_model_id=fallback_model_id,
        validity_score=last_score,
        latency_ms=round(last_latency, 1),
        input_tokens=last_in_tokens,
        output_tokens=last_out_tokens,
        cost_usd=round(cost_usd, 8),
        attempts=attempts,
        circuit_breaker_state=cb_state,
    )


@router.get("/status", summary="Proxy health: circuit breaker states")
async def proxy_status() -> dict:
    """Returns circuit breaker state for all configured providers."""
    return {
        "circuit_breakers": _circuit_registry.status_all(),
        "configured_providers": settings.available_providers,
    }
