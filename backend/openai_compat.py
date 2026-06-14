"""
OpenAI-compatible route — POST /v1/chat/completions.

The drop-in adapter. Point the official OpenAI SDK (or anything that speaks the
Chat Completions schema) at CostGuard by changing only `base_url` and the key:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="<COSTGUARD_API_KEY>")
    client.chat.completions.create(model="gpt-4.1", messages=[...])

It wraps the same retry + circuit-breaker + lexical-scoring + fallback logic as
POST /proxy, but returns the standard ChatCompletion schema so existing clients
work unchanged. CostGuard-specific data (validity score, cost, fallback) is
returned both in response headers (x-costguard-*) and in an extra `costguard`
field on the JSON body.

Streaming: `stream: true` returns Server-Sent Events in the OpenAI delta format.
The text is scored with the lexical pre-filter and logged after the stream
completes. Streaming uses the primary model only — there is no mid-stream
fallback (a circuit-breaker check happens before the first token).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.logger import logger
from backend.metrics import (
    proxy_fallbacks_total,
    proxy_latency_seconds,
    proxy_requests_total,
)
from backend.proxy import (
    _LLM_TIMEOUT_SECONDS,
    _call_llm_with_retry,
    _circuit_registry,
    _score_response_fast,
    _stream_llm,
)
from evaluation.observability import log_proxy_call


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) when a provider omits stream usage."""
    return max(1, len(text) // 4)


router = APIRouter(prefix="/v1", tags=["OpenAI-compatible"])


# ─── Request schema (subset of the OpenAI Chat Completions API) ───────────────


class ChatMessage(BaseModel):
    role: str
    # OpenAI allows string or an array of content parts; we accept both.
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=16384)
    # Newer OpenAI clients send max_completion_tokens; honour it as an alias.
    max_completion_tokens: int | None = Field(default=None, ge=1, le=16384)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    stream: bool = False

    # ── CostGuard extensions (optional, ignored by standard clients) ──────────
    cg_enforce: bool = Field(
        default=False,
        description="Reject + fall back on low validity (default off — score-and-log).",
    )
    cg_reject_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    cg_fallback_models: list[str] = Field(default_factory=list)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _content_to_text(content: str | list | None) -> str:
    """Flatten OpenAI message content (string or content-part array) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "\n".join(parts)


def _flatten_messages(messages: list[ChatMessage]) -> tuple[str, str | None]:
    """
    Reduce an OpenAI message list to (prompt, system_prompt) for CostGuard's
    single-prompt LLM callers. System messages are concatenated; the rest of the
    conversation is rendered role-prefixed when multi-turn.
    """
    system_parts = [
        _content_to_text(m.content) for m in messages if m.role == "system" and m.content
    ]
    system_prompt = "\n\n".join(p for p in system_parts if p) or None

    convo = [m for m in messages if m.role != "system"]
    if not convo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="messages must contain at least one non-system message.",
        )
    if len(convo) == 1 and convo[0].role == "user":
        prompt = _content_to_text(convo[0].content)
    else:
        prompt = "\n\n".join(
            f"{m.role.capitalize()}: {_content_to_text(m.content)}" for m in convo if m.content
        )

    if not prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No usable text content in messages.",
        )
    return prompt, system_prompt


# ─── Streaming (SSE) ──────────────────────────────────────────────────────────


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def _sse_stream(
    req: ChatCompletionRequest,
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
    call_id: str,
) -> AsyncIterator[str]:
    """
    Emit OpenAI-style SSE chunks for a streaming completion. Scores the assembled
    text and logs the call after the stream finishes. Primary model only — a
    circuit-breaker check happens before the first token; there is no mid-stream
    fallback.
    """
    from evaluation.pricing import MODELS

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = req.model

    def _chunk(delta: dict, finish_reason: str | None = None, extra: dict | None = None) -> dict:
        body = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if extra:
            body.update(extra)
        return body

    pricing = MODELS.get(model)
    if not pricing:
        yield _sse(
            {"error": {"message": f"Unknown model '{model}'.", "type": "invalid_request_error"}}
        )
        yield "data: [DONE]\n\n"
        return

    cb = _circuit_registry.get(pricing.provider)
    if not cb.allow_request():
        proxy_requests_total.labels(
            model=model, provider=pricing.provider, status="circuit_open"
        ).inc()
        yield _sse(
            {
                "error": {
                    "message": f"Circuit open for provider '{pricing.provider}'.",
                    "type": "service_unavailable",
                }
            }
        )
        yield "data: [DONE]\n\n"
        return

    # Opening chunk announces the assistant role (OpenAI convention).
    yield _sse(_chunk({"role": "assistant"}))

    assembled: list[str] = []
    in_tok = out_tok = 0
    call_start = time.monotonic()
    try:
        async for event in _stream_llm(
            model, prompt, system_prompt, max_tokens, req.temperature, None
        ):
            if event[0] == "text":
                assembled.append(event[1])
                yield _sse(_chunk({"content": event[1]}))
            elif event[0] == "usage":
                in_tok, out_tok = event[1], event[2]
        cb.record_success()
    except Exception as exc:  # provider/stream error mid-flight
        cb.record_failure()
        proxy_requests_total.labels(model=model, provider=pricing.provider, status="error").inc()
        logger.error(f"[{call_id}] Streaming failed for {model}: {exc}")
        yield _sse(_chunk({}, finish_reason="error"))
        yield _sse({"error": {"message": str(exc), "type": "upstream_error"}})
        yield "data: [DONE]\n\n"
        return

    latency_ms = (time.monotonic() - call_start) * 1000
    full_text = "".join(assembled)
    if out_tok == 0:
        out_tok = _estimate_tokens(full_text)
    if in_tok == 0:
        in_tok = _estimate_tokens(prompt + (system_prompt or ""))

    score = _score_response_fast(prompt, full_text)
    cost_usd = pricing.estimate_cost(in_tok, out_tok)

    proxy_requests_total.labels(model=model, provider=pricing.provider, status="success").inc()
    proxy_latency_seconds.labels(model=model, provider=pricing.provider).observe(latency_ms / 1000)

    # Final chunk: finish_reason + usage + CostGuard metadata (extra field; ignored by standard clients).
    yield _sse(
        _chunk(
            {},
            finish_reason="stop",
            extra={
                "usage": {
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": in_tok + out_tok,
                },
                "costguard": {
                    "validity_score": score.rdab_score,
                    "cost_usd": round(cost_usd, 8),
                    "latency_ms": round(latency_ms, 1),
                },
            },
        )
    )
    yield "data: [DONE]\n\n"

    try:
        await asyncio.to_thread(
            log_proxy_call,
            {
                "call_id": call_id,
                "model_id": model,
                "accepted": True,
                "validity_score": score.rdab_score,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "fallback_used": False,
                "attempts": 1,
            },
        )
    except Exception as obs_err:
        logger.warning(f"[{call_id}] Streaming observability logging failed: {obs_err}")


# ─── Endpoint ─────────────────────────────────────────────────────────────────


@router.post("/chat/completions", summary="OpenAI-compatible chat completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> Response:
    """
    OpenAI-compatible chat completion. Wraps CostGuard's retry, circuit-breaker,
    lexical scoring, and (opt-in) fallback logic and returns the standard schema.
    """
    from evaluation.pricing import MODELS

    call_id = getattr(request.state, "request_id", str(uuid.uuid4())[:12])
    prompt, system_prompt = _flatten_messages(req.messages)
    max_tokens = req.max_tokens or req.max_completion_tokens or 512

    if req.stream:
        return StreamingResponse(
            _sse_stream(req, prompt, system_prompt, max_tokens, call_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    models_to_try = [req.model] + [m for m in req.cg_fallback_models if m != req.model]

    content = ""
    in_tokens = out_tokens = 0
    final_model = req.model
    fallback_used = False
    score = None
    latency_ms = 0.0

    for attempt_idx, current_model in enumerate(models_to_try):
        pricing = MODELS.get(current_model)
        if not pricing:
            if attempt_idx == len(models_to_try) - 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown model '{current_model}'. Check GET /models for valid IDs.",
                )
            continue

        cb = _circuit_registry.get(pricing.provider)
        if not cb.allow_request():
            logger.warning(
                f"[{call_id}] Circuit OPEN for '{pricing.provider}', skipping {current_model}"
            )
            proxy_requests_total.labels(
                model=current_model, provider=pricing.provider, status="circuit_open"
            ).inc()
            continue

        call_start = time.monotonic()
        try:
            content, in_tokens, out_tokens = await _call_llm_with_retry(
                model_id=current_model,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=req.temperature,
                api_key=None,  # server keys only on the OpenAI-compatible route
            )
            latency_ms = (time.monotonic() - call_start) * 1000
            cb.record_success()
            score = _score_response_fast(prompt, content)
            final_model = current_model
            fallback_used = attempt_idx > 0

            proxy_requests_total.labels(
                model=current_model, provider=pricing.provider, status="success"
            ).inc()
            proxy_latency_seconds.labels(model=current_model, provider=pricing.provider).observe(
                latency_ms / 1000
            )
            if fallback_used:
                proxy_fallbacks_total.labels(from_model=req.model, to_model=current_model).inc()

            # Validity gating is opt-in; default returns the first success.
            if not req.cg_enforce or score.rdab_score >= req.cg_reject_threshold:
                break
            logger.warning(
                f"[{call_id}] {current_model} below threshold "
                f"({score.rdab_score:.3f} < {req.cg_reject_threshold:.3f}), trying fallback"
            )
            if attempt_idx == len(models_to_try) - 1:
                break  # exhausted fallbacks — return the last (low-scoring) response
        except HTTPException:
            raise
        except TimeoutError:
            cb.record_failure()
            proxy_requests_total.labels(
                model=current_model, provider=pricing.provider, status="timeout"
            ).inc()
            if attempt_idx == len(models_to_try) - 1:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"All models timed out. Last: {current_model} after {_LLM_TIMEOUT_SECONDS}s.",
                ) from None
        except Exception as exc:
            cb.record_failure()
            proxy_requests_total.labels(
                model=current_model, provider=pricing.provider, status="error"
            ).inc()
            logger.error(f"[{call_id}] LLM call failed for {current_model}: {exc}")
            if attempt_idx == len(models_to_try) - 1:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"All models failed. Last error on {current_model}: {exc}",
                ) from exc

    if score is None:
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail="No model produced a response. Check API keys and circuit breaker status.",
        )

    final_pricing = MODELS.get(final_model)
    cost_usd = final_pricing.estimate_cost(in_tokens, out_tokens) if final_pricing else 0.0

    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": final_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
        },
        "costguard": {
            "validity_score": score.rdab_score,
            "cost_usd": round(cost_usd, 8),
            "fallback_used": fallback_used,
            "latency_ms": round(latency_ms, 1),
            "scorecard": score.model_dump(),
        },
    }

    return JSONResponse(
        content=body,
        headers={
            "x-costguard-validity": str(score.rdab_score),
            "x-costguard-cost-usd": f"{cost_usd:.8f}",
            "x-costguard-fallback-used": str(fallback_used).lower(),
            "x-costguard-model": final_model,
        },
    )
