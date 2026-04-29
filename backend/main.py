"""
CostGuard FastAPI application.
Handles file uploads, real-time proxy, evaluation orchestration, and monitoring.
"""

from __future__ import annotations

import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.logger import configure_logging, logger
from backend.metrics import PrometheusMiddleware, metrics_endpoint, setup_otel
from backend.middleware import RateLimitMiddleware, RequestIDMiddleware, SecurityHeadersMiddleware
from backend.models import (
    ErrorResponse,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    SessionKeys,
)
from evaluation.data_loader import DataLoadError

settings = get_settings()
configure_logging(settings.log_level)


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"CostGuard API starting (env={settings.app_env}, version=0.2.0)")
    logger.info(f"Configured providers: {settings.available_providers or ['none — demo mode']}")

    # Ensure SQLite DB is initialised at startup
    try:
        from evaluation.observability import init_db
        init_db()
        logger.info("Observability DB ready")
    except Exception as exc:
        logger.warning(f"Observability DB init failed: {exc}")

    yield

    logger.info("CostGuard API shutting down")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CostGuard API",
    description=(
        "Real-time LLM reliability and cost optimization platform. "
        "Use POST /proxy to intercept and evaluate every LLM call. "
        "Use POST /evaluate for batch dataset benchmarking."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# OTEL setup (no-op if OTEL_EXPORTER_OTLP_ENDPOINT not set)
setup_otel(app)

# Middleware — order matters: first added = outermost wrapper
app.add_middleware(PrometheusMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Exception handlers ──────────────────────────────────────────────────────

@app.exception_handler(DataLoadError)
async def data_load_error_handler(request: Request, exc: DataLoadError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(error="Invalid file", detail=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    logger.error(f"[{request_id}] Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred. Please try again.",
            request_id=request_id,
        ).model_dump(),
    )


# ─── Proxy router (new) ───────────────────────────────────────────────────────

from backend.proxy import router as proxy_router  # noqa: E402
app.include_router(proxy_router)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request):
    """Prometheus metrics scrape endpoint."""
    return await metrics_endpoint(request)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Deep health check. Returns component-level status.
    Used by load balancers, Docker, and Kubernetes probes.
    """
    rdab_available = False
    try:
        import realdataagentbench  # noqa: F401
        rdab_available = True
    except ImportError:
        pass

    # Check DB connectivity
    db_ok = False
    try:
        from evaluation.observability import get_total_eval_count
        get_total_eval_count()
        db_ok = True
    except Exception:
        pass

    # Check circuit breaker states
    cb_states: dict = {}
    try:
        from backend.proxy import _circuit_registry
        cb_states = _circuit_registry.status_all()
    except Exception:
        pass

    overall = "ok" if db_ok else "degraded"

    return HealthResponse(
        status=overall,
        version="0.2.0",
        available_providers=settings.available_providers,
        environment=settings.app_env,
        rdab_available=rdab_available,
        db_ok=db_ok,
        circuit_breakers=cb_states,
    )


@app.post(
    "/evaluate",
    response_model=EvalResponse,
    tags=["Evaluation"],
    summary="Upload a dataset and get LLM recommendations",
    response_description="Model recommendations with cost estimates",
)
async def evaluate_dataset(
    file: Annotated[UploadFile, File(description="CSV or Parquet file to evaluate")],
    task_description: Annotated[
        str,
        Form(description="Describe what you want to do with this data"),
    ] = "Analyze this dataset and answer questions about it.",
    num_questions: Annotated[
        int,
        Form(description="Number of evaluation questions (1–20)", ge=1, le=20),
    ] = 5,
    anthropic_api_key: Annotated[str | None, Form()] = None,
    openai_api_key: Annotated[str | None, Form()] = None,
    groq_api_key: Annotated[str | None, Form()] = None,
    xai_api_key: Annotated[str | None, Form()] = None,
    gemini_api_key: Annotated[str | None, Form()] = None,
) -> EvalResponse:
    """
    Upload a CSV or Parquet file. CostGuard will:
    1. Parse and analyze the dataset
    2. Run a benchmark evaluation across all available LLMs using RDAB
    3. Return cost estimates and the best model recommendation

    **Modes:**
    - *Live*: Pass one or more provider API keys to run real RDAB agents.
    - *Simulation*: Omit all keys to get calibrated scores from RDAB benchmark history.

    API keys are used only for this request and are never stored or logged.
    """
    from backend.metrics import eval_requests_total, active_evaluations
    import time

    request_id = "eval-" + str(uuid.uuid4())[:8]
    content = await file.read()

    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_mb}MB limit.",
        )

    filename = file.filename or "upload.csv"
    if not filename.lower().endswith((".csv", ".parquet")):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only CSV and Parquet files are supported.",
        )

    # Truncate task_description to prevent prompt injection via long strings
    task_description = task_description[:500].strip()

    session_keys = SessionKeys(
        anthropic_api_key=anthropic_api_key or None,
        openai_api_key=openai_api_key or None,
        groq_api_key=groq_api_key or None,
        xai_api_key=xai_api_key or None,
        gemini_api_key=gemini_api_key or None,
    )
    mode_label = "live" if session_keys.has_any_key() else "simulation"

    logger.info(
        f"[{request_id}] Evaluation started: '{filename}' ({len(content)/1024:.1f} KB) "
        f"mode={mode_label} providers={session_keys.live_providers()}"
    )

    from evaluation.engine import run_evaluation

    active_evaluations.inc()
    eval_start = time.monotonic()
    try:
        result = await run_evaluation(
            file_content=content,
            filename=filename,
            file_size_bytes=len(content),
            task_description=task_description,
            num_questions=num_questions,
            session_keys=session_keys,
        )
        eval_requests_total.labels(mode=mode_label, status="success").inc()
        from backend.metrics import eval_duration_seconds
        eval_duration_seconds.labels(mode=mode_label).observe(time.monotonic() - eval_start)
    except Exception:
        eval_requests_total.labels(mode=mode_label, status="error").inc()
        raise
    finally:
        active_evaluations.dec()

    return EvalResponse(**result)


@app.get(
    "/models",
    tags=["Models"],
    summary="List all supported LLM models and their pricing",
)
async def list_models() -> dict:
    """Return the full model catalogue with pricing information."""
    from evaluation.pricing import MODELS

    return {
        "models": [
            {
                "model_id": m.model_id,
                "provider": m.provider,
                "display_name": m.display_name,
                "tier": m.tier,
                "input_per_1k_usd": m.input_per_1k,
                "output_per_1k_usd": m.output_per_1k,
                "context_window": m.context_window,
                "strengths": m.strengths,
            }
            for m in MODELS.values()
        ],
        "total": len(MODELS),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

def run() -> None:
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
