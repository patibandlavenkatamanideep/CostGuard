"""
CostGuard FastAPI application.
Handles file uploads, evaluation orchestration, and returns recommendations.
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
from backend.models import (
    ErrorResponse,
    EvalRequest,
    EvalResponse,
    HealthResponse,
)
from evaluation.data_loader import DataLoadError

settings = get_settings()
configure_logging(settings.log_level)


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"CostGuard API starting up (env={settings.app_env})")
    logger.info(f"Configured providers: {settings.available_providers or ['none — demo mode']}")
    yield
    logger.info("CostGuard API shutting down")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CostGuard API",
    description=(
        "Upload any CSV or Parquet file and instantly get the best LLM "
        "recommendation with exact cost estimates."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
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
    request_id = str(uuid.uuid4())[:8]
    logger.error(f"[{request_id}] Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred. Please try again.",
            request_id=request_id,
        ).model_dump(),
    )


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint — used by load balancers and Docker healthchecks."""
    try:
        import realdataagentbench  # noqa: F401
        rdab_available = True
    except ImportError:
        rdab_available = False

    return HealthResponse(
        status="ok",
        version="0.1.0",
        available_providers=settings.available_providers,
        environment=settings.app_env,
        rdab_available=rdab_available,
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
) -> EvalResponse:
    """
    Upload a CSV or Parquet file. CostGuard will:
    1. Parse and analyze the dataset
    2. Run a benchmark evaluation across all available LLMs
    3. Return cost estimates and the best model recommendation
    """
    # Validate file size
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_mb}MB limit.",
        )

    # Validate file type
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith((".csv", ".parquet")):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only CSV and Parquet files are supported.",
        )

    logger.info(
        f"Received file '{filename}' ({len(content)/1024:.1f} KB), "
        f"task='{task_description[:60]}'"
    )

    # Lazy import to keep startup fast
    from evaluation.engine import run_evaluation

    result = await run_evaluation(
        file_content=content,
        filename=filename,
        file_size_bytes=len(content),
        task_description=task_description,
        num_questions=num_questions,
    )

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
