"""
Prometheus metrics and OpenTelemetry setup for CostGuard.

Metrics exposed at GET /metrics (Prometheus scrape format).
OTEL traces exported to OTLP endpoint if OTEL_EXPORTER_OTLP_ENDPOINT is set.
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CollectorRegistry,
    REGISTRY,
)
from starlette.middleware.base import BaseHTTPMiddleware

from backend.logger import logger


# ─── Prometheus metric definitions ────────────────────────────────────────────

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)

# API-level metrics
api_requests_total = Counter(
    "costguard_api_requests_total",
    "Total HTTP requests by endpoint and status",
    ["endpoint", "method", "status_code"],
)
api_request_duration_seconds = Histogram(
    "costguard_api_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint", "method"],
    buckets=_LATENCY_BUCKETS,
)

# Proxy-level metrics
proxy_requests_total = Counter(
    "costguard_proxy_requests_total",
    "Total proxy calls by model, provider, and outcome",
    ["model", "provider", "status"],
)
proxy_latency_seconds = Histogram(
    "costguard_proxy_latency_seconds",
    "Proxy LLM call latency (excludes CostGuard overhead)",
    ["model", "provider"],
    buckets=_LATENCY_BUCKETS,
)
proxy_rejections_total = Counter(
    "costguard_proxy_rejections_total",
    "Total proxy responses rejected below validity threshold",
    ["model", "reason"],
)
proxy_fallbacks_total = Counter(
    "costguard_proxy_fallbacks_total",
    "Total proxy fallbacks (primary rejected, fallback used)",
    ["from_model", "to_model"],
)

# Evaluation metrics
eval_requests_total = Counter(
    "costguard_eval_requests_total",
    "Total dataset evaluation requests",
    ["mode", "status"],
)
eval_duration_seconds = Histogram(
    "costguard_eval_duration_seconds",
    "Full dataset evaluation duration",
    ["mode"],
    buckets=(1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0),
)
eval_cost_usd = Histogram(
    "costguard_eval_cost_usd",
    "Estimated cost per evaluation by model",
    ["model", "provider"],
    buckets=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0),
)
rdab_score = Gauge(
    "costguard_rdab_score",
    "Latest RDAB score per model and dimension",
    ["model", "dimension"],
)

# Alerting metrics
alerts_fired_total = Counter(
    "costguard_alerts_fired_total",
    "Total alerts fired by type and channel",
    ["alert_type", "channel"],
)

# System metrics
active_evaluations = Gauge(
    "costguard_active_evaluations",
    "Currently running dataset evaluations",
)
circuit_breaker_state = Gauge(
    "costguard_circuit_breaker_open",
    "1 if circuit breaker is open for a provider, 0 otherwise",
    ["provider"],
)


# ─── Prometheus scrape endpoint ───────────────────────────────────────────────

async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics scrape endpoint."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


# ─── Request metrics middleware ───────────────────────────────────────────────

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Tracks request count and duration for all API endpoints."""

    _SKIP_PATHS = {"/metrics", "/health", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in self._SKIP_PATHS:
            return await call_next(request)

        # Normalize path params to avoid cardinality explosion
        # e.g. /proxy/abc123 → /proxy/{id}
        normalized = _normalize_path(path)

        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start

        api_requests_total.labels(
            endpoint=normalized,
            method=request.method,
            status_code=response.status_code,
        ).inc()
        api_request_duration_seconds.labels(
            endpoint=normalized,
            method=request.method,
        ).observe(duration)

        return response


def _normalize_path(path: str) -> str:
    parts = path.strip("/").split("/")
    normalized = []
    for part in parts:
        # Replace UUIDs and hex IDs with placeholders
        if len(part) in (8, 32, 36) and all(c in "0123456789abcdef-" for c in part.lower()):
            normalized.append("{id}")
        else:
            normalized.append(part)
    return "/" + "/".join(normalized)


# ─── OTEL setup (optional — only if OTLP endpoint configured) ────────────────

def setup_otel(app: FastAPI) -> None:
    """
    Configure OpenTelemetry if OTEL_EXPORTER_OTLP_ENDPOINT is set.
    Gracefully skips if opentelemetry packages are not installed.
    """
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    service_name = os.getenv("OTEL_SERVICE_NAME", "costguard")

    if not otlp_endpoint:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — OpenTelemetry tracing disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()

        logger.info(f"OpenTelemetry tracing enabled → {otlp_endpoint} (service={service_name})")
    except ImportError:
        logger.warning(
            "OpenTelemetry packages not installed. "
            "Install opentelemetry-sdk + opentelemetry-exporter-otlp-proto-grpc to enable tracing."
        )
    except Exception as exc:
        logger.warning(f"OpenTelemetry setup failed: {exc}")
