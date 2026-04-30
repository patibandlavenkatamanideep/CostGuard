"""
CostGuard middleware stack.

Applied in order:
  1. RequestIDMiddleware   — injects X-Request-ID into every request/response
  2. RateLimitMiddleware   — token bucket per client IP
  3. SecurityHeadersMiddleware — adds standard security headers
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.logger import logger

_RATE_LIMIT_MAX_IPS = 10_000  # cap to prevent unbounded memory growth


class _LRUBucketDict:
    """Bounded LRU dict for per-IP token buckets. Evicts oldest when full."""

    def __init__(self, maxsize: int, factory) -> None:
        self._maxsize = maxsize
        self._factory = factory
        self._data: OrderedDict = OrderedDict()

    def __getitem__(self, key: str) -> "_TokenBucket":
        if key not in self._data:
            if len(self._data) >= self._maxsize:
                self._data.popitem(last=False)  # evict oldest
            self._data[key] = self._factory()
        self._data.move_to_end(key)
        return self._data[key]


# ─── 1. Request ID ────────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects a correlation ID into every request. Logged and returned in response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:12]
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ─── 2. Rate Limiting (token bucket per IP) ───────────────────────────────────

class _TokenBucket:
    """Simple token bucket for a single client."""

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-IP token bucket rate limiting.

    Limits:
      - /evaluate:  5 requests per minute (expensive — runs 12 model evals)
      - /proxy:     60 requests per minute (lighter — single LLM call)
      - Other:      120 requests per minute

    Configure via env vars:
      RATE_LIMIT_EVALUATE_RPM (default: 5)
      RATE_LIMIT_PROXY_RPM    (default: 60)
      RATE_LIMIT_DEFAULT_RPM  (default: 120)
    """

    import os as _os
    _EVALUATE_CAPACITY = int(_os.getenv("RATE_LIMIT_EVALUATE_RPM", "5"))
    _PROXY_CAPACITY = int(_os.getenv("RATE_LIMIT_PROXY_RPM", "60"))
    _DEFAULT_CAPACITY = int(_os.getenv("RATE_LIMIT_DEFAULT_RPM", "120"))

    # Paths exempt from rate limiting
    _EXEMPT = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}

    def __init__(self, app) -> None:
        super().__init__(app)
        self._evaluate_buckets = _LRUBucketDict(
            _RATE_LIMIT_MAX_IPS,
            lambda: _TokenBucket(self._EVALUATE_CAPACITY, self._EVALUATE_CAPACITY / 60.0),
        )
        self._proxy_buckets = _LRUBucketDict(
            _RATE_LIMIT_MAX_IPS,
            lambda: _TokenBucket(self._PROXY_CAPACITY, self._PROXY_CAPACITY / 60.0),
        )
        self._default_buckets = _LRUBucketDict(
            _RATE_LIMIT_MAX_IPS,
            lambda: _TokenBucket(self._DEFAULT_CAPACITY, self._DEFAULT_CAPACITY / 60.0),
        )

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        if path in self._EXEMPT:
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        if path.startswith("/evaluate"):
            bucket = self._evaluate_buckets[client_ip]
            limit = self._EVALUATE_CAPACITY
            window = "60s"
        elif path.startswith("/proxy"):
            bucket = self._proxy_buckets[client_ip]
            limit = self._PROXY_CAPACITY
            window = "60s"
        else:
            bucket = self._default_buckets[client_ip]
            limit = self._DEFAULT_CAPACITY
            window = "60s"

        if not bucket.consume():
            request_id = getattr(request.state, "request_id", "?")
            logger.warning(f"[{request_id}] Rate limited: {client_ip} → {path}")
            return Response(
                content=f'{{"error":"rate_limited","detail":"Too many requests. Limit: {limit} per {window}"}}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        return await call_next(request)


# ─── 3. Security Headers ──────────────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds standard security headers to every response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        # Only add HSTS on HTTPS deployments (Railway/Render set this header)
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
