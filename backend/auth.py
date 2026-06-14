"""
API-key authentication for CostGuard.

Every non-public route depends on `require_api_key`. The proxy and evaluation
endpoints hold or accept provider API keys, so an unauthenticated, publicly
reachable instance is an open relay against your provider credits. This module
closes that hole.

Policy (enforced together with the startup guard in backend/main.py):
  - If COSTGUARD_API_KEY is set, every protected route requires
    `Authorization: Bearer <key>`.
  - If it is unset, the app refuses to start in production unless
    COSTGUARD_ALLOW_UNAUTHENTICATED=true is set explicitly (the public-demo
    escape hatch). In that mode protected routes are open by design.
  - In development, an unset key leaves routes open for convenience.
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.config import get_settings

settings = get_settings()

# auto_error=False so we can return our own 401 shape and allow the open-demo mode.
_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def _extract_token(raw: str | None) -> str | None:
    """Pull the bearer token out of an Authorization header value."""
    if not raw:
        return None
    raw = raw.strip()
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return raw


async def require_api_key(authorization: str | None = Security(_api_key_header)) -> None:
    """
    FastAPI dependency enforcing the CostGuard API key on protected routes.

    No-op when no key is configured (dev or explicit public-demo mode — the
    startup guard has already decided that is acceptable). Otherwise requires a
    matching `Authorization: Bearer <key>` header, compared in constant time.
    """
    expected = settings.costguard_api_key
    if not expected:
        return  # open mode — startup guard already enforced the policy

    token = _extract_token(authorization)
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Send 'Authorization: Bearer <COSTGUARD_API_KEY>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )


__all__ = ["require_api_key"]
