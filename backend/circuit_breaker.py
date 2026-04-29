"""
Circuit breaker per LLM provider.

Prevents hammering a failing provider with retries during an outage.
State machine: CLOSED → OPEN (on N failures) → HALF_OPEN (after timeout) → CLOSED

Thread-safe via asyncio locks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


State = Literal["closed", "open", "half_open"]

_FAILURE_THRESHOLD = 5       # failures before opening
_SUCCESS_THRESHOLD = 2       # successes in HALF_OPEN before closing
_TIMEOUT_SECONDS = 60.0      # seconds before OPEN → HALF_OPEN


@dataclass
class CircuitBreaker:
    provider: str
    failure_threshold: int = _FAILURE_THRESHOLD
    success_threshold: int = _SUCCESS_THRESHOLD
    timeout_seconds: float = _TIMEOUT_SECONDS

    _state: State = field(default="closed", init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _success_count: int = field(default=0, init=False, repr=False)
    _last_failure_time: float = field(default=0.0, init=False, repr=False)

    @property
    def state(self) -> State:
        if self._state == "open":
            if time.monotonic() - self._last_failure_time >= self.timeout_seconds:
                self._state = "half_open"
                self._success_count = 0
        return self._state

    def allow_request(self) -> bool:
        s = self.state
        return s in ("closed", "half_open")

    def record_success(self) -> None:
        if self._state == "half_open":
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = "closed"
                self._failure_count = 0
        elif self._state == "closed":
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            self._success_count = 0

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "state": self.state,
            "failure_count": self._failure_count,
            "last_failure_age_s": round(time.monotonic() - self._last_failure_time, 1) if self._last_failure_time else None,
        }


class CircuitBreakerRegistry:
    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, provider: str) -> CircuitBreaker:
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker(provider=provider)
        return self._breakers[provider]

    def status_all(self) -> dict[str, dict]:
        return {p: cb.to_dict() for p, cb in self._breakers.items()}
