"""
CostGuard Comprehensive Alerting Engine.

Alert types (all configurable via env vars):
  1. ValidityThreshold  — proxy response validity drops below threshold
  2. CostSpike          — single call cost > N× rolling average
  3. HighFailureRate    — >20% of recent calls failed
  4. ConsecutiveLowValidity — 3+ consecutive responses below threshold
  5. RateLimit          — receiving 429s from a provider
  6. CircuitBreakerOpen — a provider circuit breaker opened

Channels: Slack (webhook) + console (always) + generic webhook (optional)

Thread-safe. All state is in-memory. Non-blocking: alert failures never
propagate to the caller.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import time
from dataclasses import dataclass, field
from typing import Deque

import httpx

from backend.logger import logger


# ─── Configuration (from env vars) ───────────────────────────────────────────

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
ALERT_WEBHOOK_URL = os.getenv("COSTGUARD_ALERT_WEBHOOK_URL", "").strip()

VALIDITY_ALERT_THRESHOLD = float(os.getenv("VALIDITY_ALERT_THRESHOLD", "0.25"))
COST_SPIKE_MULTIPLIER = float(os.getenv("COST_SPIKE_MULTIPLIER", "3.0"))
FAILURE_RATE_THRESHOLD = float(os.getenv("FAILURE_RATE_THRESHOLD", "0.20"))
CONSECUTIVE_LOW_VALIDITY = int(os.getenv("CONSECUTIVE_LOW_VALIDITY", "3"))
ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", "300"))

# Rolling window size for failure rate calculation
_FAILURE_WINDOW = 100


# ─── Alert payload ────────────────────────────────────────────────────────────

@dataclass
class Alert:
    alert_type: str
    severity: str           # critical | warning | info
    model_id: str
    provider: str
    message: str
    details: dict
    call_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_slack_block(self) -> dict:
        icons = {"critical": ":rotating_light:", "warning": ":warning:", "info": ":information_source:"}
        icon = icons.get(self.severity, ":bell:")
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(self.timestamp))
        details_text = "\n".join(f"> `{k}`: `{v}`" for k, v in self.details.items())
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"{icon} CostGuard {self.severity.upper()}: {self.alert_type}"},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{self.message}*\n\n{details_text}"},
                },
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"Model: `{self.model_id}` | Call: `{self.call_id}` | {ts_str}"}],
                },
            ]
        }

    def to_webhook_payload(self) -> dict:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "model_id": self.model_id,
            "provider": self.provider,
            "message": self.message,
            "details": self.details,
            "call_id": self.call_id,
            "timestamp": self.timestamp,
        }


# ─── Alert Engine ─────────────────────────────────────────────────────────────

class AlertEngine:
    """
    Stateful alert engine. One shared instance per process.
    State is in-memory — fine for single-process deployment.
    """

    def __init__(self) -> None:
        # Cooldown tracking: alert_type -> last fired timestamp
        self._last_fired: dict[str, float] = {}

        # Rolling window of recent call outcomes for failure rate
        self._recent_outcomes: Deque[bool] = collections.deque(maxlen=_FAILURE_WINDOW)

        # Cost rolling average per model
        self._cost_history: dict[str, Deque[float]] = {}

        # Consecutive low validity counter per model
        self._consecutive_low: dict[str, int] = {}

        self._http: httpx.AsyncClient | None = None

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=5.0)
        return self._http

    def _is_on_cooldown(self, key: str) -> bool:
        last = self._last_fired.get(key, 0.0)
        return (time.time() - last) < ALERT_COOLDOWN_SECONDS

    def _mark_fired(self, key: str) -> None:
        self._last_fired[key] = time.time()

    async def _fire(self, alert: Alert) -> None:
        """Fire alert to all configured channels. Never raises."""
        try:
            # Always log to console
            log_fn = logger.critical if alert.severity == "critical" else logger.warning
            log_fn(
                f"[ALERT:{alert.alert_type}] {alert.message} | "
                f"model={alert.model_id} call={alert.call_id} details={alert.details}"
            )

            # Slack
            if SLACK_WEBHOOK_URL:
                await self._send_slack(alert)

            # Generic webhook
            if ALERT_WEBHOOK_URL:
                await self._send_webhook(alert)

            # Track in metrics
            try:
                from backend.metrics import alerts_fired_total
                channels = ["console"]
                if SLACK_WEBHOOK_URL:
                    channels.append("slack")
                if ALERT_WEBHOOK_URL:
                    channels.append("webhook")
                for ch in channels:
                    alerts_fired_total.labels(alert_type=alert.alert_type, channel=ch).inc()
            except Exception:
                pass

        except Exception as exc:
            logger.warning(f"[alerting] Failed to fire alert {alert.alert_type}: {exc}")

    async def _send_slack(self, alert: Alert) -> None:
        try:
            client = self._get_http()
            resp = await client.post(
                SLACK_WEBHOOK_URL,
                json=alert.to_slack_block(),
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                logger.warning(f"[alerting] Slack returned {resp.status_code}: {resp.text[:100]}")
        except Exception as exc:
            logger.warning(f"[alerting] Slack send failed: {exc}")

    async def _send_webhook(self, alert: Alert) -> None:
        try:
            client = self._get_http()
            await client.post(
                ALERT_WEBHOOK_URL,
                json=alert.to_webhook_payload(),
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            logger.warning(f"[alerting] Webhook send failed: {exc}")

    # ─── Alert checkers ───────────────────────────────────────────────────────

    async def check_validity(
        self,
        call_id: str,
        model_id: str,
        validity_score: float,
        threshold: float,
    ) -> None:
        """Alert when validity is critically below threshold."""
        if validity_score >= threshold:
            self._consecutive_low[model_id] = 0
            return

        self._consecutive_low[model_id] = self._consecutive_low.get(model_id, 0) + 1
        consecutive = self._consecutive_low[model_id]

        cooldown_key = f"validity:{model_id}"
        if self._is_on_cooldown(cooldown_key):
            return

        severity = "critical" if validity_score < 0.15 else "warning"
        self._mark_fired(cooldown_key)

        await self._fire(Alert(
            alert_type="ValidityThreshold",
            severity=severity,
            model_id=model_id,
            provider=_get_provider(model_id),
            message=f"{model_id} validity score {validity_score:.3f} below threshold {threshold:.3f}",
            details={
                "validity_score": f"{validity_score:.3f}",
                "threshold": f"{threshold:.3f}",
                "consecutive_rejections": consecutive,
            },
            call_id=call_id,
        ))

    async def check_consecutive_low_validity(
        self,
        call_id: str,
        model_id: str,
        validity_score: float,
        threshold: float,
    ) -> None:
        """Alert when a model has N consecutive low-validity responses."""
        consecutive = self._consecutive_low.get(model_id, 0)
        if consecutive < CONSECUTIVE_LOW_VALIDITY:
            return

        cooldown_key = f"consecutive_low:{model_id}"
        if self._is_on_cooldown(cooldown_key):
            return
        self._mark_fired(cooldown_key)

        await self._fire(Alert(
            alert_type="ConsecutiveLowValidity",
            severity="critical",
            model_id=model_id,
            provider=_get_provider(model_id),
            message=(
                f"{model_id} has {consecutive} consecutive low-validity responses. "
                "Consider switching to a fallback model."
            ),
            details={
                "consecutive_count": consecutive,
                "threshold": f"{threshold:.3f}",
                "last_score": f"{validity_score:.3f}",
            },
            call_id=call_id,
        ))

    async def check_cost_spike(
        self,
        call_id: str,
        model_id: str,
        cost_usd: float,
    ) -> None:
        """Alert when a single call cost is significantly above rolling average."""
        if cost_usd <= 0:
            return

        if model_id not in self._cost_history:
            self._cost_history[model_id] = collections.deque(maxlen=50)

        history = self._cost_history[model_id]
        history.append(cost_usd)

        if len(history) < 5:
            return  # not enough data

        avg = sum(history) / len(history)
        if avg == 0 or cost_usd < avg * COST_SPIKE_MULTIPLIER:
            return

        cooldown_key = f"cost_spike:{model_id}"
        if self._is_on_cooldown(cooldown_key):
            return
        self._mark_fired(cooldown_key)

        await self._fire(Alert(
            alert_type="CostSpike",
            severity="warning",
            model_id=model_id,
            provider=_get_provider(model_id),
            message=(
                f"{model_id} cost spike: ${cost_usd:.6f} is "
                f"{cost_usd/avg:.1f}× above rolling average ${avg:.6f}"
            ),
            details={
                "current_cost_usd": f"{cost_usd:.6f}",
                "rolling_avg_usd": f"{avg:.6f}",
                "spike_ratio": f"{cost_usd/avg:.2f}×",
                "window_size": len(history),
            },
            call_id=call_id,
        ))

    async def check_failure_rate(
        self,
        call_id: str,
        model_id: str,
        provider: str,
        error: str,
    ) -> None:
        """Alert when failure rate over recent window exceeds threshold."""
        self._recent_outcomes.append(False)  # False = failure

        if len(self._recent_outcomes) < 10:
            return

        failure_rate = self._recent_outcomes.count(False) / len(self._recent_outcomes)
        if failure_rate < FAILURE_RATE_THRESHOLD:
            return

        cooldown_key = f"failure_rate:{provider}"
        if self._is_on_cooldown(cooldown_key):
            return
        self._mark_fired(cooldown_key)

        await self._fire(Alert(
            alert_type="HighFailureRate",
            severity="critical",
            model_id=model_id,
            provider=provider,
            message=(
                f"High failure rate for {provider}: "
                f"{failure_rate:.0%} of last {len(self._recent_outcomes)} calls failed"
            ),
            details={
                "failure_rate": f"{failure_rate:.1%}",
                "window_size": len(self._recent_outcomes),
                "threshold": f"{FAILURE_RATE_THRESHOLD:.0%}",
                "last_error": error[:200],
            },
            call_id=call_id,
        ))

    def record_success(self, model_id: str) -> None:
        """Record a successful call (used for failure rate tracking)."""
        self._recent_outcomes.append(True)
        self._consecutive_low[model_id] = 0

    async def check_circuit_breaker_opened(
        self,
        provider: str,
        failure_count: int,
    ) -> None:
        """Alert when a circuit breaker opens."""
        cooldown_key = f"circuit_breaker:{provider}"
        if self._is_on_cooldown(cooldown_key):
            return
        self._mark_fired(cooldown_key)

        await self._fire(Alert(
            alert_type="CircuitBreakerOpen",
            severity="critical",
            model_id=f"{provider}/*",
            provider=provider,
            message=(
                f"Circuit breaker OPENED for {provider} after {failure_count} consecutive failures. "
                "All {provider} calls will be blocked until the breaker resets."
            ),
            details={
                "failure_count": failure_count,
                "timeout_seconds": str(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60")),
                "action": "Calls to this provider are rejected until recovery",
            },
        ))

    async def check_rate_limit(
        self,
        call_id: str,
        model_id: str,
        provider: str,
    ) -> None:
        """Alert when receiving 429 rate limit responses."""
        cooldown_key = f"rate_limit:{provider}"
        if self._is_on_cooldown(cooldown_key):
            return
        self._mark_fired(cooldown_key)

        await self._fire(Alert(
            alert_type="RateLimit",
            severity="warning",
            model_id=model_id,
            provider=provider,
            message=f"Rate limit (429) hit for {provider}. Consider reducing concurrency or upgrading quota.",
            details={
                "provider": provider,
                "action": "Automatic retry with exponential backoff will proceed",
            },
            call_id=call_id,
        ))


def _get_provider(model_id: str) -> str:
    try:
        from evaluation.pricing import MODELS
        p = MODELS.get(model_id)
        return p.provider if p else "unknown"
    except Exception:
        return "unknown"
