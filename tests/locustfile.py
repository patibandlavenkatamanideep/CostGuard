"""
CostGuard load test — finds the actual RPS ceiling for the /proxy endpoint.

Usage (against local server):
    locust -f tests/locustfile.py --host http://localhost:8000 --users 20 --spawn-rate 5

Usage (headless, 60-second run):
    locust -f tests/locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 10 --run-time 60s --headless \
           --csv /tmp/costguard_load

Environment variables:
    LOAD_TEST_API_KEY   — provider API key (if omitted, proxy returns 424 — still valid for latency/rate-limit testing)
    LOAD_TEST_MODEL     — model_id to hit (default: gpt-4.1)
    LOAD_TEST_HOST      — base URL (default: http://localhost:8000)

What to look for:
    - p95 latency < 200ms for the heuristic scorer path
    - RPS ceiling before 429s appear from CostGuard's rate limiter (default: 60/min/IP)
    - Zero 5xx under normal load
    - Circuit breaker status stays "closed" throughout
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task


_API_KEY = os.getenv("LOAD_TEST_API_KEY", "")
_MODEL = os.getenv("LOAD_TEST_MODEL", "gpt-4.1")

_PROMPTS = [
    "What is the mean of [1, 2, 3, 4, 5]?",
    "Summarize the key findings from a dataset with 1000 rows and 5 columns.",
    "List three statistical tests suitable for comparing two continuous variables.",
    "What does a p-value of 0.03 indicate?",
    "Explain the difference between correlation and causation in one sentence.",
    "How would you detect outliers in a numeric column?",
    "What is a 95% confidence interval?",
]


class ProxyUser(HttpUser):
    """
    Simulates a real agent application calling CostGuard as its LLM proxy.

    Task weights:
      10× /proxy  — the hot path we're benchmarking
       2× /health — lightweight probe (simulates load balancer + monitoring)
       1× /proxy/status — circuit breaker dashboard poll
    """

    wait_time = between(0.1, 0.5)  # think time between requests

    @task(10)
    def proxy_call(self) -> None:
        payload: dict = {
            "model_id": _MODEL,
            "prompt": random.choice(_PROMPTS),
            "max_tokens": 128,
            "temperature": 0.0,
            "reject_threshold": 0.10,
        }
        if _API_KEY:
            payload["api_key"] = _API_KEY

        with self.client.post(
            "/proxy",
            json=payload,
            name="/proxy [POST]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if "content" not in body:
                    resp.failure(f"Missing 'content' in response: {body!r:.200}")
                else:
                    resp.success()
            elif resp.status_code == 424:
                # No API key configured — acceptable in load test mode
                resp.success()
            elif resp.status_code == 429:
                # Rate limited by CostGuard — expected ceiling behaviour
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(2)
    def health_check(self) -> None:
        with self.client.get("/health", name="/health [GET]", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: HTTP {resp.status_code}")

    @task(1)
    def circuit_breaker_status(self) -> None:
        with self.client.get(
            "/proxy/status", name="/proxy/status [GET]", catch_response=True
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"CB status failed: HTTP {resp.status_code}")
