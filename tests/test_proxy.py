"""Unit tests for proxy, circuit breaker, alerting, and middleware."""

from __future__ import annotations

import asyncio
import time

import pytest

from backend.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from backend.alerting import AlertEngine
from backend.proxy import _score_response_fast
from backend.middleware import _TokenBucket, _LRUBucketDict


# ─── Circuit Breaker ─────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(provider="openai")
        assert cb.state == "closed"
        assert cb.allow_request()

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(provider="openai", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert not cb.allow_request()

    def test_does_not_open_below_threshold(self):
        cb = CircuitBreaker(provider="openai", failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == "closed"
        assert cb.allow_request()

    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(provider="openai", failure_threshold=1, timeout_seconds=0.01)
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.02)
        assert cb.state == "half_open"
        assert cb.allow_request()

    def test_closes_after_successes_in_half_open(self):
        cb = CircuitBreaker(provider="openai", failure_threshold=1, timeout_seconds=0.01, success_threshold=2)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == "half_open"
        cb.record_success()
        cb.record_success()
        assert cb.state == "closed"

    def test_reopens_on_failure_in_half_open(self):
        cb = CircuitBreaker(provider="openai", failure_threshold=1, timeout_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == "half_open"
        cb.record_failure()
        assert cb.state == "open"

    def test_success_decrements_failure_count(self):
        cb = CircuitBreaker(provider="openai", failure_threshold=5)
        for _ in range(3):
            cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 2

    def test_registry_returns_same_instance(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get("anthropic")
        cb2 = reg.get("anthropic")
        assert cb1 is cb2

    def test_registry_separate_per_provider(self):
        reg = CircuitBreakerRegistry()
        cb_a = reg.get("anthropic")
        cb_o = reg.get("openai")
        assert cb_a is not cb_o

    def test_status_all(self):
        reg = CircuitBreakerRegistry()
        reg.get("anthropic")
        reg.get("openai")
        status = reg.status_all()
        assert "anthropic" in status
        assert "openai" in status
        assert status["anthropic"]["state"] == "closed"


# ─── Heuristic Validity Scorer ───────────────────────────────────────────────

class TestScoreResponseFast:
    def test_empty_response_scores_zero(self):
        sc = _score_response_fast("q", "")
        assert sc.rdab_score == 0.0
        assert sc.simulated is True

    def test_very_short_response_scores_zero(self):
        sc = _score_response_fast("q", "yes")
        assert sc.rdab_score == 0.0

    def test_error_response_penalized(self):
        sc = _score_response_fast("q", "I cannot help with that request.")
        assert sc.correctness < 0.75

    def test_stat_markers_boost_score(self):
        base = _score_response_fast("q", "The answer is 42.")
        with_stats = _score_response_fast("q", "The answer is 42 ± 3 (95% CI), p < 0.05.")
        assert with_stats.stat_validity > base.stat_validity

    def test_verbose_response_lower_efficiency(self):
        concise = _score_response_fast("q", "The mean is 5.2.")
        verbose = _score_response_fast("q", ("word " * 600).strip())
        assert concise.efficiency > verbose.efficiency

    def test_score_in_bounds(self):
        for text in ["hello world", "I cannot do that", "p < 0.05, 95% CI: [1.2, 3.4]", ""]:
            sc = _score_response_fast("prompt", text)
            assert 0.0 <= sc.rdab_score <= 1.0

    def test_all_fields_populated(self):
        sc = _score_response_fast("q", "The value is 42.")
        assert sc.correctness >= 0
        assert sc.code_quality >= 0
        assert sc.efficiency >= 0
        assert sc.stat_validity >= 0
        assert sc.token_count >= 0
        assert sc.step_count == 1


# ─── Alerting Engine ─────────────────────────────────────────────────────────

class TestAlertEngine:
    def test_validity_no_alert_above_threshold(self):
        engine = AlertEngine()
        fired = []
        engine._fire = lambda a: fired.append(a) or asyncio.coroutine(lambda: None)()
        asyncio.run(engine.check_validity("id", "gpt-4.1", 0.8, 0.30))
        assert len(fired) == 0

    def test_cost_spike_no_alert_insufficient_history(self):
        engine = AlertEngine()
        fired_alerts = []

        async def fake_fire(a):
            fired_alerts.append(a)

        engine._fire = fake_fire
        # Only 3 data points — minimum is 5 before spike detection
        asyncio.run(engine.check_cost_spike("id", "gpt-4.1", 0.01))
        asyncio.run(engine.check_cost_spike("id", "gpt-4.1", 0.01))
        asyncio.run(engine.check_cost_spike("id", "gpt-4.1", 0.01))
        assert len(fired_alerts) == 0

    def test_record_success_resets_consecutive_low(self):
        engine = AlertEngine()
        engine._consecutive_low["gpt-4.1"] = 5
        engine.record_success("gpt-4.1")
        assert engine._consecutive_low["gpt-4.1"] == 0

    def test_record_success_adds_to_outcomes(self):
        engine = AlertEngine()
        engine.record_success("gpt-4.1")
        assert True in engine._recent_outcomes

    def test_cooldown_prevents_duplicate_alerts(self):
        import time
        engine = AlertEngine()
        engine._last_fired["validity:gpt-4.1"] = time.time()  # just fired
        fired_alerts = []

        async def fake_fire(a):
            fired_alerts.append(a)

        engine._fire = fake_fire
        asyncio.run(engine.check_validity("id", "gpt-4.1", 0.1, 0.30))
        assert len(fired_alerts) == 0


# ─── Token Bucket ─────────────────────────────────────────────────────────────

class TestTokenBucket:
    def test_allows_requests_within_capacity(self):
        bucket = _TokenBucket(capacity=5, refill_rate=1.0)
        for _ in range(5):
            assert bucket.consume()

    def test_rejects_when_empty(self):
        bucket = _TokenBucket(capacity=2, refill_rate=0.0)
        bucket.consume()
        bucket.consume()
        assert not bucket.consume()

    def test_refills_over_time(self):
        bucket = _TokenBucket(capacity=1, refill_rate=100.0)
        assert bucket.consume()
        assert not bucket.consume()
        time.sleep(0.02)
        assert bucket.consume()


# ─── LRU Bucket Dict ─────────────────────────────────────────────────────────

class TestLRUBucketDict:
    def test_evicts_oldest_when_full(self):
        d = _LRUBucketDict(maxsize=3, factory=lambda: _TokenBucket(10, 1.0))
        for i in range(4):
            _ = d[f"ip-{i}"]
        # ip-0 should have been evicted (oldest)
        assert len(d._data) == 3
        assert "ip-0" not in d._data

    def test_access_refreshes_lru_order(self):
        d = _LRUBucketDict(maxsize=2, factory=lambda: _TokenBucket(10, 1.0))
        _ = d["ip-a"]
        _ = d["ip-b"]
        _ = d["ip-a"]  # refreshes ip-a to most-recently-used
        _ = d["ip-c"]  # should evict ip-b, not ip-a
        assert "ip-a" in d._data
        assert "ip-b" not in d._data

    def test_returns_same_bucket_for_same_ip(self):
        d = _LRUBucketDict(maxsize=10, factory=lambda: _TokenBucket(10, 1.0))
        b1 = d["192.168.1.1"]
        b2 = d["192.168.1.1"]
        assert b1 is b2


# ─── Retry helper ─────────────────────────────────────────────────────────────

class TestIsRetryable:
    def test_429_is_retryable(self):
        from backend.proxy import _is_retryable
        assert _is_retryable(Exception("HTTP 429 Too Many Requests"))

    def test_rate_limit_is_retryable(self):
        from backend.proxy import _is_retryable
        assert _is_retryable(Exception("rate limit exceeded"))

    def test_503_is_retryable(self):
        from backend.proxy import _is_retryable
        assert _is_retryable(Exception("503 Service Unavailable"))

    def test_overloaded_is_retryable(self):
        from backend.proxy import _is_retryable
        assert _is_retryable(Exception("model overloaded"))

    def test_timeout_not_retryable(self):
        from backend.proxy import _is_retryable
        assert not _is_retryable(asyncio.TimeoutError())

    def test_http_exception_not_retryable(self):
        from backend.proxy import _is_retryable
        from fastapi import HTTPException
        assert not _is_retryable(HTTPException(status_code=400, detail="bad"))

    def test_auth_error_not_retryable(self):
        from backend.proxy import _is_retryable
        assert not _is_retryable(Exception("401 Unauthorized"))

    def test_connection_refused_retryable(self):
        from backend.proxy import _is_retryable
        assert _is_retryable(ConnectionError("connection refused"))


# ─── Circuit Breaker State Persistence ───────────────────────────────────────

class TestCircuitBreakerPersistence:
    def test_save_and_restore_open_state(self, tmp_path, monkeypatch):
        """CB state saved in one registry should reload correctly in a new one."""
        import os
        db_path = tmp_path / "test_state.db"
        monkeypatch.setenv("COSTGUARD_DB_PATH", str(db_path))
        monkeypatch.setenv("COSTGUARD_STATE_BACKEND", "sqlite")
        # Invalidate lru_cache so settings pick up new env
        from backend.config import get_settings
        get_settings.cache_clear()

        reg1 = CircuitBreakerRegistry()
        cb1 = reg1.get("openai")
        for _ in range(5):
            cb1.record_failure()
        assert cb1.state == "open"
        reg1.save_state()

        reg2 = CircuitBreakerRegistry()
        reg2.load_state()
        cb2 = reg2.get("openai")
        assert cb2._state == "open"
        assert cb2._failure_count == 5

        get_settings.cache_clear()

    def test_save_and_restore_closed_state(self, tmp_path, monkeypatch):
        db_path = tmp_path / "test_state2.db"
        monkeypatch.setenv("COSTGUARD_DB_PATH", str(db_path))
        monkeypatch.setenv("COSTGUARD_STATE_BACKEND", "sqlite")
        from backend.config import get_settings
        get_settings.cache_clear()

        reg1 = CircuitBreakerRegistry()
        reg1.get("anthropic")  # register without any failures
        reg1.save_state()

        reg2 = CircuitBreakerRegistry()
        reg2.load_state()
        assert reg2.get("anthropic")._state == "closed"

        get_settings.cache_clear()

    def test_load_state_no_prior_data(self, tmp_path, monkeypatch):
        """load_state with an empty DB should not raise and leave CB at defaults."""
        db_path = tmp_path / "empty.db"
        monkeypatch.setenv("COSTGUARD_DB_PATH", str(db_path))
        monkeypatch.setenv("COSTGUARD_STATE_BACKEND", "sqlite")
        from backend.config import get_settings
        get_settings.cache_clear()

        reg = CircuitBreakerRegistry()
        reg.load_state()  # must not raise
        assert reg.get("openai").state == "closed"

        get_settings.cache_clear()


# ─── Alert Engine State Persistence ──────────────────────────────────────────

class TestAlertEnginePersistence:
    def test_save_and_restore_cooldowns(self, tmp_path, monkeypatch):
        db_path = tmp_path / "alert_state.db"
        monkeypatch.setenv("COSTGUARD_DB_PATH", str(db_path))
        monkeypatch.setenv("COSTGUARD_STATE_BACKEND", "sqlite")
        from backend.config import get_settings
        get_settings.cache_clear()

        eng1 = AlertEngine()
        eng1._last_fired["validity:gpt-4.1"] = 1234567890.0
        eng1._consecutive_low["gpt-4.1"] = 3
        eng1.save_state()

        eng2 = AlertEngine()
        eng2.load_state()
        assert eng2._last_fired.get("validity:gpt-4.1") == 1234567890.0
        assert eng2._consecutive_low.get("gpt-4.1") == 3

        get_settings.cache_clear()

    def test_load_state_disabled_backend(self, monkeypatch):
        monkeypatch.setenv("COSTGUARD_STATE_BACKEND", "none")
        eng = AlertEngine()
        eng.load_state()   # must not raise or write anything
        eng.save_state()   # must not raise
