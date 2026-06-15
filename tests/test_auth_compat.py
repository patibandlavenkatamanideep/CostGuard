"""Tests for API-key auth, the OpenAI-compatible route, and opt-in validity blocking."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.auth import settings as auth_settings
from backend.main import app

client = TestClient(app)


# ─── Auth matrix ─────────────────────────────────────────────────────────────


class TestAuth:
    def test_health_is_public(self):
        assert client.get("/health").status_code == 200

    def test_open_when_no_key_configured(self, monkeypatch):
        """Default (no COSTGUARD_API_KEY) leaves protected routes open."""
        monkeypatch.setattr(auth_settings, "costguard_api_key", None)
        assert client.get("/models").status_code == 200

    def test_protected_route_requires_key_when_configured(self, monkeypatch):
        monkeypatch.setattr(auth_settings, "costguard_api_key", "secret-key")
        assert client.get("/models").status_code == 401

    def test_wrong_key_rejected(self, monkeypatch):
        monkeypatch.setattr(auth_settings, "costguard_api_key", "secret-key")
        resp = client.get("/models", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_correct_bearer_accepted(self, monkeypatch):
        monkeypatch.setattr(auth_settings, "costguard_api_key", "secret-key")
        resp = client.get("/models", headers={"Authorization": "Bearer secret-key"})
        assert resp.status_code == 200

    def test_bare_key_without_bearer_prefix_accepted(self, monkeypatch):
        monkeypatch.setattr(auth_settings, "costguard_api_key", "secret-key")
        resp = client.get("/models", headers={"Authorization": "secret-key"})
        assert resp.status_code == 200

    def test_health_public_even_with_key(self, monkeypatch):
        monkeypatch.setattr(auth_settings, "costguard_api_key", "secret-key")
        assert client.get("/health").status_code == 200


# ─── OpenAI-compatible contract ──────────────────────────────────────────────


class TestOpenAICompat:
    def test_returns_standard_chat_completion_shape(self, monkeypatch):
        monkeypatch.setattr(auth_settings, "costguard_api_key", None)
        with patch(
            "backend.openai_compat._call_llm_with_retry",
            new=AsyncMock(return_value=("Hello there!", 12, 4)),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "Hi"}]},
            )
        assert resp.status_code == 200
        body = resp.json()
        # Standard OpenAI fields the official SDK parses.
        assert body["object"] == "chat.completion"
        assert body["id"].startswith("chatcmpl-")
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "Hello there!"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"] == {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
        }

    def test_costguard_metadata_in_body_and_headers(self, monkeypatch):
        monkeypatch.setattr(auth_settings, "costguard_api_key", None)
        with patch(
            "backend.openai_compat._call_llm_with_retry",
            new=AsyncMock(return_value=("The mean is 5.2.", 10, 6)),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "mean?"}]},
            )
        assert "x-costguard-validity" in resp.headers
        assert "x-costguard-cost-usd" in resp.headers
        assert resp.headers["x-costguard-fallback-used"] == "false"
        assert "costguard" in resp.json()

    def test_streaming_returns_sse(self, monkeypatch):
        monkeypatch.setattr(auth_settings, "costguard_api_key", None)

        async def fake_stream(*args, **kwargs):
            yield ("text", "Hello")
            yield ("text", " world")
            yield ("usage", 10, 5)

        with patch("backend.openai_compat._stream_llm", new=fake_stream):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4.1",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = resp.text
        # OpenAI streaming format markers.
        assert "chat.completion.chunk" in text
        assert '"role": "assistant"' in text
        assert "Hello" in text and "world" in text
        assert '"finish_reason": "stop"' in text
        assert "[DONE]" in text
        # Usage and CostGuard metadata land in the final chunk.
        assert '"total_tokens": 15' in text
        assert "costguard" in text

    def test_streaming_unknown_model_emits_error_event(self, monkeypatch):
        monkeypatch.setattr(auth_settings, "costguard_api_key", None)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "no-such-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        # Stream opens (200) but carries an error event + [DONE].
        assert resp.status_code == 200
        assert "error" in resp.text
        assert "[DONE]" in resp.text

    def test_unknown_model_rejected(self, monkeypatch):
        monkeypatch.setattr(auth_settings, "costguard_api_key", None)
        with patch(
            "backend.openai_compat._call_llm_with_retry",
            new=AsyncMock(return_value=("x", 1, 1)),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "no-such-model", "messages": [{"role": "user", "content": "Hi"}]},
            )
        assert resp.status_code == 400

    def test_requires_auth_when_key_set(self, monkeypatch):
        monkeypatch.setattr(auth_settings, "costguard_api_key", "secret-key")
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4.1", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 401


# ─── Refusal is not auto-rejected under default (enforce=false) config ────────


class TestRefusalNotRejected:
    def test_refusal_accepted_by_default(self, monkeypatch):
        """A correct refusal scores low on the lexical filter, but with enforce=false
        (the default) it must still be accepted, not retried onto a fallback."""
        monkeypatch.setattr(auth_settings, "costguard_api_key", None)
        refusal = "I cannot help with that request as it is unanswerable."
        with patch(
            "backend.proxy._call_llm_with_retry",
            new=AsyncMock(return_value=(refusal, 8, 12)),
        ):
            resp = client.post(
                "/proxy",
                json={
                    "model_id": "gpt-4.1",
                    "prompt": "Do something impossible",
                    "fallback_models": ["claude-sonnet-4-6"],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] is True
        assert body["fallback_used"] is False
        assert body["rejection_reason"] is None
        # The refusal is penalized on correctness (the lexical false-positive),
        # but with enforce=false it is never rejected or retried onto a fallback.
        assert body["validity_score"]["correctness"] < 0.75

    def test_low_validity_accepted_by_default(self, monkeypatch):
        """Even a genuinely low-scoring response is accepted when enforce=false."""
        monkeypatch.setattr(auth_settings, "costguard_api_key", None)
        with patch(
            "backend.proxy._call_llm_with_retry",
            new=AsyncMock(return_value=("", 4, 0)),  # empty → score 0.0
        ):
            resp = client.post(
                "/proxy",
                json={
                    "model_id": "gpt-4.1",
                    "prompt": "Analyze something",
                    "fallback_models": ["claude-sonnet-4-6"],
                },
            )
        body = resp.json()
        assert body["accepted"] is True
        assert body["fallback_used"] is False
        assert body["validity_score"]["rdab_score"] == 0.0

    def test_enforce_true_triggers_fallback_on_low_validity(self, monkeypatch):
        """With enforce=true, a low-scoring response triggers the fallback chain."""
        monkeypatch.setattr(auth_settings, "costguard_api_key", None)
        good = "The answer is 42 ± 3 (95% CI), p < 0.05. Approximately correct."

        async def _fake(model_id, prompt, system_prompt, max_tokens, temperature, api_key):
            return ("", 4, 0) if model_id == "gpt-4.1" else (good, 8, 20)

        with patch("backend.proxy._call_llm_with_retry", new=_fake):
            resp = client.post(
                "/proxy",
                json={
                    "model_id": "gpt-4.1",
                    "prompt": "Analyze something",
                    "enforce": True,
                    "reject_threshold": 0.30,
                    "fallback_models": ["claude-sonnet-4-6"],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["fallback_used"] is True
        assert body["model_id"] == "claude-sonnet-4-6"
        assert body["accepted"] is True
