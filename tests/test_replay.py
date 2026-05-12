"""Unit tests for POST /replay endpoint."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

# ─── Helpers ──────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS steps (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    model           TEXT,
    inputs          TEXT NOT NULL DEFAULT '{}',
    outputs         TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        TEXT,
    created_at      TEXT NOT NULL
);
"""


def _make_step(
    step_id: str,
    run_id: str,
    seq: int,
    prompt: str = "What is 2+2?",
    response: str = "The answer is 4.",
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 10,
    output_tokens: int = 20,
    cost_usd: str = "0.001",
    kind: str = "llm_call",
) -> tuple:
    inputs = json.dumps({"messages": [{"role": "user", "content": prompt}]})
    outputs = json.dumps({"choices": [{"message": {"role": "assistant", "content": response}}]})
    return (step_id, run_id, seq, kind, model, inputs, outputs, input_tokens, output_tokens, cost_usd, "2026-05-11T00:00:00")


def _seed_db(conn: sqlite3.Connection, steps: list[tuple]) -> None:
    conn.execute(_SCHEMA)
    conn.executemany(
        "INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        steps,
    )
    conn.commit()


def _temp_db(steps: list[tuple]) -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    conn = sqlite3.connect(path)
    _seed_db(conn, steps)
    conn.close()
    return path


# ─── Test cases ───────────────────────────────────────────────────────────────

class TestReplayHappyPath:
    def test_response_shape_and_delta(self):
        steps = [
            _make_step(f"s{i}", "run-1", i, response="Answer " * 5)
            for i in range(5)
        ]
        db_path = _temp_db(steps)

        mock_score = type("S", (), {"rdab_score": 0.7})()

        with (
            patch("backend.replay._call_llm", new=AsyncMock(return_value=("alt response text", 8, 15))),
            patch("backend.replay._score_response_fast", return_value=mock_score),
        ):
            resp = client.post("/replay", json={
                "tether_db_path": db_path,
                "run_id": "run-1",
                "alternate_model": "gpt-4.1",
                "n_bootstrap_samples": 100,
            })

        assert resp.status_code == 200
        body = resp.json()
        for field in (
            "primary_model", "alternate_model", "n_calls",
            "primary_mean_score", "alternate_mean_score", "delta",
            "ci_low", "ci_high", "primary_cost_usd", "alternate_cost_usd",
            "savings_per_call_usd",
        ):
            assert field in body, f"Missing field: {field}"

        assert body["n_calls"] == 5
        assert body["alternate_model"] == "gpt-4.1"
        assert abs(body["delta"] - (body["alternate_mean_score"] - body["primary_mean_score"])) < 1e-6

        Path(db_path).unlink(missing_ok=True)


class TestReplayMissingDB:
    def test_returns_400(self):
        resp = client.post("/replay", json={
            "tether_db_path": "/nonexistent/path/tether.db",
            "run_id": "run-1",
            "alternate_model": "gpt-4.1",
        })
        assert resp.status_code == 400
        assert "tether_db_path" in resp.json()["detail"].lower() or "does not exist" in resp.json()["detail"].lower()


class TestReplayUnknownRunId:
    def test_returns_404(self):
        steps = [_make_step("s0", "run-real", 0)]
        db_path = _temp_db(steps)

        with patch("backend.replay._call_llm", new=AsyncMock(return_value=("x", 1, 1))):
            resp = client.post("/replay", json={
                "tether_db_path": db_path,
                "run_id": "run-does-not-exist",
                "alternate_model": "gpt-4.1",
            })

        assert resp.status_code == 404
        Path(db_path).unlink(missing_ok=True)


class TestReplayAllEmptyPrompts:
    def test_returns_422(self):
        steps = [
            _make_step("s0", "run-1", 0, prompt=""),
            _make_step("s1", "run-1", 1, prompt="   "),
        ]
        db_path = _temp_db(steps)

        resp = client.post("/replay", json={
            "tether_db_path": db_path,
            "run_id": "run-1",
            "alternate_model": "gpt-4.1",
        })

        assert resp.status_code == 422
        Path(db_path).unlink(missing_ok=True)


class TestReplayUnknownAlternateModel:
    def test_returns_400(self):
        steps = [_make_step("s0", "run-1", 0)]
        db_path = _temp_db(steps)

        resp = client.post("/replay", json={
            "tether_db_path": db_path,
            "run_id": "run-1",
            "alternate_model": "not-a-real-model-xyz",
        })

        assert resp.status_code == 400
        assert "alternate_model" in resp.json()["detail"].lower() or "unknown" in resp.json()["detail"].lower()
        Path(db_path).unlink(missing_ok=True)


class TestReplayBootstrapCIContainsZero:
    def test_ci_contains_zero_when_scores_identical(self):
        steps = [_make_step(f"s{i}", "run-1", i) for i in range(10)]
        db_path = _temp_db(steps)

        mock_score = type("S", (), {"rdab_score": 0.5})()

        with (
            patch("backend.replay._call_llm", new=AsyncMock(return_value=("same text", 10, 10))),
            patch("backend.replay._score_response_fast", return_value=mock_score),
        ):
            resp = client.post("/replay", json={
                "tether_db_path": db_path,
                "run_id": "run-1",
                "alternate_model": "gpt-4.1",
                "n_bootstrap_samples": 200,
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["ci_low"] <= 0 <= body["ci_high"]
        Path(db_path).unlink(missing_ok=True)


class TestReplayCostCalculation:
    def test_savings_per_call(self):
        n = 4
        steps = [
            _make_step(f"s{i}", "run-1", i, cost_usd="0.002")
            for i in range(n)
        ]
        db_path = _temp_db(steps)

        mock_score = type("S", (), {"rdab_score": 0.6})()

        with (
            patch("backend.replay._call_llm", new=AsyncMock(return_value=("response", 5, 10))),
            patch("backend.replay._score_response_fast", return_value=mock_score),
        ):
            resp = client.post("/replay", json={
                "tether_db_path": db_path,
                "run_id": "run-1",
                "alternate_model": "gpt-4.1",
                "n_bootstrap_samples": 100,
            })

        assert resp.status_code == 200
        body = resp.json()
        expected_savings = (body["primary_cost_usd"] - body["alternate_cost_usd"]) / body["n_calls"]
        assert abs(body["savings_per_call_usd"] - expected_savings) < 1e-9
        Path(db_path).unlink(missing_ok=True)
