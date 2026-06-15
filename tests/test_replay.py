"""Unit tests for POST /replay endpoint and supporting modules."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.replay import _bootstrap_ci
from evaluation.tether_reader import (
    REQUIRED_STEPS_COLUMNS,
    TetherSchemaError,
    TetherStep,
    iter_steps_for_run,
    validate_steps_schema,
)

client = TestClient(app)


# ─── DB helpers ───────────────────────────────────────────────────────────────

# Full Tether v0.1 schema — matches tether/core/storage.py exactly so
# tether_reader.iter_steps_for_run works against these test databases.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
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
    latency_ms      REAL,
    error           TEXT,
    created_at      TEXT NOT NULL,
    completed_at    TEXT,
    UNIQUE (run_id, sequence_number)
);
"""


def _make_row(
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
    latency_ms: float | None = 42.0,
    error: str | None = None,
) -> tuple:
    inputs = json.dumps({"messages": [{"role": "user", "content": prompt}]})
    outputs = json.dumps({"choices": [{"message": {"role": "assistant", "content": response}}]})
    return (
        step_id,
        run_id,
        seq,
        kind,
        model,
        inputs,
        outputs,
        input_tokens,
        output_tokens,
        cost_usd,
        latency_ms,
        error,
        "2026-05-11T00:00:00+00:00",  # created_at
        "2026-05-11T00:00:01+00:00",  # completed_at
    )


def _seed_db(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    conn.executescript(_SCHEMA)
    conn.executemany(
        "INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def _temp_db(rows: list[tuple]) -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    conn = sqlite3.connect(path)
    _seed_db(conn, rows)
    conn.close()
    return path


# ─── tether_reader tests ──────────────────────────────────────────────────────


class TestTetherReader:
    def test_yields_typed_steps(self):
        rows = [_make_row("s0", "run-1", 1, prompt="Hello?", response="Hi!")]
        path = _temp_db(rows)
        try:
            steps = list(iter_steps_for_run(path, "run-1"))
            assert len(steps) == 1
            step = steps[0]
            assert isinstance(step, TetherStep)
            assert step.step_id == "s0"
            assert step.run_id == "run-1"
            assert step.sequence_number == 1
            assert step.kind == "llm_call"
            assert step.model == "claude-sonnet-4-6"
            assert step.inputs == {"messages": [{"role": "user", "content": "Hello?"}]}
            assert step.outputs is not None
            assert step.input_tokens == 10
            assert step.output_tokens == 20
            assert step.cost_usd == Decimal("0.001")
            assert step.latency_ms == 42.0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_run_yields_nothing(self):
        rows = [_make_row("s0", "run-other", 1)]
        path = _temp_db(rows)
        try:
            steps = list(iter_steps_for_run(path, "run-target"))
            assert steps == []
        finally:
            Path(path).unlink(missing_ok=True)

    def test_ordered_by_sequence_number(self):
        # Insert in non-sequential order — reader must yield in ascending seq order.
        rows = [_make_row(f"s{i}", "run-1", i) for i in [5, 3, 1, 4, 2]]
        path = _temp_db(rows)
        try:
            steps = list(iter_steps_for_run(path, "run-1"))
            assert [s.sequence_number for s in steps] == [1, 2, 3, 4, 5]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            list(iter_steps_for_run("/nonexistent/path/tether.db", "run-1"))

    def test_completed_only_filters_in_flight_steps(self):
        rows = [_make_row("s0", "run-1", 1)]
        path = _temp_db(rows)
        conn = sqlite3.connect(path)
        conn.execute(
            "INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "s1",
                "run-1",
                2,
                "llm_call",
                "gpt-4o",
                '{"messages":[{"role":"user","content":"hi"}]}',
                None,  # no outputs yet — in-flight
                5,
                0,
                None,
                None,
                None,
                "2026-05-11T00:00:02+00:00",
                None,
            ),
        )
        conn.commit()
        conn.close()
        try:
            steps = list(iter_steps_for_run(path, "run-1", completed_only=True))
            assert len(steps) == 1
            assert steps[0].step_id == "s0"
        finally:
            Path(path).unlink(missing_ok=True)


# ─── _bootstrap_ci unit tests ─────────────────────────────────────────────────


class TestBootstrapCIUnit:
    def test_zero_deltas_ci_is_zero(self):
        """When all deltas are 0, both CI bounds must be 0."""
        lo, hi = _bootstrap_ci([0.0] * 20, n_samples=500, seed=42)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert hi == pytest.approx(0.0, abs=1e-9)

    def test_ci_brackets_the_sample_mean(self):
        """The 95% CI must contain the sample mean of the deltas."""
        rng = np.random.default_rng(0)
        deltas = rng.normal(loc=0.1, scale=0.05, size=30).tolist()
        lo, hi = _bootstrap_ci(deltas, n_samples=1000, seed=42)
        sample_mean = float(np.mean(deltas))
        assert lo <= sample_mean <= hi

    def test_same_seed_is_reproducible(self):
        """Same seed → identical CI bounds on repeated calls."""
        deltas = [0.1, -0.05, 0.2, 0.0, 0.15]
        lo1, hi1 = _bootstrap_ci(deltas, n_samples=500, seed=7)
        lo2, hi2 = _bootstrap_ci(deltas, n_samples=500, seed=7)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_known_seed_bounds_regression(self):
        """Regression: pinned bounds for seed=42 on a fixed delta list."""
        deltas = [0.1, -0.05, 0.2, 0.0, 0.15, 0.08, -0.02, 0.12, 0.05, 0.18]
        lo, hi = _bootstrap_ci(deltas, n_samples=1000, seed=42)
        # Mean ≈ 0.081; CI should be positive-ish and finite.
        assert lo <= hi
        assert lo > -0.10
        assert hi < 0.30

    def test_single_delta_returns_that_value(self):
        lo, hi = _bootstrap_ci([0.5], n_samples=100, seed=42)
        assert lo == hi == 0.5


# ─── POST /replay endpoint tests ─────────────────────────────────────────────


class TestReplayHappyPath:
    def test_response_shape_and_delta(self):
        rows = [_make_row(f"s{i}", "run-1", i, response="Answer " * 5) for i in range(5)]
        db_path = _temp_db(rows)
        mock_score = type("S", (), {"rdab_score": 0.7})()
        try:
            with (
                patch(
                    "backend.replay._call_llm",
                    new=AsyncMock(return_value=("alt response text", 8, 15)),
                ),
                patch("backend.replay._score_response_fast", return_value=mock_score),
            ):
                resp = client.post(
                    "/replay",
                    json={
                        "tether_db_path": db_path,
                        "run_id": "run-1",
                        "alternate_model": "gpt-4.1",
                        "n_bootstrap_samples": 100,
                    },
                )

            assert resp.status_code == 200
            body = resp.json()
            for field in (
                "primary_model",
                "alternate_model",
                "n_calls",
                "primary_mean_score",
                "alternate_mean_score",
                "delta",
                "ci_low",
                "ci_high",
                "primary_cost_usd",
                "alternate_cost_usd",
                "savings_per_call_usd",
            ):
                assert field in body, f"Missing field: {field}"

            assert body["n_calls"] == 5
            assert body["alternate_model"] == "gpt-4.1"
            assert (
                abs(body["delta"] - (body["alternate_mean_score"] - body["primary_mean_score"]))
                < 1e-6
            )
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestReplayMissingDB:
    def test_returns_400(self):
        resp = client.post(
            "/replay",
            json={
                "tether_db_path": "/nonexistent/path/tether.db",
                "run_id": "run-1",
                "alternate_model": "gpt-4.1",
            },
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"].lower()
        assert "tether_db_path" in detail or "does not exist" in detail or "not found" in detail


class TestReplayUnknownRunId:
    def test_returns_404(self):
        rows = [_make_row("s0", "run-real", 1)]
        db_path = _temp_db(rows)
        try:
            with patch("backend.replay._call_llm", new=AsyncMock(return_value=("x", 1, 1))):
                resp = client.post(
                    "/replay",
                    json={
                        "tether_db_path": db_path,
                        "run_id": "run-does-not-exist",
                        "alternate_model": "gpt-4.1",
                    },
                )
            assert resp.status_code == 404
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestReplayAllEmptyPrompts:
    def test_returns_422(self):
        rows = [
            _make_row("s0", "run-1", 1, prompt=""),
            _make_row("s1", "run-1", 2, prompt="   "),
        ]
        db_path = _temp_db(rows)
        try:
            resp = client.post(
                "/replay",
                json={
                    "tether_db_path": db_path,
                    "run_id": "run-1",
                    "alternate_model": "gpt-4.1",
                },
            )
            assert resp.status_code == 422
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestReplayUnknownAlternateModel:
    def test_returns_400(self):
        rows = [_make_row("s0", "run-1", 1)]
        db_path = _temp_db(rows)
        try:
            resp = client.post(
                "/replay",
                json={
                    "tether_db_path": db_path,
                    "run_id": "run-1",
                    "alternate_model": "not-a-real-model-xyz",
                },
            )
            assert resp.status_code == 400
            detail = resp.json()["detail"].lower()
            assert "alternate_model" in detail or "unknown" in detail
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestReplayBootstrapCIContainsZero:
    def test_ci_contains_zero_when_scores_identical(self):
        rows = [_make_row(f"s{i}", "run-1", i) for i in range(10)]
        db_path = _temp_db(rows)
        mock_score = type("S", (), {"rdab_score": 0.5})()
        try:
            with (
                patch(
                    "backend.replay._call_llm", new=AsyncMock(return_value=("same text", 10, 10))
                ),
                patch("backend.replay._score_response_fast", return_value=mock_score),
            ):
                resp = client.post(
                    "/replay",
                    json={
                        "tether_db_path": db_path,
                        "run_id": "run-1",
                        "alternate_model": "gpt-4.1",
                        "n_bootstrap_samples": 200,
                    },
                )

            assert resp.status_code == 200
            body = resp.json()
            assert body["ci_low"] <= 0 <= body["ci_high"]
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestReplayCostCalculation:
    def test_savings_per_call(self):
        n = 4
        rows = [_make_row(f"s{i}", "run-1", i, cost_usd="0.002") for i in range(n)]
        db_path = _temp_db(rows)
        mock_score = type("S", (), {"rdab_score": 0.6})()
        try:
            with (
                patch("backend.replay._call_llm", new=AsyncMock(return_value=("response", 5, 10))),
                patch("backend.replay._score_response_fast", return_value=mock_score),
            ):
                resp = client.post(
                    "/replay",
                    json={
                        "tether_db_path": db_path,
                        "run_id": "run-1",
                        "alternate_model": "gpt-4.1",
                        "n_bootstrap_samples": 100,
                    },
                )

            assert resp.status_code == 200
            body = resp.json()
            expected = (body["primary_cost_usd"] - body["alternate_cost_usd"]) / body["n_calls"]
            assert abs(body["savings_per_call_usd"] - expected) < 1e-9
        finally:
            Path(db_path).unlink(missing_ok=True)


# ─── Tether schema contract ───────────────────────────────────────────────────


class TestSchemaContract:
    """Guards the vendored Tether schema contract so an upstream schema change
    surfaces as a clear error instead of a cryptic SQL failure mid-replay."""

    def test_valid_schema_passes(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(_SCHEMA)
        validate_steps_schema(conn)  # should not raise
        conn.close()

    def test_missing_steps_table_raises(self):
        conn = sqlite3.connect(":memory:")  # no tables at all
        with pytest.raises(TetherSchemaError, match="no `steps` table"):
            validate_steps_schema(conn)
        conn.close()

    def test_missing_column_raises(self):
        conn = sqlite3.connect(":memory:")
        # steps table missing the cost_usd column that replay depends on.
        conn.executescript(
            """
            CREATE TABLE steps (
                id TEXT PRIMARY KEY, run_id TEXT, sequence_number INTEGER,
                kind TEXT, model TEXT, inputs TEXT, outputs TEXT,
                input_tokens INTEGER, output_tokens INTEGER,
                latency_ms REAL, error TEXT, created_at TEXT, completed_at TEXT
            );
            """
        )
        with pytest.raises(TetherSchemaError, match="cost_usd"):
            validate_steps_schema(conn)
        conn.close()

    def test_required_columns_match_select(self):
        # The contract set must contain the ordering and payload columns replay reads.
        for col in ("run_id", "kind", "sequence_number", "inputs", "outputs", "cost_usd"):
            assert col in REQUIRED_STEPS_COLUMNS

    def test_replay_endpoint_returns_400_on_bad_schema(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            conn = sqlite3.connect(path)
            # Wrong schema: steps table missing required columns.
            conn.executescript(
                "CREATE TABLE steps (id TEXT, run_id TEXT, sequence_number INTEGER, kind TEXT);"
            )
            conn.commit()
            conn.close()

            resp = client.post(
                "/replay",
                json={
                    "tether_db_path": path,
                    "run_id": "run-1",
                    "alternate_model": "gpt-4.1",
                },
            )
            assert resp.status_code == 400
            assert "Incompatible Tether schema" in resp.json()["detail"]
        finally:
            Path(path).unlink(missing_ok=True)
