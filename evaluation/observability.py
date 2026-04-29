"""
CostGuard Layer 4 — Observability & Alerting

Logs every evaluation to a SQLite database, tracks historical model scores,
detects score drift (>10% drop from historical average), and optionally fires
Slack webhook alerts when drift is detected.

Storage: lightweight SQLite file at $COSTGUARD_DB_PATH (default: /tmp/costguard_history.db)
Alerts:  optional — only fires if SLACK_WEBHOOK_URL env var is set
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from backend.logger import logger

# ─── PII sanitisation ────────────────────────────────────────────────────────

def sanitize_for_logging(response: dict) -> dict:
    """
    Strip or hash PII from a serialised EvalResponse before it reaches SQLite.

    - column_names: replaced with 8-char SHA-256 prefixes (unique but opaque).
    - data_sample fields anywhere in the tree: replaced with "[REDACTED]".

    Returns a deep copy — the original dict is never mutated.
    """
    sanitized = copy.deepcopy(response)

    ds = sanitized.get("dataset_stats")
    if isinstance(ds, dict) and "column_names" in ds:
        ds["column_names"] = [
            hashlib.sha256(c.encode()).hexdigest()[:8]
            for c in ds["column_names"]
        ]

    def _redact_samples(node: Any) -> None:
        if isinstance(node, dict):
            if "data_sample" in node:
                node["data_sample"] = "[REDACTED]"
            for v in node.values():
                _redact_samples(v)
        elif isinstance(node, list):
            for item in node:
                _redact_samples(item)

    _redact_samples(sanitized)
    return sanitized


# ─── Database setup ───────────────────────────────────────────────────────────

DB_PATH = Path(os.getenv("COSTGUARD_DB_PATH", "/tmp/costguard_history.db"))

DRIFT_THRESHOLD_PCT = 10.0   # alert if score drops >10% from historical avg
MIN_HISTORY_RUNS    = 3      # need at least this many prior runs to detect drift


@contextmanager
def _db():
    import sqlite3
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL mode: allows concurrent readers while a writer is active.
    # Critical for production — without this, concurrent evaluations block each other.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables and indexes if they don't exist yet. Safe to call multiple times."""
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_id           TEXT    NOT NULL,
                timestamp         REAL    NOT NULL,
                dataset_hash      TEXT    NOT NULL,
                recommended_model TEXT    NOT NULL,
                rdab_score        REAL    NOT NULL,
                correctness       REAL    NOT NULL,
                code_quality      REAL    NOT NULL,
                efficiency        REAL    NOT NULL,
                stat_validity     REAL    NOT NULL,
                cost_usd          REAL    NOT NULL,
                eval_mode         TEXT    NOT NULL,
                simulated         INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drift_events (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      REAL    NOT NULL,
                model_id       TEXT    NOT NULL,
                historical_avg REAL    NOT NULL,
                current_score  REAL    NOT NULL,
                drop_pct       REAL    NOT NULL,
                eval_id        TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS proxy_calls (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id       TEXT    NOT NULL,
                timestamp     REAL    NOT NULL,
                model_id      TEXT    NOT NULL,
                accepted      INTEGER NOT NULL DEFAULT 1,
                validity_score REAL   NOT NULL,
                cost_usd      REAL    NOT NULL,
                latency_ms    REAL    NOT NULL,
                input_tokens  INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                fallback_used INTEGER NOT NULL DEFAULT 0,
                attempts      INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                alert_type  TEXT    NOT NULL,
                severity    TEXT    NOT NULL,
                model_id    TEXT    NOT NULL,
                message     TEXT    NOT NULL,
                call_id     TEXT    NOT NULL DEFAULT ''
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluations(recommended_model)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_ts    ON evaluations(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proxy_ts   ON proxy_calls(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proxy_model ON proxy_calls(model_id)"
        )


# ─── Logging ──────────────────────────────────────────────────────────────────

def _dataset_hash(eval_result: dict[str, Any]) -> str:
    stats = eval_result.get("dataset_stats", {})
    cols  = stats.get("column_names", [])
    fp    = f"{stats.get('rows', 0)}:{stats.get('columns', 0)}:{','.join(cols[:8])}"
    return hashlib.md5(fp.encode()).hexdigest()[:12]


def log_evaluation(eval_result: dict[str, Any]) -> None:
    """
    Persist one evaluation result to SQLite, then check for score drift.

    Called at the end of run_evaluation() in engine.py.
    Non-blocking — any storage/IO error is caught and logged, never raised.
    """
    try:
        init_db()
        rec = eval_result.get("recommended_model", {})
        sc  = rec.get("rdab_scorecard", {})

        row = {
            "eval_id":           eval_result.get("eval_id", "unknown"),
            "timestamp":         time.time(),
            "dataset_hash":      _dataset_hash(eval_result),
            "recommended_model": rec.get("model_id", "unknown"),
            "rdab_score":        sc.get("rdab_score", 0.0),
            "correctness":       sc.get("correctness", 0.0),
            "code_quality":      sc.get("code_quality", 0.0),
            "efficiency":        sc.get("efficiency", 0.0),
            "stat_validity":     sc.get("stat_validity", 0.0),
            "cost_usd":          rec.get("estimated_total_cost_usd", 0.0),
            "eval_mode":         str(eval_result.get("eval_mode", "simulation")),
            "simulated":         1 if sc.get("simulated", True) else 0,
        }

        with _db() as conn:
            conn.execute(
                """
                INSERT INTO evaluations
                    (eval_id, timestamp, dataset_hash, recommended_model,
                     rdab_score, correctness, code_quality, efficiency,
                     stat_validity, cost_usd, eval_mode, simulated)
                VALUES
                    (:eval_id, :timestamp, :dataset_hash, :recommended_model,
                     :rdab_score, :correctness, :code_quality, :efficiency,
                     :stat_validity, :cost_usd, :eval_mode, :simulated)
                """,
                row,
            )

        logger.info(
            f"[observability] Logged eval {row['eval_id']} — "
            f"{row['recommended_model']} RDAB={row['rdab_score']:.3f}"
        )

        _check_and_record_drift(row)

    except Exception as exc:
        logger.warning(f"[observability] Failed to log evaluation: {exc}")


# ─── Drift detection ──────────────────────────────────────────────────────────

def _check_and_record_drift(row: dict[str, Any]) -> None:
    """
    Compare the just-logged score against the historical average for that model.
    If it dropped >DRIFT_THRESHOLD_PCT%, record a drift event and fire an alert.
    """
    model_id      = row["recommended_model"]
    current_score = row["rdab_score"]
    eval_id       = row["eval_id"]

    try:
        with _db() as conn:
            hist = conn.execute(
                """
                SELECT AVG(rdab_score) AS avg_score, COUNT(*) AS cnt
                FROM   evaluations
                WHERE  recommended_model = ?
                  AND  eval_id           != ?
                """,
                (model_id, eval_id),
            ).fetchone()

        if hist["cnt"] < MIN_HISTORY_RUNS:
            return  # not enough prior runs

        historical_avg = hist["avg_score"]
        if historical_avg == 0:
            return

        drop_pct = (historical_avg - current_score) / historical_avg * 100
        if drop_pct <= DRIFT_THRESHOLD_PCT:
            return

        drift_row = {
            "timestamp":      time.time(),
            "model_id":       model_id,
            "historical_avg": historical_avg,
            "current_score":  current_score,
            "drop_pct":       drop_pct,
            "eval_id":        eval_id,
        }
        with _db() as conn:
            conn.execute(
                """
                INSERT INTO drift_events
                    (timestamp, model_id, historical_avg, current_score, drop_pct, eval_id)
                VALUES
                    (:timestamp, :model_id, :historical_avg, :current_score, :drop_pct, :eval_id)
                """,
                drift_row,
            )

        logger.warning(
            f"[observability] DRIFT DETECTED — {model_id}: "
            f"score={current_score:.3f} vs avg={historical_avg:.3f} "
            f"(drop={drop_pct:.1f}%)"
        )
        _send_slack_alert(model_id, current_score, historical_avg, drop_pct, eval_id)

    except Exception as exc:
        logger.warning(f"[observability] Drift check failed: {exc}")


# ─── Alerting ─────────────────────────────────────────────────────────────────

def _send_slack_alert(
    model_id: str,
    current_score: float,
    historical_avg: float,
    drop_pct: float,
    eval_id: str,
) -> None:
    """
    POST a Slack block-kit message to SLACK_WEBHOOK_URL.
    No-op if the env var is not set.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return

    try:
        import urllib.request

        text = (
            f":rotating_light: *CostGuard Score Drift Alert*\n"
            f"> Model:           `{model_id}`\n"
            f"> Current score:   `{current_score:.3f}`\n"
            f"> Historical avg:  `{historical_avg:.3f}`\n"
            f"> Drop:            `{drop_pct:.1f}%`  (threshold: {DRIFT_THRESHOLD_PCT}%)\n"
            f"> Eval ID:         `{eval_id}`"
        )
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                logger.info(f"[observability] Slack alert sent for {model_id}")
            else:
                logger.warning(
                    f"[observability] Slack returned status {resp.status}"
                )
    except Exception as exc:
        logger.warning(f"[observability] Slack alert failed: {exc}")


# ─── Proxy call logging ───────────────────────────────────────────────────────

def log_proxy_call(call_data: dict) -> None:
    """Log a single proxy call to the proxy_calls table."""
    try:
        init_db()
        with _db() as conn:
            conn.execute(
                """
                INSERT INTO proxy_calls
                    (call_id, timestamp, model_id, accepted, validity_score,
                     cost_usd, latency_ms, input_tokens, output_tokens,
                     fallback_used, attempts)
                VALUES
                    (:call_id, :timestamp, :model_id, :accepted, :validity_score,
                     :cost_usd, :latency_ms, :input_tokens, :output_tokens,
                     :fallback_used, :attempts)
                """,
                {
                    "call_id": call_data.get("call_id", ""),
                    "timestamp": time.time(),
                    "model_id": call_data.get("model_id", "unknown"),
                    "accepted": 1 if call_data.get("accepted", True) else 0,
                    "validity_score": call_data.get("validity_score", 0.0),
                    "cost_usd": call_data.get("cost_usd", 0.0),
                    "latency_ms": call_data.get("latency_ms", 0.0),
                    "input_tokens": call_data.get("input_tokens", 0),
                    "output_tokens": call_data.get("output_tokens", 0),
                    "fallback_used": 1 if call_data.get("fallback_used", False) else 0,
                    "attempts": call_data.get("attempts", 1),
                },
            )
    except Exception as exc:
        logger.warning(f"[observability] Failed to log proxy call: {exc}")


def get_recent_proxy_calls(limit: int = 100) -> list[dict]:
    """Return most recent proxy calls for dashboard display."""
    init_db()
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM proxy_calls ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_proxy_stats() -> dict:
    """Aggregate proxy call statistics for dashboard."""
    init_db()
    with _db() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                    AS total_calls,
                AVG(validity_score)         AS avg_validity,
                AVG(cost_usd)               AS avg_cost_usd,
                AVG(latency_ms)             AS avg_latency_ms,
                SUM(CASE WHEN accepted=0 THEN 1 ELSE 0 END) AS rejections,
                SUM(CASE WHEN fallback_used=1 THEN 1 ELSE 0 END) AS fallbacks,
                SUM(cost_usd)               AS total_cost_usd
            FROM proxy_calls
            """
        ).fetchone()
    return dict(row) if row else {}


# ─── Query helpers (used by Streamlit dashboard) ──────────────────────────────

def get_recent_evaluations(limit: int = 50) -> list[dict]:
    """Return the most recent evaluation records, newest first."""
    init_db()
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM evaluations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_model_averages() -> list[dict]:
    """
    Return historical average scores per model.
    Only includes models with at least 2 logged runs.
    """
    init_db()
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT
                recommended_model,
                COUNT(*)           AS run_count,
                AVG(rdab_score)    AS avg_rdab,
                AVG(correctness)   AS avg_correctness,
                AVG(code_quality)  AS avg_code_quality,
                AVG(efficiency)    AS avg_efficiency,
                AVG(stat_validity) AS avg_stat_validity,
                AVG(cost_usd)      AS avg_cost,
                MAX(timestamp)     AS last_seen
            FROM   evaluations
            GROUP  BY recommended_model
            HAVING COUNT(*) >= 2
            ORDER  BY avg_rdab DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_drift_events(limit: int = 20) -> list[dict]:
    """Return the most recent drift events, newest first."""
    init_db()
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM drift_events ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_total_eval_count() -> int:
    """Return total number of logged evaluations."""
    init_db()
    with _db() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM evaluations").fetchone()
    return row["cnt"]
