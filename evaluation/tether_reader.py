"""
CostGuard — Tether trace database reader.

Opens a Tether SQLite database *read-only* and yields typed TetherStep
records for a given run. Intentionally has zero dependency on the Tether
Python package — it reads the schema directly, avoiding any circular
import and keeping CostGuard deployable without Tether installed.

Schema reference: tether/core/storage.py (Tether v0.1+)
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

# ─── Vendored Tether schema contract ──────────────────────────────────────────
# CostGuard reads Tether's SQLite directly (no package dependency), so a Tether
# schema change would otherwise surface as a cryptic SQL error mid-replay. We
# pin the contract here and validate before querying. If Tether changes its
# `steps` table, bump this and re-verify against tether/core/storage.py.
TETHER_SCHEMA_VERSION = "0.1"

# Columns iter_steps_for_run selects — the replay contract depends on every one.
REQUIRED_STEPS_COLUMNS: frozenset[str] = frozenset(
    {
        "id",
        "run_id",
        "sequence_number",
        "kind",
        "model",
        "inputs",
        "outputs",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_ms",
        "error",
        "created_at",
        "completed_at",
    }
)


class TetherSchemaError(Exception):
    """Raised when a Tether database does not match the expected schema contract."""


def validate_steps_schema(conn: sqlite3.Connection) -> None:
    """Check the `steps` table exists and carries every column replay needs.

    Raises TetherSchemaError with an actionable message when the contract is
    broken, instead of letting a raw sqlite3.OperationalError leak from the query.
    """
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(steps)")}
    except sqlite3.OperationalError as exc:
        raise TetherSchemaError(f"Cannot read Tether schema: {exc}") from exc

    if not cols:
        raise TetherSchemaError(
            "Tether database has no `steps` table — not a Tether trace DB, or the "
            f"schema changed (CostGuard expects Tether schema v{TETHER_SCHEMA_VERSION})."
        )

    missing = REQUIRED_STEPS_COLUMNS - cols
    if missing:
        raise TetherSchemaError(
            f"Tether `steps` table is missing column(s): {sorted(missing)}. "
            f"CostGuard expects Tether schema v{TETHER_SCHEMA_VERSION}; re-verify "
            "evaluation/tether_reader.py against the Tether storage schema."
        )


@dataclass(frozen=True)
class TetherStep:
    """A single captured LLM call from a Tether trace database.

    Field names mirror Tether's ``steps`` table columns exactly so the
    mapping stays obvious and grep-able.
    """

    step_id: str
    run_id: str
    sequence_number: int
    kind: str
    model: str | None
    inputs: dict
    outputs: dict | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: Decimal | None
    latency_ms: float | None
    error: dict | None
    created_at: str
    completed_at: str | None


def iter_steps_for_run(
    db_path: str | Path,
    run_id: str,
    *,
    kind: str = "llm_call",
    completed_only: bool = True,
) -> Iterator[TetherStep]:
    """Yield TetherStep records for a run, ordered by sequence_number.

    Opens the database read-only (``?mode=ro`` URI flag) so it is safe
    to call while Tether is still writing to the same file on disk.

    Args:
        db_path: Absolute or relative path to the Tether ``.db`` file.
        run_id: The run UUID string from Tether's ``steps`` table.
        kind: Step kind to filter on. Defaults to ``"llm_call"`` — the
              only kind that carries prompt/response payloads.
        completed_only: When True (default) only yield steps that have
                        ``outputs IS NOT NULL`` — i.e. the call finished.

    Yields:
        :class:`TetherStep` records in ascending ``sequence_number`` order.

    Raises:
        FileNotFoundError: If ``db_path`` does not exist.
        sqlite3.OperationalError: If the file is not a valid SQLite
            database or the ``steps`` table is missing.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Tether database not found: {path}")

    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row

    # Fail fast with a clear message if Tether's schema has drifted.
    try:
        validate_steps_schema(conn)
    except TetherSchemaError:
        conn.close()
        raise

    conditions = ["run_id = ?", "kind = ?"]
    params: list[str] = [run_id, kind]
    if completed_only:
        conditions.append("outputs IS NOT NULL")

    where = " AND ".join(conditions)

    # `where` is composed only of fixed literal conditions; all runtime values are
    # bound via `params`, so this is not a SQL injection vector.
    columns = (
        "id, run_id, sequence_number, kind, model, inputs, outputs, "
        "input_tokens, output_tokens, cost_usd, latency_ms, error, created_at, completed_at"
    )
    query = f"SELECT {columns} FROM steps WHERE {where} ORDER BY sequence_number ASC"  # nosec B608
    try:
        cursor = conn.execute(query, params)
        for row in cursor:
            yield _row_to_step(row)
    finally:
        conn.close()


def _row_to_step(row: sqlite3.Row) -> TetherStep:
    """Convert a raw SQLite row to a :class:`TetherStep`."""

    def _json(v: str | None) -> dict | None:
        if v is None:
            return None
        try:
            result = json.loads(v)
            return result if isinstance(result, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    def _decimal(v: str | None) -> Decimal | None:
        if v is None:
            return None
        try:
            return Decimal(str(v))
        except InvalidOperation:
            return None

    return TetherStep(
        step_id=row["id"],
        run_id=row["run_id"],
        sequence_number=row["sequence_number"],
        kind=row["kind"],
        model=row["model"],
        inputs=_json(row["inputs"]) or {},
        outputs=_json(row["outputs"]),
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cost_usd=_decimal(row["cost_usd"]),
        latency_ms=row["latency_ms"],
        error=_json(row["error"]),
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )
