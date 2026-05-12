# Tether–CostGuard Replay Integration — MVP Spec

**Status:** Spec only. No code exists yet.  
**Scope lock date:** 2026-05-11  
**Do not expand scope before this ships.**

---

## Goal

A user can point CostGuard at a Tether SQLite database, pick a captured run by
ID, replay all its prompts against an alternate model, and receive a JSON report
containing quality delta, cost delta, and a bootstrap confidence interval on the
quality difference.

---

## What Is and Is Not In Scope

### In scope
- Read a Tether SQLite file from a caller-supplied path (local filesystem only).
- Extract one run's prompt–response pairs from Tether's `steps` table.
- Re-call the alternate model with each original prompt (text completion, no
  system-prompt mutation).
- Score both the original response and the alternate response with the existing
  `_score_response_fast()` heuristic from `backend/proxy.py`. No new scorer.
- Bootstrap-resample the per-call score deltas (alternate − primary) to produce
  a 95 % CI.
- Return a single JSON summary document.

### Explicitly not in scope for MVP
- Streaming responses.
- Tool calls / function calling. Text completions only.
- Async LLM client (synchronous `httpx` calls are fine for MVP).
- Multi-run replay (one `run_id` per request).
- Web UI or Streamlit integration.
- Writing replay results back to any database.
- Authentication or rate limiting on the new endpoint beyond what the existing
  middleware already provides.
- Any changes to Tether's schema or code.

---

## Endpoint

```
POST /replay
Content-Type: application/json
```

### Request body

```json
{
  "tether_db_path":    "/path/to/tether.db",
  "run_id":            "abc123",
  "alternate_model":   "gpt-4.1",
  "n_bootstrap_samples": 1000
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `tether_db_path` | string | yes | Absolute path to Tether SQLite file. Must exist and be readable. |
| `run_id` | string | yes | Must match exactly one run in Tether's `steps` table. |
| `alternate_model` | string | yes | Must be a model ID present in CostGuard's `evaluation/pricing.py` catalogue. |
| `n_bootstrap_samples` | int | no | Default 1000. Range 100–10 000. |

### Response body

```json
{
  "primary_model":         "claude-sonnet-4-6",
  "alternate_model":       "gpt-4.1",
  "n_calls":               42,
  "primary_mean_score":    0.614,
  "alternate_mean_score":  0.683,
  "delta":                 0.069,
  "ci_low":               -0.012,
  "ci_high":               0.148,
  "primary_cost_usd":      0.00381,
  "alternate_cost_usd":    0.00214,
  "savings_per_call_usd":  0.000040
}
```

| Field | Notes |
|---|---|
| `primary_model` | Model recorded in the Tether run. |
| `n_calls` | Number of steps in the run that had a non-empty prompt. |
| `primary_mean_score` | Mean heuristic RDAB score across all original responses. |
| `alternate_mean_score` | Mean heuristic RDAB score across all replayed responses. |
| `delta` | `alternate_mean_score − primary_mean_score`. Positive = alternate is better. |
| `ci_low`, `ci_high` | 95 % bootstrap CI on `delta`. If the interval contains 0, the difference is not significant. |
| `primary_cost_usd` | Total cost of the original run (from Tether token counts if available, else estimated). |
| `alternate_cost_usd` | Total cost of replaying with alternate model (estimated from token counts). |
| `savings_per_call_usd` | `(primary_cost_usd − alternate_cost_usd) / n_calls`. Negative means alternate costs more. |

### Error responses

| HTTP | Condition |
|---|---|
| 400 | `tether_db_path` does not exist, or `n_bootstrap_samples` out of range. |
| 400 | `alternate_model` not in pricing catalogue. |
| 404 | `run_id` not found in Tether's `steps` table. |
| 422 | Run found but contains zero usable steps (empty prompts). |
| 424 | No API key available for the alternate model's provider. |

---

## Data Flow

```
POST /replay
      │
      ▼
1. Validate request fields (model exists, file exists, run_id non-empty)
      │
      ▼
2. Open Tether SQLite (read-only, sqlite3.connect with uri=True)
   SELECT inputs, outputs, model, input_tokens, output_tokens, cost_usd
   FROM   steps
   WHERE  run_id = ?
     AND  kind   = 'llm_call'
     AND  outputs IS NOT NULL
   ORDER  BY sequence_number ASC
      │
      ▼
3. For each row, extract:
     prompt   = json.loads(inputs)["messages"]          # list of message dicts
     response = json.loads(outputs)["choices"][0]["message"]["content"]
   Filter out rows where prompt is empty → n_calls
      │
      ▼
4. For each step:
   a. Format prompt as a single string for the scorer:
        prompt_text = " ".join(m["content"] for m in messages if m.get("content"))
   b. Score original response with _score_response_fast(prompt_text, response)
      → primary_score[i]
   c. Call alternate model via _call_llm() with messages list as-is
      (30 s timeout, no retry in MVP — keeps scope tight)
   d. Score alternate response with _score_response_fast(prompt_text, alt_response)
      → alternate_score[i]
   e. Accumulate token counts for cost calculation
      │
      ▼
5. Bootstrap CI:
   delta[i] = alternate_score[i] − primary_score[i]
   For k in range(n_bootstrap_samples):
       sample = random.choices(delta, k=n_calls)
       boot_means[k] = mean(sample)
   ci_low  = percentile(boot_means, 2.5)
   ci_high = percentile(boot_means, 97.5)
      │
      ▼
6. Return JSON response document
```

---

## Tether Schema (verified 2026-05-12)

Verified against `tether/core/storage.py` and `tether/core/models.py` in
https://github.com/patibandlavenkatamanideep/Tether.

```sql
CREATE TABLE IF NOT EXISTS steps (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    sequence_number INTEGER NOT NULL,
    kind            TEXT NOT NULL,        -- filter: kind = 'llm_call'
    provider        TEXT,                 -- nullable
    model           TEXT,                 -- nullable
    inputs          TEXT NOT NULL DEFAULT '{}',   -- JSON: full request payload
    outputs         TEXT,                 -- JSON: full response payload; NULL on failure
    input_tokens    INTEGER,              -- nullable
    output_tokens   INTEGER,              -- nullable
    cost_usd        TEXT,                 -- Decimal stored as string; nullable
    latency_ms      REAL,                 -- nullable
    error           TEXT,                 -- JSON; NULL on success
    created_at      TEXT NOT NULL,
    completed_at    TEXT
);
```

**Three things the original spec got wrong — corrected here:**

1. **No `prompt` / `response` columns.** The prompt lives inside `inputs` as a
   JSON-encoded OpenAI request dict (`inputs["messages"]`). The response lives
   inside `outputs` as a JSON-encoded OpenAI response dict
   (`outputs["choices"][0]["message"]["content"]`). Both must be parsed with
   `json.loads()`.

2. **No `step_index` column.** The ordering column is `sequence_number`.

3. **Must filter by `kind = 'llm_call'`.** Steps can be other kinds
   (checkpoints, tool calls, failures). Only `kind = 'llm_call'` steps have
   meaningful `inputs`/`outputs` for replay.

**Additional notes:**
- `cost_usd` is TEXT (Decimal string). Cast with `Decimal(row["cost_usd"])` if
  present; otherwise estimate via `evaluation/token_counter.py`.
- `run_id` in `steps` matches the `id` column in the `runs` table — both are
  UUID strings stored as TEXT.
- Use the `model` value from the first non-NULL step as `primary_model`.
- Steps where `outputs` is NULL (failed calls) are skipped — they have no
  response to score.

---

## Files Touched

| File | Change |
|---|---|
| `backend/replay.py` | New. All replay logic: DB read, LLM call, scoring, bootstrap, response assembly. |
| `backend/main.py` | Add `from backend.replay import router as replay_router` and `app.include_router(replay_router)`. Two lines only. |
| `tests/test_replay.py` | New. Unit tests (see below). |
| `docs/INTEGRATION_MVP.md` | This file. |

**No other files are modified.**

---

## Tests (`tests/test_replay.py`)

Minimum test cases to ship:

1. **Happy path** — synthetic Tether DB with 5 steps, mock `_call_llm`, assert response shape and that `delta == alternate_mean_score − primary_mean_score`.
2. **Missing DB path** — assert 400.
3. **Unknown run_id** — assert 404.
4. **All empty prompts** — assert 422.
5. **Unknown alternate model** — assert 400.
6. **Bootstrap CI contains 0** — assert `ci_low <= 0 <= ci_high` when scores are identical.
7. **Cost calculation** — assert `savings_per_call_usd == (primary_cost_usd − alternate_cost_usd) / n_calls`.

Tests use an in-memory SQLite DB seeded with the verified Tether schema
(columns: `id`, `run_id`, `sequence_number`, `kind`, `model`, `inputs`,
`outputs`, `input_tokens`, `output_tokens`, `cost_usd`, `created_at`).
No dependency on a real Tether installation.

---

## Definition of Done

- [ ] `POST /replay` returns the specified JSON for a real Tether DB with at least one run.
- [ ] All 7 test cases pass.
- [ ] `ruff check backend/replay.py tests/test_replay.py` passes.
- [ ] This spec is not modified during implementation. If scope changes, update the spec in a separate commit with a reason before changing code.
